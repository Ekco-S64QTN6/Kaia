import os
import sys
import time
import socket
import struct
import sqlite3
import logging
import threading
from collections import OrderedDict
from queue import Empty, Full, Queue
import config

logger = logging.getLogger(__name__)

def get_default_interface() -> str:
    try:
        if os.path.exists("/proc/net/route"):
            with open("/proc/net/route", "r") as f:
                lines = f.readlines()
                for line in lines[1:]:
                    parts = line.split()
                    if len(parts) >= 3 and parts[1] == "00000000": # Destination 0.0.0.0
                        iface = parts[0]
                        if os.path.exists(f"/sys/class/net/{iface}"):
                            return iface
    except Exception:
        pass

    # Fallback to the first available non-loopback interface in /sys/class/net/
    try:
        if os.path.exists("/sys/class/net"):
            for iface in os.listdir("/sys/class/net"):
                if iface != "lo" and os.path.exists(f"/sys/class/net/{iface}"):
                    return iface
    except Exception:
        pass

    return "eth0"

def load_oui() -> dict:
    oui_dict = {}
    path = os.path.join(config.THREAT_INTEL_DIR, "oui.txt")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split(None, 1)
                    if len(parts) == 2:
                        prefix = parts[0].replace(":", "").replace("-", "").upper()[:6]
                        oui_dict[prefix] = parts[1]
        except Exception as e:
            logger.error(f"Error loading OUI database: {e}")
    return oui_dict

class PassiveDiscoveryEngine:
    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def __init__(self):
        self.stop_event = threading.Event()
        self.thread = None
        self.sock = None
        self.oui = {}
        self.seen_cache = OrderedDict()  # bounded LRU: (mac, ip) -> last_seen_timestamp
        self.cache_lock = threading.Lock()
        self.seen_cache_max_entries = 4096
        self.enrichment_queue = Queue(maxsize=256)
        self.enrichment_thread = None
        self.enrichment_dropped = 0

    def start(self):
        if self.thread and self.thread.is_alive():
            return True

        self.stop_event.clear()
        # Initialize DB
        os.makedirs(os.path.dirname(config.ASSETS_DB_PATH), exist_ok=True)
        conn = None
        try:
            conn = sqlite3.connect(config.ASSETS_DB_PATH)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS assets (
                    ip TEXT,
                    mac TEXT,
                    hostname TEXT,
                    vendor TEXT,
                    detection_vector TEXT,
                    first_seen TEXT,
                    last_seen TEXT,
                    PRIMARY KEY (ip, mac)
                )
            """)
            conn.commit()
        except Exception as e:
            logger.error(f"Failed to initialize assets database: {e}")
            return False
        finally:
            try:
                if conn:
                    conn.close()
            except Exception:
                pass

        # Load OUI
        self.oui = load_oui()

        # Keep InternetDB enrichment off the packet-processing path and cap its
        # concurrency. A hostile LAN can spoof arbitrarily many MAC/IP pairs;
        # it must not be able to create one thread (or one pending task) per pair.
        if not self.enrichment_thread or not self.enrichment_thread.is_alive():
            self.enrichment_thread = threading.Thread(
                target=self._enrichment_worker, daemon=True, name="net-discovery-enrichment"
            )
            self.enrichment_thread.start()

        self.thread = threading.Thread(target=self._run, daemon=True, name="net-discovery")
        self.thread.start()
        logger.info("PassiveDiscoveryEngine started.")
        return True

    def stop(self):
        self.stop_event.set()
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None

        # These threads are daemonized so shutdown cannot hang forever on a
        # broken raw socket or remote enrichment request.
        current = threading.current_thread()
        if self.thread and self.thread is not current:
            self.thread.join(timeout=2.0)
        if self.enrichment_thread and self.enrichment_thread is not current:
            self.enrichment_thread.join(timeout=2.0)

    def _is_recent_duplicate(self, key: tuple, now: float, window_seconds: float = 300.0) -> bool:
        """Update the deduplication LRU while enforcing a hard memory ceiling."""
        with self.cache_lock:
            last_seen = self.seen_cache.get(key)
            duplicate = last_seen is not None and now - last_seen < window_seconds
            self.seen_cache[key] = now
            self.seen_cache.move_to_end(key)
            while len(self.seen_cache) > self.seen_cache_max_entries:
                self.seen_cache.popitem(last=False)
            return duplicate

    def _queue_enrichment(self, ip: str, mac: str) -> bool:
        """Queue enrichment without blocking packet ingestion; reject on saturation."""
        try:
            self.enrichment_queue.put_nowait((ip, mac))
            return True
        except Full:
            self.enrichment_dropped += 1
            # Use debug to avoid allowing a packet flood to become a log flood.
            logger.debug("InternetDB enrichment queue full; dropping enrichment for %s", ip)
            return False

    def _enrichment_worker(self):
        """Run serialized enrichment with bounded queued work."""
        while not self.stop_event.is_set():
            try:
                ip, mac = self.enrichment_queue.get(timeout=0.5)
            except Empty:
                continue
            try:
                self._enrich_shodan(ip, mac)
            except Exception:
                logger.exception("Unexpected network-asset enrichment failure")
            finally:
                self.enrichment_queue.task_done()

    def get_recent_assets(self, n=20) -> list:
        assets = []
        if not os.path.exists(config.ASSETS_DB_PATH):
            return assets
        conn = None
        try:
            conn = sqlite3.connect(config.ASSETS_DB_PATH)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT ip, mac, hostname, vendor, detection_vector, first_seen, last_seen
                FROM assets
                ORDER BY last_seen DESC LIMIT ?
            """, (n,))
            rows = cursor.fetchall()
            for r in rows:
                assets.append({
                    "ip": r[0],
                    "mac": r[1],
                    "hostname": r[2],
                    "vendor": r[3],
                    "detection_vector": r[4],
                    "first_seen": r[5],
                    "last_seen": r[6]
                })
        except Exception as e:
            logger.error(f"Error querying assets DB: {e}")
        finally:
            try:
                if conn:
                    conn.close()
            except Exception:
                pass
        return assets

    def _run(self):
        # Open raw PACKET socket
        try:
            # ETH_P_ALL is 3 (host byte order), socket requires network byte order (ntohs(3))
            self.sock = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.ntohs(3))
            self.sock.settimeout(1.0)
            
            ifname = config.DISCOVERY_INTERFACE
            if ifname == "auto":
                ifname = get_default_interface()
            logger.info(f"Binding raw socket to interface: {ifname}")
            self.sock.bind((ifname, 0))
        except Exception as e:
            logger.error(f"Failed to open raw socket for passive discovery: {e}. Requires CAP_NET_RAW.")
            return

        while not self.stop_event.is_set():
            try:
                packet, addr = self.sock.recvfrom(65535)
            except socket.timeout:
                continue
            except Exception:
                break

            if len(packet) < 14:
                continue

            # Parse ethernet header
            dst_mac, src_mac, ethertype = struct.unpack("!6s6sH", packet[:14])
            src_mac_str = ":".join(f"{b:02x}" for b in src_mac)

            if ethertype == 0x0806:  # ARP
                self._parse_arp(packet[14:], src_mac_str)
            elif ethertype == 0x0800:  # IPv4
                self._parse_ipv4(packet[14:], src_mac_str)

    def _parse_arp(self, payload: bytes, src_mac_str: str):
        if len(payload) < 28:
            return
        htype, ptype, hlen, plen, oper, sha, spa, tha, tpa = struct.unpack("!HHBBH6s4s6s4s", payload[:28])
        if oper == 1 or oper == 2:  # Request or Reply
            ip = socket.inet_ntoa(spa)
            mac = ":".join(f"{b:02x}" for b in sha)
            self._log_asset(ip, mac, "Unknown", "ARP")

    def _parse_ipv4(self, payload: bytes, src_mac_str: str):
        if len(payload) < 20:
            return
        version_ihl = payload[0]
        ihl = (version_ihl & 0x0F) * 4
        if len(payload) < ihl + 8:
            return
        protocol = payload[9]
        src_ip = socket.inet_ntoa(payload[12:16])

        if protocol == 17:  # UDP
            udp_hdr = payload[ihl:ihl+8]
            sport, dport, length, checksum = struct.unpack("!HHHH", udp_hdr)
            udp_payload = payload[ihl+8:]
            
            if dport == 5353:  # mDNS
                self._parse_mdns(udp_payload, src_ip, src_mac_str)
            elif dport == 5355:  # LLMNR
                self._parse_llmnr(udp_payload, src_ip, src_mac_str)
            elif dport == 1900:  # SSDP
                self._parse_ssdp(udp_payload, src_ip, src_mac_str)
            elif dport == 137:  # NetBIOS
                self._parse_netbios(udp_payload, src_ip, src_mac_str)

    def _parse_mdns(self, payload: bytes, ip: str, mac: str):
        name = self._parse_dns_name(payload)
        if name:
            self._log_asset(ip, mac, name, "mDNS")

    def _parse_llmnr(self, payload: bytes, ip: str, mac: str):
        name = self._parse_dns_name(payload)
        if name:
            self._log_asset(ip, mac, name, "LLMNR")

    def _parse_ssdp(self, payload: bytes, ip: str, mac: str):
        try:
            text = payload.decode("utf-8", "ignore")
            st = ""
            for line in text.split("\r\n"):
                if line.upper().startswith("ST:"):
                    st = line.split(":", 1)[1].strip()
                    break
            name = f"SSDP ({st})" if st else "SSDP Device"
            self._log_asset(ip, mac, name, "SSDP")
        except Exception:
            pass

    def _parse_netbios(self, payload: bytes, ip: str, mac: str):
        if len(payload) >= 12 + 32 + 1 and payload[12] == 0x20:
            encoded_name = payload[13:13+32]
            name = self._decode_netbios_name(encoded_name)
            if name:
                self._log_asset(ip, mac, name, "NetBIOS")

    def _parse_dns_name(self, payload: bytes, offset: int = 12) -> str:
        parts = []
        try:
            while offset < len(payload):
                length = payload[offset]
                if length == 0:
                    break
                if (length & 0xC0) == 0xC0:  # Compression pointer
                    break
                offset += 1
                if offset + length > len(payload):
                    break
                part = payload[offset:offset+length].decode("utf-8", "ignore")
                parts.append(part)
                offset += length
            return ".".join(parts)
        except Exception:
            pass
        return ""

    def _decode_netbios_name(self, encoded: bytes) -> str:
        try:
            name_bytes = bytearray()
            for i in range(0, len(encoded), 2):
                if i+1 >= len(encoded):
                    break
                c1 = encoded[i] - 0x41
                c2 = encoded[i+1] - 0x41
                name_bytes.append((c1 << 4) | c2)
            name = name_bytes.decode("utf-8", "ignore").strip()
            # NetBIOS names are padded with spaces, strip them
            return name
        except Exception:
            pass
        return ""

    def _log_asset(self, ip: str, mac: str, hostname: str, vector: str):
        now = time.time()

        # Sanitize all fields from untrusted network input (INV-004)
        from security.telemetry_sanitizer import sanitize_telemetry
        raw = {"ip": ip, "hostname": hostname}
        clean = sanitize_telemetry(raw)
        ip = clean.get("ip", "")
        hostname = clean.get("hostname", "")
        if not ip:
            return

        key = (mac, ip)
        if self._is_recent_duplicate(key, now):
            return

        prefix = mac.replace(":", "").upper()[:6]
        vendor = self.oui.get(prefix, "unknown")
        ts_str = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        # The packet parser must not spawn threads or queue work if persistence
        # failed. The asset record is durable before enrichment is requested.
        conn = None
        saved = False
        try:
            conn = sqlite3.connect(config.ASSETS_DB_PATH)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT hostname, first_seen FROM assets WHERE ip = ? AND mac = ?",
                (ip, mac),
            )
            row = cursor.fetchone()
            if row:
                db_hostname, first_seen = row
                final_hostname = db_hostname if db_hostname != "Unknown" else hostname
                cursor.execute(
                    "UPDATE assets SET hostname = ?, last_seen = ? WHERE ip = ? AND mac = ?",
                    (final_hostname, ts_str, ip, mac),
                )
            else:
                cursor.execute(
                    """
                    INSERT INTO assets (ip, mac, hostname, vendor, detection_vector, first_seen, last_seen)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (ip, mac, hostname, vendor, vector, ts_str, ts_str),
                )
            conn.commit()
            saved = True
        except Exception as e:
            logger.error(f"Failed to log asset to database: {e}")
        finally:
            try:
                if conn:
                    conn.close()
            except Exception:
                pass

        if saved:
            self._queue_enrichment(ip, mac)

    def _enrich_shodan(self, ip: str, mac: str):
        from security import threat_intel

        conn = None
        try:
            shodan = threat_intel.lookup_internetdb(ip)
            tags = shodan.get("tags", [])
            if not tags:
                return
            tags_str = ",".join(str(tag) for tag in tags[:32])
            conn = sqlite3.connect(config.ASSETS_DB_PATH, timeout=2.0)
            cursor = conn.cursor()
            cursor.execute("SELECT hostname FROM assets WHERE ip = ? AND mac = ?", (ip, mac))
            row = cursor.fetchone()
            if row:
                current_name = row[0] or "Unknown"
                if "tags:" not in current_name:
                    new_name = f"{current_name} (tags:{tags_str})"
                    cursor.execute(
                        "UPDATE assets SET hostname = ? WHERE ip = ? AND mac = ?",
                        (new_name[:256], ip, mac),
                    )
            conn.commit()
        except Exception as e:
            logger.debug("InternetDB enrichment failed for %s: %s", ip, e)
        finally:
            try:
                if conn:
                    conn.close()
            except Exception:
                pass

