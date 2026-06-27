import os
import sys
import time
import socket
import struct
import sqlite3
import logging
import threading
import config

logger = logging.getLogger(__name__)

def get_default_interface() -> str:
    try:
        with open("/proc/net/route", "r") as f:
            lines = f.readlines()
            for line in lines[1:]:
                parts = line.split()
                if len(parts) >= 3 and parts[1] == "00000000": # Destination 0.0.0.0
                    return parts[0]
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
        self.seen_cache = {}  # (mac, ip) -> last_seen_timestamp
        self.cache_lock = threading.Lock()

    def start(self):
        # Initialize DB
        os.makedirs(os.path.dirname(config.ASSETS_DB_PATH), exist_ok=True)
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
            conn.close()
        except Exception as e:
            logger.error(f"Failed to initialize assets database: {e}")
            return False

        # Load OUI
        self.oui = load_oui()

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

    def get_recent_assets(self, n=20) -> list:
        assets = []
        if not os.path.exists(config.ASSETS_DB_PATH):
            return assets
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
            conn.close()
        except Exception as e:
            logger.error(f"Error querying assets DB: {e}")
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
        key = (mac, ip)
        
        with self.cache_lock:
            last_seen = self.seen_cache.get(key, 0)
            if now - last_seen < 300:  # 5 minutes deduplication
                # Just update seen timestamp in memory
                self.seen_cache[key] = now
                return
            self.seen_cache[key] = now

        # Determine vendor from OUI
        prefix = mac.replace(":", "").upper()[:6]
        vendor = self.oui.get(prefix, "unknown")
        
        ts_str = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        
        # Write to SQLite
        try:
            conn = sqlite3.connect(config.ASSETS_DB_PATH)
            cursor = conn.cursor()
            
            # Check if exists
            cursor.execute("SELECT hostname, first_seen FROM assets WHERE ip = ? AND mac = ?", (ip, mac))
            row = cursor.fetchone()
            if row:
                db_hostname = row[0]
                first_seen = row[1]
                # Keep hostname if it's already enriched, otherwise update
                final_hostname = db_hostname if db_hostname != "Unknown" else hostname
                cursor.execute("""
                    UPDATE assets SET hostname = ?, last_seen = ?
                    WHERE ip = ? AND mac = ?
                """, (final_hostname, ts_str, ip, mac))
            else:
                cursor.execute("""
                    INSERT INTO assets (ip, mac, hostname, vendor, detection_vector, first_seen, last_seen)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (ip, mac, hostname, vendor, vector, ts_str, ts_str))
                
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to log asset to database: {e}")

        # Spawn asynchronous threat intel enrichment
        threading.Thread(target=self._enrich_shodan, args=(ip, mac), daemon=True).start()

    def _enrich_shodan(self, ip: str, mac: str):
        from security import threat_intel
        try:
            shodan = threat_intel.lookup_internetdb(ip)
            tags = shodan.get("tags", [])
            if tags:
                tags_str = ",".join(tags)
                conn = sqlite3.connect(config.ASSETS_DB_PATH)
                cursor = conn.cursor()
                # Append tags safely without duplicate appending
                cursor.execute("SELECT hostname FROM assets WHERE ip = ? AND mac = ?", (ip, mac))
                row = cursor.fetchone()
                if row:
                    curr_name = row[0]
                    if "tags:" not in curr_name:
                        new_name = f"{curr_name} (tags:{tags_str})"
                        cursor.execute("UPDATE assets SET hostname = ? WHERE ip = ? AND mac = ?", (new_name, ip, mac))
                conn.commit()
                conn.close()
        except Exception:
            pass
