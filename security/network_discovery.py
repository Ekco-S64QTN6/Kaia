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

# Passive discovery must not let spoofed IP/MAC pairs grow persistent state forever.
ASSET_DB_MAX_ENTRIES = 10000
ASSET_DB_PRUNE_INTERVAL = 100

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
        self.asset_db_max_entries = ASSET_DB_MAX_ENTRIES
        self.asset_db_prune_interval = ASSET_DB_PRUNE_INTERVAL
        self._assets_since_prune = 0

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
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_assets_last_seen ON assets(last_seen)"
            )
            # Trim any oversized database from a previous version on startup.
            self._prune_asset_database(conn)
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
        # Validate the IPv4 header and length before interpreting transport bytes.
        # Captured Ethernet frames can be truncated or contain non-initial fragments;
        # neither should be mistaken for a complete UDP datagram.
        if len(payload) < 20 or payload[0] >> 4 != 4:
            return
        version_ihl = payload[0]
        ihl = (version_ihl & 0x0F) * 4
        total_length = struct.unpack("!H", payload[2:4])[0]
        if ihl < 20 or total_length < ihl or total_length > len(payload):
            return
        if total_length < ihl + 8:
            return
        fragment_field = struct.unpack("!H", payload[6:8])[0]
        # This collector does not reassemble IP fragments. Reject the first
        # fragment (MF), later fragments (non-zero offset), and the reserved
        # flag rather than treating an incomplete datagram as complete UDP.
        # The DF flag (0x4000) is not a fragmentation indicator and is allowed.
        if fragment_field & 0xBFFF:
            return

        protocol = payload[9]
        if protocol != 17:  # UDP only
            return
        src_ip = socket.inet_ntoa(payload[12:16])
        udp_segment = payload[ihl:total_length]
        sport, dport, udp_length, _checksum = struct.unpack("!HHHH", udp_segment[:8])
        if udp_length < 8 or udp_length > len(udp_segment):
            return
        udp_payload = udp_segment[8:udp_length]

        if dport == 5353:  # mDNS queries / responses
            self._parse_mdns(udp_payload, src_ip, src_mac_str)
        elif dport == 5355:  # LLMNR queries / responses
            self._parse_llmnr(udp_payload, src_ip, src_mac_str)
        elif dport == 1900:  # SSDP announcements / search requests
            self._parse_ssdp(udp_payload, src_ip, src_mac_str)

        # NBNS requests use destination port 137, while ordinary unicast responses
        # originate at source port 137 and target an ephemeral client port. Only
        # response names are associated with the source asset below: a query's
        # QNAME describes the name being requested, not the querier's hostname.
        if sport == 137 or dport == 137:
            self._parse_netbios(
                udp_payload,
                src_ip,
                src_mac_str,
                source_port=sport,
            )

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

    def _parse_netbios(
        self,
        payload: bytes,
        ip: str,
        mac: str,
        *,
        source_port: int = 137,
    ):
        """Parse bounded NBNS answer records without trusting packet names blindly."""
        if source_port != 137 or len(payload) < 12:
            return

        try:
            _transaction_id, flags, question_count, answer_count, authority_count, additional_count = (
                struct.unpack_from("!HHHHHH", payload, 0)
            )
        except struct.error:
            return

        # Only successful query responses are useful for passive name discovery.
        if not (flags & 0x8000) or (flags & 0x7800) or (flags & 0x000F):
            return
        record_count = answer_count + authority_count + additional_count
        if question_count > 64 or record_count > 128:
            return

        offset = 12
        for _ in range(question_count):
            _labels, next_offset = self._read_dns_name(payload, offset)
            if next_offset is None or next_offset + 4 > len(payload):
                return
            offset = next_offset + 4  # QTYPE and QCLASS

        discovered_names = []
        for record_index in range(record_count):
            labels, record_offset = self._read_dns_name(payload, offset)
            if labels is None or record_offset is None or record_offset + 10 > len(payload):
                return

            try:
                record_type, record_class, _ttl, data_length = struct.unpack_from(
                    "!HHIH", payload, record_offset
                )
            except struct.error:
                return

            data_offset = record_offset + 10
            data_end = data_offset + data_length
            if data_end > len(payload):
                return

            # Only answer-section names describe the responder's answer to
            # this query. Authority/additional records are parsed for framing,
            # but are not used to create passive asset labels.
            if record_index < answer_count and record_class == 0x0001:  # IN answer
                if record_type == 0x0020 and labels:
                    # NB RDATA is NB_FLAGS (2 bytes) plus one or more IPv4 addresses.
                    if data_length < 6 or (data_length - 2) % 4:
                        return
                    # NB answer: RR_NAME is the first-level encoded owner name.
                    name = self._decode_netbios_name(labels[0])
                    if name:
                        discovered_names.append(name)
                elif record_type == 0x0021:  # NBSTAT node-status answer
                    name = self._decode_nbstat_hostname(payload[data_offset:data_end])
                    if name is None:
                        return
                    if name:
                        discovered_names.append(name)

            offset = data_end

        # Do not persist partial results from a truncated or malformed packet.
        for name in dict.fromkeys(discovered_names):
            self._log_asset(ip, mac, name, "NetBIOS")

    @staticmethod
    def _read_dns_name(payload: bytes, offset: int):
        """Read an NBNS compressed name, returning labels and the next wire offset."""
        if offset < 0 or offset >= len(payload):
            return None, None

        labels = []
        position = offset
        next_offset = None
        visited_pointers = set()
        expanded_length = 0

        # A valid expanded DNS name is at most 255 octets. The loop bound also
        # protects malformed pointer chains from consuming unbounded CPU.
        for _ in range(128):
            if position >= len(payload):
                return None, None

            length = payload[position]
            if length == 0:
                if next_offset is None:
                    next_offset = position + 1
                return labels, next_offset

            label_type = length & 0xC0
            if label_type == 0xC0:
                if position + 1 >= len(payload):
                    return None, None
                pointer = ((length & 0x3F) << 8) | payload[position + 1]
                # RFC 1035 compression pointers refer to an earlier name. This
                # also rules out header/forward-pointer tricks and self loops.
                if pointer >= position or pointer >= len(payload):
                    return None, None
                if pointer in visited_pointers:
                    return None, None
                visited_pointers.add(pointer)
                if next_offset is None:
                    next_offset = position + 2
                position = pointer
                continue

            if label_type != 0 or length > 63:
                return None, None
            label_end = position + 1 + length
            if label_end > len(payload):
                return None, None

            expanded_length += length + 1
            if expanded_length > 255:
                return None, None
            labels.append(payload[position + 1:label_end])
            position = label_end

        return None, None

    @staticmethod
    def _decode_nbstat_hostname(rdata: bytes) -> str | None:
        """Extract a hostname, empty for no eligible names, or None for malformed RDATA."""
        if not rdata:
            return None

        name_count = rdata[0]
        entries_end = 1 + name_count * 18
        required_length = entries_end + 46  # fixed NODE STATUS statistics structure
        if required_length > len(rdata):
            return None

        candidates = []
        for index in range(name_count):
            entry_offset = 1 + index * 18
            raw_name = rdata[entry_offset:entry_offset + 16]
            name_flags = struct.unpack_from("!H", rdata, entry_offset + 16)[0]

            # Reserved bits must be zero; only active, unique, non-conflicted,
            # non-deregistering names are eligible as host labels.
            if name_flags & 0xFF80 or not (name_flags & 0x0020):
                continue
            if name_flags & 0x0019:  # conflict, group name or deregistration
                continue

            label = raw_name[:15]
            if any(byte < 0x20 or byte > 0x7E for byte in label):
                continue
            name = label.decode("ascii").rstrip(" ").strip()
            if not name:
                continue

            permanent = bool(name_flags & 0x0040)
            workstation_suffix = raw_name[15] == 0x00
            candidates.append((permanent, workstation_suffix, name))

        if not candidates:
            return ""
        candidates.sort(key=lambda item: item[:2], reverse=True)
        return candidates[0][2]

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

    @staticmethod
    def _decode_netbios_name(encoded: bytes) -> str:
        """Decode the 32 A-P nibble characters representing a 16-byte NBNS name.

        The sixteenth byte is the NetBIOS service suffix, not part of the host label.
        Reject malformed first-level encodings instead of returning partial names.
        """
        if len(encoded) != 32:
            return ""
        try:
            encoded = encoded.upper()
            if any(byte < 0x41 or byte > 0x50 for byte in encoded):
                return ""
            decoded = bytes(
                ((encoded[i] - 0x41) << 4) | (encoded[i + 1] - 0x41)
                for i in range(0, 32, 2)
            )
            raw_name = decoded[:15]
            if any(byte < 0x20 or byte > 0x7E for byte in raw_name):
                return ""
            return raw_name.decode("ascii").rstrip(" ").strip()
        except (ValueError, UnicodeDecodeError):
            return ""

    def _prune_asset_database(self, conn):
        """Keep persistent passive-discovery inventory within a hard row ceiling."""
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM assets")
        count_row = cursor.fetchone()
        asset_count = count_row[0] if count_row and isinstance(count_row[0], int) else 0
        excess = max(0, asset_count - self.asset_db_max_entries)
        if excess:
            cursor.execute(
                """
                DELETE FROM assets WHERE rowid IN (
                    SELECT rowid FROM assets
                    ORDER BY last_seen ASC, rowid ASC
                    LIMIT ?
                )
                """,
                (excess,),
            )
        self._assets_since_prune = 0

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
            is_new_asset = row is None
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
                self._assets_since_prune += 1
                if self._assets_since_prune >= self.asset_db_prune_interval:
                    self._prune_asset_database(conn)
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

