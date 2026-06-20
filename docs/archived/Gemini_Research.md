# **Zero-Trust Host Security Daemon: Systems-Level Architecture and Implementation Blueprint**

This specification details the architecture, design, and implementation of a zero-trust endpoint security daemon tailored for the Arch Linux and systemd ecosystem1. The architecture is divided into four integration spheres, operating under strict system invariants: Large Language Models (LLMs) are treated as untrusted data parsers; all internal states, metadata, and rules are modeled using deterministic Pydantic schemas; host execution streams are guarded against command-line wrappers; and the system maintains a strict fail-closed security posture3.

## **1\. File Integrity Monitoring & Engine Scanners (Wazuh & YARA Layer)**

### **Asynchronous High-Performance FIM Engine**

The File Integrity Monitoring (FIM) subsystem relies on the Linux kernel’s native fanotify API, bypassing inotify to achieve mount-wide event monitoring that scales independently of filesystem size6. This prevents the resource exhaustion typical of recursive user-space directory watches7. The engine operates in a non-blocking asynchronous event loop, using a single-threaded kernel polling socket that shifts intensive scanning operations to a thread pool8.  
To minimize system overhead, the engine excludes directories with high-frequency writes and transient metrics2. The table below lists these exclusions and the reasons for omitting them:

| Mount Point / Path Pattern | Excluded Event Mask | Technical Rationale |
| :---- | :---- | :---- |
| /var/log/\* | FAN\_MODIFY | FAN\_ATTRIB | Prevents infinite logging loops caused by the security daemon's audit trail writes. |
| /tmp/\*, /var/tmp/\* | FAN\_CREATE | FAN\_MODIFY | Prevents overhead from transient, short-lived compiler outputs and temporary execution buffers2. |
| /proc/\*, /sys/\*, /dev/\* | All Events (FAN\_ALL\_EVENTS) | Excludes virtual filesystems that do not represent persistent storage. |
| /home/\*/.cache/\* | FAN\_MODIFY | FAN\_ATTRIB | Avoids tracking transient browser or utility caches. |

The FIM core interfaces directly with C-bindings via Python's ctypes library, eliminating the need for external shell dependencies or third-party binary execution11:

Python  
import os  
import sys  
import asyncio  
import ctypes  
import struct  
from concurrent.futures import ThreadPoolExecutor  
from typing import Generator, Dict, Any, List  
from pydantic import BaseModel, Field, IPvAnyAddress

\# Native C Library Binding Initialization  
libc \= ctypes.CDLL("libc.so.6", use\_errno=True)

\# fanotify Kernel Constants \[cite: 8, 9\]  
FAN\_CLASS\_NOTIF \= 0x00000000  
FAN\_CLOEXEC \= 0x00000001  
FAN\_NONBLOCK \= 0x00000002  
FAN\_MARK\_ADD \= 0x00000001  
FAN\_MARK\_MOUNT \= 0x00000010

\# Event Types \[cite: 8\]  
FAN\_ACCESS \= 0x00000001  
FAN\_MODIFY \= 0x00000002  
FAN\_ATTRIB \= 0x00000004  
FAN\_CREATE \= 0x00000100  
FAN\_DELETE \= 0x00000200  
FAN\_ONDIR \= 0x40000000

class FanotifyEventMetadata(ctypes.Structure):  
    \_fields\_ \= \[  
        ("event\_len", ctypes.c\_uint32),  
        ("vers", ctypes.c\_uint8),  
        ("reserved", ctypes.c\_uint8),  
        ("metadata\_len", ctypes.c\_uint16),  
        ("mask", ctypes.c\_uint64),  
        ("fd", ctypes.c\_int32),  
        ("pid", ctypes.c\_int32),  
    \]

\# Syscall Prototype Mappings  
libc.fanotify\_init.argtypes \= \[ctypes.c\_uint, ctypes.c\_uint\]  
libc.fanotify\_init.restype \= ctypes.c\_int

libc.fanotify\_mark.argtypes \= \[  
    ctypes.c\_int,  
    ctypes.c\_uint,  
    ctypes.c\_uint64,  
    ctypes.c\_int,  
    ctypes.c\_char\_p,  
\]  
libc.fanotify\_mark.restype \= ctypes.c\_int

### **In-Memory YARA Compilation and Verification Workflow**

Re-opening files using their absolute paths creates a Time-of-Check to Time-of-Use (TOCTOU) vulnerability window7. An attacker could modify a file after it triggers an event but before the scanner reads it7.  
To eliminate this race condition, the daemon scans the file descriptor (fd) passed directly by the kernel in the fanotify\_event\_metadata structure9. The system references the descriptor via /proc/self/fd/{fd}. This acts as a direct pin to the inode, ensuring YARA scans the exact version of the file that triggered the kernel event7.

Python  
import yara  
import sqlite3  
from datetime import datetime

class YARAMatchMetadata(BaseModel):  
    rule: str  
    namespace: str  
    tags: List\[str\]  
    meta: Dict\[str, Any\]

class FIMTelemetryPayload(BaseModel):  
    timestamp: str \= Field(default\_factory=lambda: datetime.utcnow().isoformat())  
    event\_type: str  
    pid: int  
    process\_comm: str  
    target\_path: str  
    yara\_matches: List\[YARAMatchMetadata\] \= \[\]

class AppendOnlyAuditTrail:  
    """Provides a tamper-resistant SQLite database using Write-Ahead Logging (WAL)."""  
    def \_\_init\_\_(self, db\_path: str \= "/var/lib/secdaemon/fim\_audit.db"):  
        self.db\_path \= db\_path  
        self.\_initialize\_database()

    def \_initialize\_database(self) \-\> None:  
        os.makedirs(os.path.dirname(self.db\_path), exist\_ok=True)  
        conn \= sqlite3.connect(self.db\_path)  
        with conn:  
            conn.execute("PRAGMA journal\_mode=WAL;")  
            conn.execute("PRAGMA synchronous=EXTRA;")  
            conn.execute(  
                """  
                CREATE TABLE IF NOT EXISTS security\_log (  
                    id INTEGER PRIMARY KEY AUTOINCREMENT,  
                    timestamp TEXT NOT NULL,  
                    event\_type TEXT NOT NULL,  
                    pid INTEGER NOT NULL,  
                    process\_comm TEXT NOT NULL,  
                    target\_path TEXT NOT NULL,  
                    payload\_json TEXT NOT NULL  
                );  
                """  
            )  
        conn.close()

    def write\_record(self, payload: FIMTelemetryPayload) \-\> None:  
        conn \= sqlite3.connect(self.db\_path)  
        try:  
            with conn:  
                conn.execute(  
                    """  
                    INSERT INTO security\_log (timestamp, event\_type, pid, process\_comm, target\_path, payload\_json)  
                    VALUES (?, ?, ?, ?, ?, ?);  
                    """,  
                    (  
                        payload.timestamp,  
                        payload.event\_type,  
                        payload.pid,  
                        payload.process\_comm,  
                        payload.target\_path,  
                        payload.model\_dump\_json()  
                    )  
                )  
        except Exception as e:  
            \# Enforce fail-closed state if writing to the log fails  
            self.trigger\_emergency\_panic(f"Audit log write failure: {e}")  
        finally:  
            conn.close()

    def trigger\_emergency\_panic(self, reason: str) \-\> None:  
        sys.stderr.write(f"PANIC: FAIL-CLOSED INVARIANT BROKEN: {reason}\\n")  
        \# Direct kernel sysrq panic to guarantee lockdown without subprocess dependencies  
        try:  
            with open("/proc/sysrq-trigger", "wb") as f:  
                f.write(b"c")  \# Triggers a kernel crash/panic  
        except Exception:  
            sys.exit(1)

class FIMDaemon:  
    def \_\_init\_\_(self, watch\_mount: str, rules\_path: str):  
        self.watch\_mount \= watch\_mount  
        self.fan\_fd \= \-1  
        self.db \= AppendOnlyAuditTrail()  
        self.yara\_rules \= yara.compile(filepath=rules\_path) \[cite: 15\]  
        self.executor \= ThreadPoolExecutor(max\_workers=os.cpu\_count() or 4\)

    def initialize\_fim(self) \-\> None:  
        self.fan\_fd \= libc.fanotify\_init(  
            FAN\_CLASS\_NOTIF | FAN\_CLOEXEC | FAN\_NONBLOCK,  
            os.O\_RDONLY  
        )  
        if self.fan\_fd \< 0:  
            err \= ctypes.get\_errno()  
            raise OSError(err, f"Unable to initialize fanotify: {os.strerror(err)}")

        mask \= FAN\_MODIFY | FAN\_CREATE | FAN\_ATTRIB | FAN\_ONDIR  
        res \= libc.fanotify\_mark(  
            self.fan\_fd,  
            FAN\_MARK\_ADD | FAN\_MARK\_MOUNT,  
            mask,  
            \-1,  
            self.watch\_mount.encode('utf-8')  
        )  
        if res \< 0:  
            err \= ctypes.get\_errno()  
            os.close(self.fan\_fd)  
            raise OSError(err, f"Unable to mark fanotify mount point: {os.strerror(err)}")

    def \_resolve\_process\_comm(self, pid: int) \-\> str:  
        try:  
            with open(f"/proc/{pid}/comm", "r") as f:  
                return f.read().strip()  
        except FileNotFoundError:  
            return "stale\_process"

    def \_get\_fd\_filepath(self, fd: int) \-\> str:  
        try:  
            return os.readlink(f"/proc/self/fd/{fd}")  
        except OSError:  
            return "unknown\_filepath"

    async def poll\_events(self) \-\> None:  
        loop \= asyncio.get\_running\_loop()  
        while True:  
            \# Wait for the file descriptor to have data available  
            await self.\_wait\_for\_fd()  
            try:  
                \# Read 4096 bytes of event metadata from the kernel  
                data \= os.read(self.fan\_fd, 4096\)  
            except BlockingIOError:  
                continue

            offset \= 0  
            struct\_size \= ctypes.sizeof(FanotifyEventMetadata)  
            while offset \+ struct\_size \<= len(data):  
                metadata \= FanotifyEventMetadata.from\_buffer\_copy(data, offset)  
                if metadata.fd \>= 0:  
                    \# Duplicate descriptor to isolate it from the reader loop  
                    duplicated\_fd \= os.dup(metadata.fd)  
                    loop.run\_in\_executor(  
                        self.executor,  
                        self.process\_file\_safety,  
                        duplicated\_fd,  
                        metadata.pid,  
                        metadata.mask  
                    )  
                    os.close(metadata.fd)  
                offset \+= metadata.event\_len

    async def \_wait\_for\_fd(self) \-\> None:  
        loop \= asyncio.get\_running\_loop()  
        future \= loop.create\_future()  
        def callback():  
            loop.remove\_reader(self.fan\_fd)  
            if not future.done():  
                future.set\_result(None)  
        loop.add\_reader(self.fan\_fd, callback)  
        await future

    def process\_file\_safety(self, fd: int, pid: int, mask: int) \-\> None:  
        \# Exclude actions initiated by the security daemon itself to prevent recursive scanning loops  
        if pid \== os.getpid():  
            os.close(fd)  
            return

        target\_path \= self.\_get\_fd\_filepath(fd)  
        process\_comm \= self.\_resolve\_process\_comm(pid)  
          
        event\_type \= "UNKNOWN"  
        if mask & FAN\_MODIFY:  
            event\_type \= "MODIFY"  
        elif mask & FAN\_CREATE:  
            event\_type \= "CREATE"  
        elif mask & FAN\_ATTRIB:  
            event\_type \= "ATTRIB"

        payload \= FIMTelemetryPayload(  
            event\_type=event\_type,  
            pid=pid,  
            process\_comm=process\_comm,  
            target\_path=target\_path  
        )

        try:  
            \# Wrap the raw file descriptor and pass it directly to the YARA scanning context \[cite: 13, 15\]  
            with open(f"/proc/self/fd/{fd}", "rb") as scan\_file:  
                matches \= self.yara\_rules.match(data=scan\_file.read()) \[cite: 15\]  
                for match in matches:  
                    payload.yara\_matches.append(  
                        YARAMatchMetadata(  
                            rule=match.rule,  
                            namespace=match.namespace,  
                            tags=match.tags,  
                            meta=match.meta  
                        )  
                    )  
        except Exception as e:  
            \# Any failures during scanning halt the execution stream to maintain a fail-closed posture  
            self.db.trigger\_emergency\_panic(f"YARA execution fault: {e}")  
        finally:  
            os.close(fd)

        \# Commit telemetry records to the database  
        self.db.write\_record(payload)

## **2\. Passive/Active Layer-2 & Layer-3 Discovery (runZero Network Layer)**

### **Low-Level Packet Socket Parsing Loop**

To map local network assets without using subprocesses (such as calling out to external binary utilities like arp or nmap), the network layer establishes a raw socket connection on AF\_PACKET16. This socket captures and decodes broadcast packets directly at Layer 212.

Python  
import socket  
import struct  
import binascii

class AssetProfileSchema(BaseModel):  
    mac: str \= Field(..., pattern=r"^(\[0-9a-f\]{2}:){5}\[0-9a-f\]{2}$")  
    ip: str  
    hostname: str \= "Unresolved"  
    vendor: str \= "Unknown"  
    detected\_via: str  
    last\_seen: float

class PassiveNetworkDiscoverer:  
    """Parses Ethernet frames to discover assets via ARP, mDNS, LLMNR, and SSDP \[cite: 14, 18, 19\]."""  
    def \_\_init\_\_(self, interface: str):  
        self.interface \= interface  
        self.raw\_sock \= None  
        self.mac\_vendor\_index \= {  
            "00:1e:06": "Wintel Corp",  
            "3c:a6:16": "Apple Inc.",  
            "b4:b5:2f": "Ubiquiti Networks",  
            "00:0c:29": "VMware"  
        }

    def bind\_socket(self) \-\> None:  
        \# Bind raw physical packet socket (0x0003 translates to ETH\_P\_ALL) \[cite: 17, 20\]  
        self.raw\_sock \= socket.socket(socket.AF\_PACKET, socket.SOCK\_RAW, socket.htons(0x0003)) \[cite: 17, 20\]  
        self.raw\_sock.bind((self.interface, 0))

    def \_resolve\_vendor(self, mac: str) \-\> str:  
        prefix \= mac.lower()\[:8\]  
        return self.mac\_vendor\_index.get(prefix, "Unknown Hardware Vendor")

    def read\_and\_parse\_packets(self) \-\> Generator\[AssetProfileSchema, None, None\]:  
        while True:  
            raw\_data, \_ \= self.raw\_sock.recvfrom(65535)  
            if len(raw\_data) \< 14:  
                continue

            \# Unpack Ethernet Header \[cite: 12, 21\]  
            eth\_hdr \= struct.unpack("\!6s6sH", raw\_data\[:14\]) \[cite: 12, 21\]  
            dst\_mac \= ":".join(f"{b:02x}" for b in eth\_hdr\[0\]) \[cite: 21\]  
            src\_mac \= ":".join(f"{b:02x}" for b in eth\_hdr\[1\]) \[cite: 21\]  
            proto\_type \= eth\_hdr\[2\]

            \# Vector 1: Address Resolution Protocol (ARP \- 0x0806) \[cite: 21\]  
            if proto\_type \== 0x0806:  
                arp\_raw \= raw\_data\[14:42\]  
                if len(arp\_raw) \< 28:  
                    continue  
                arp\_hdr \= struct.unpack("\!HHBBH6s4s6s4s", arp\_raw) \[cite: 12\]  
                operation \= arp\_hdr\[4\]  
                sender\_ip \= socket.inet\_ntoa(arp\_hdr\[6\])  
                  
                if operation in (1, 2):  \# ARP Request or Reply \[cite: 22, 23\]  
                    yield AssetProfileSchema(  
                        mac=src\_mac,  
                        ip=sender\_ip,  
                        vendor=self.\_resolve\_vendor(src\_mac),  
                        detected\_via="ARP",  
                        last\_seen=datetime.utcnow().timestamp()  
                    )

            \# Vector 2: Internet Protocol Version 4 (IP \- 0x0800) \[cite: 21\]  
            elif proto\_type \== 0x0800:  
                ip\_raw \= raw\_data\[14:34\]  
                if len(ip\_raw) \< 20:  
                    continue  
                ip\_hdr \= struct.unpack("\!BBHHHBBH4s4s", ip\_raw) \[cite: 12\]  
                ip\_protocol \= ip\_hdr\[6\]  
                src\_ip \= socket.inet\_ntoa(ip\_hdr\[8\])

                \# Focus on User Datagram Protocol (UDP \- 17\) \[cite: 12\]  
                if ip\_protocol \== 17:  
                    udp\_raw \= raw\_data\[34:42\]  
                    if len(udp\_raw) \< 8:  
                        continue  
                    udp\_hdr \= struct.unpack("\!HHHH", udp\_raw)  
                    src\_port \= udp\_hdr\[0\]  
                    dst\_port \= udp\_hdr\[1\]

                    \# Sub-Vector 2a: Multicast DNS (mDNS \- Port 5353\) \[cite: 18\]  
                    if dst\_port \== 5353:  
                        dns\_raw \= raw\_data\[42:\]  
                        hostname \= self.\_extract\_dns\_hostname(dns\_raw)  
                        if hostname:  
                            yield AssetProfileSchema(  
                                mac=src\_mac,  
                                ip=src\_ip,  
                                hostname=hostname,  
                                vendor=self.\_resolve\_vendor(src\_mac),  
                                detected\_via="mDNS",  
                                last\_seen=datetime.utcnow().timestamp()  
                            )

                    \# Sub-Vector 2b: Link-Local Multicast Name Resolution (LLMNR \- Port 5355\) \[cite: 19\]  
                    elif dst\_port \== 5355:  
                        yield AssetProfileSchema(  
                            mac=src\_mac,  
                            ip=src\_ip,  
                            vendor=self.\_resolve\_vendor(src\_mac),  
                            detected\_via="LLMNR",  
                            last\_seen=datetime.utcnow().timestamp()  
                        )

                    \# Sub-Vector 2c: Simple Service Discovery Protocol (SSDP \- Port 1900\)  
                    elif dst\_port \== 1900:  
                        yield AssetProfileSchema(  
                            mac=src\_mac,  
                            ip=src\_ip,  
                            vendor=self.\_resolve\_vendor(src\_mac),  
                            detected\_via="SSDP",  
                            last\_seen=datetime.utcnow().timestamp()  
                        )

    def \_extract\_dns\_hostname(self, payload: bytes) \-\> str:  
        """Parses DNS query labels to extract the host identifier \[cite: 18\]."""  
        if len(payload) \< 12:  
            return ""  
        try:  
            offset \= 12  
            labels \= \[\]  
            while offset \< len(payload):  
                length \= payload\[offset\]  
                if length \== 0:  
                    break  
                \# Handle compressed name pointers \[cite: 18\]  
                if (length & 0xC0) \== 0xC0:  
                    break  
                offset \+= 1  
                label \= payload\[offset:offset+length\].decode('utf-8', errors='ignore')  
                labels.append(label)  
                offset \+= length  
            return ".".join(labels) if labels else "Unresolved"  
        except Exception:  
            return "Unresolved"

### **eBPF Connection Tracker and Telemetry Exporter**

For outbound connection monitoring, the daemon bypasses application wrappers and uses kernel-level tracing24. The eBPF monitor attaches to the IPv4 connection initiator (\_\_sys\_connect) to extract connection endpoints and generate telemetry logs25.

Python  
from bcc import BPF

bpf\_source\_connection\_tracker \= """  
\#include \<uapi/linux/ptrace.h\>  
\#include \<net/sock.h\>  
\#include \<linux/socket.h\>

struct connection\_record\_t {  
    u32 pid;  
    char process\_comm\[16\];  
    u32 destination\_ip;  
    u16 destination\_port;  
};

BPF\_PERF\_OUTPUT(socket\_connections\_buffer);

int trace\_socket\_connect(struct pt\_regs \*ctx, struct socket \*sock, struct sockaddr \*address, int addr\_len) {  
    if (address-\>sa\_family \!= AF\_INET) {  
        return 0;  
    }

    struct sockaddr\_in \*target \= (struct sockaddr\_in \*)address;  
    struct connection\_record\_t record \= {};

    record.pid \= bpf\_get\_current\_pid\_tgid() \>\> 32;  
    bpf\_get\_current\_comm(\&record.process\_comm, sizeof(record.process\_comm));  
    record.destination\_ip \= target-\>sin\_addr.s\_addr;  
    record.destination\_port \= be16\_to\_cpu(target-\>sin\_port);

    socket\_connections\_buffer.perf\_submit(ctx, \&record, sizeof(record));  
    return 0;  
}  
"""

class ConnectionTelemetrySchema(BaseModel):  
    timestamp: str \= Field(default\_factory=lambda: datetime.utcnow().isoformat())  
    pid: int  
    process\_comm: str  
    destination\_ip: IPvAnyAddress  
    destination\_port: int  
    is\_beacon\_candidate: bool \= False

class NetworkTelemetryMonitor:  
    def \_\_init\_\_(self):  
        self.bpf\_context \= None

    def begin\_monitoring(self, telemetry\_callback) \-\> None:  
        self.bpf\_context \= BPF(text=bpf\_source\_connection\_tracker) \[cite: 24\]  
        \# Attach the kprobe directly to the sys\_connect system call execution stream \[cite: 24, 26\]  
        self.bpf\_context.attach\_kprobe(event="\_\_sys\_connect", fn\_name="trace\_socket\_connect") \[cite: 24, 26\]  
          
        def handle\_event(cpu, raw\_data, size):  
            event \= self.bpf\_context\["socket\_connections\_buffer"\].event(raw\_data) \[cite: 24\]  
            \# Convert destination IP from big-endian integer format \[cite: 12\]  
            ip\_str \= socket.inet\_ntoa(struct.pack("\<I", event.destination\_ip))  
              
            payload \= ConnectionTelemetrySchema(  
                pid=event.pid,  
                process\_comm=event.process\_comm.decode('utf-8', errors='ignore'),  
                destination\_ip=ip\_str,  
                destination\_port=event.destination\_port  
            )  
            telemetry\_callback(payload)

        self.bpf\_context\["socket\_connections\_buffer"\].open\_perf\_buffer(handle\_event) \[cite: 24\]

    def poll\_buffer(self) \-\> None:  
        if self.bpf\_context:  
            self.bpf\_context.perf\_buffer\_poll() \[cite: 24\]

## **3\. Proactive System Tripwires & Canary Constraints**

The defense configuration deploys un-trackable system canary paths alongside isolated network decoy services to capture local host propagation attempts and reconnaissance sweeps26.

### **eBPF Tripwire Detection Engine**

Canary files are written to system paths (such as /etc/api\_keys.json) to act as honey-tokens. Any access to these paths by an unauthorized security context triggers an immediate kernel-level alarm26.

Python  
ebpf\_tripwire\_kernel\_sensor \= """  
\#include \<uapi/linux/ptrace.h\>  
\#include \<linux/sched.h\>

struct canary\_violation\_t {  
    u32 pid;  
    u32 uid;  
    char comm\[16\];  
    char filepath\[64\];  
};

BPF\_PERF\_OUTPUT(canary\_alarms);

int trace\_sys\_openat\_entry(struct pt\_regs \*ctx, int dfd, const char \_\_user \*filename, int flags) {  
    char target\_path\[64\] \= {};  
    bpf\_probe\_read\_user\_str(\&target\_path, sizeof(target\_path), filename); \[cite: 24, 25\]

    // Validate if the first path segment matches '/etc/api\_keys'  
    if (target\_path\[0\] \== '/' && target\_path\[1\] \== 'e' && target\_path\[2\] \== 't' && target\_path\[3\] \== 'c' &&   
        target\_path\[4\] \== '/' && target\_path\[5\] \== 'a' && target\_path\[6\] \== 'p' && target\_path\[7\] \== 'i' &&   
        target\_path\[8\] \== '\_') {  
          
        struct canary\_violation\_t alarm \= {};  
        alarm.pid \= bpf\_get\_current\_pid\_tgid() \>\> 32;  
        alarm.uid \= bpf\_get\_current\_uid\_gid() & 0xFFFFFFFF;  
        bpf\_get\_current\_comm(\&alarm.comm, sizeof(alarm.comm));  
        bpf\_probe\_read\_kernel(\&alarm.filepath, sizeof(alarm.filepath), target\_path);

        canary\_alarms.perf\_submit(ctx, \&alarm, sizeof(alarm));  
    }  
    return 0;  
}  
"""

### **Decoy Network Services and Netlink Mitigation**

To trap and mitigate local host enumeration sweeps, the daemon spawns an isolated network namespace that hosts a lightweight decoy listener mimicking administrative endpoints28. If any transaction occurs on these endpoints, the daemon uses raw netlink calls to apply dynamic nftables blocking rules on the host interface30.

Python  
from pyroute2 import NetNS, NFTables, IPRoute  
from pyroute2.nftables.expressions import ipv4addr, verdict

class DecoySystemCoordinator:  
    """Controls isolated namespaces and configures firewall state changes using Netlink."""  
    def \_\_init\_\_(self, interface: str, decoy\_ip: str \= "192.168.100.1"):  
        self.interface \= interface  
        self.decoy\_ip \= decoy\_ip  
        self.ns\_name \= "ns\_decoy\_sandbox"  
        self.nft \= NFTables()  
        self.\_initialize\_firewall\_ruleset()

    def \_initialize\_firewall\_ruleset(self) \-\> None:  
        """Sets up the initial nftables structure to support dynamic blocks \[cite: 31, 32\]."""  
        try:  
            \# Create a dedicated table and hook chain for blocking operations \[cite: 31, 32\]  
            self.nft.table('add', name='filter\_mitigation')  
            self.nft.chain('add', table='filter\_mitigation', name='input\_blocker', hook='input', type='filter', priority=-1)  
        except Exception as e:  
            sys.stderr.write(f"Unable to initialize firewall: {e}\\n")  
            sys.exit(1)

    def provision\_decoy\_isolation(self) \-\> None:  
        """Spawns an isolated network namespace to route attacker traffic."""  
        \# Create a new network namespace  
        netns\_mgr \= NetNS(self.ns\_name) \[cite: 28\]  
        ip\_cmd \= IPRoute()

        \# Find the physical device interface identifier  
        interface\_idx \= ip\_cmd.link\_lookup(ifname=self.interface)\[0\]

        \# Push the link interface into the decoy namespace \[cite: 28\]  
        ip\_cmd.link('set', index=interface\_idx, net\_ns\_fd=self.ns\_name) \[cite: 28\]

        \# Configure network properties inside the namespace  
        ns\_ip\_cmd \= IPRoute(netns=self.ns\_name) \[cite: 28\]  
        ns\_ip\_cmd.link('set', index=interface\_idx, state='up') \[cite: 28, 29\]  
        ns\_ip\_cmd.addr('add', index=interface\_idx, address=self.decoy\_ip, prefixlen=24) \[cite: 28, 29\]

        ns\_ip\_cmd.close()  
        ip\_cmd.close()

    def process\_interception(self, attacker\_ip: str) \-\> None:  
        """Applies immediate filtering rules using raw Netlink expressions \[cite: 30\]."""  
        try:  
            \# Drop incoming traffic from the target IP \[cite: 30, 32\]  
            self.nft.rule('add', table='filter\_mitigation', chain='input\_blocker',  
                          expressions=(  
                              ipv4addr(src=attacker\_ip),  
                              verdict(code=0)  \# Verdict 0 drops the packet \[cite: 30, 32, 33\]  
                          ))  
        except Exception as e:  
            \# Trigger immediate lockdown if the mitigation rule cannot be applied  
            self.lockdown\_system\_and\_panic(f"Firewall state application error: {e}")

    def lockdown\_system\_and\_panic(self, cause: str) \-\> None:  
        """Isolates the host immediately in a fail-closed posture when a fault is detected."""  
        sys.stderr.write(f"FAIL-CLOSED EXECUTED: {cause}\\n")  
        \# Write directly to sysrq to trigger a kernel panic and halt processing instantly  
        try:  
            with open("/proc/sysrq-trigger", "wb") as sysrq\_file:  
                sysrq\_file.write(b"o")  \# Hard shutdown request  
        except Exception:  
            sys.exit(1)

## **4\. Sigma & YARA Syntax Generation Engine**

### **Un-Trusted Prompt Parsing Boundary**

Large Language Models (LLMs) are highly non-deterministic and must be treated as completely untrusted interfaces3. The daemon processes telemetry and formats instructions strictly into standardized validation schemas before generating any rules3.

Python  
from typing import Optional  
from pydantic import BaseModel, Field

class RuleInputIntent(BaseModel):  
    rule\_name: str \= Field(..., pattern=r"^\[A-Za-z0-9\_\]+$")  
    author: str \= Field(default="SecDaemon Automated Compiler")  
    description: str \= Field(..., min\_length=10)  
    ioc\_indicator: str \= Field(..., min\_length=4)  
    mitre\_id: Optional\[str\] \= Field(default=None, pattern=r"^T\[0-9\]{4}$")

class SignatureCompiler:  
    """Translates sanitized JSON inputs into validated rule syntax."""  
    @staticmethod  
    def construct\_yara(intent: RuleInputIntent) \-\> str:  
        \# Escape path and data modifiers to protect against injection attempts  
        sanitized\_ioc \= intent.ioc\_indicator.replace('"', '\\\\"').replace('\\\\', '\\\\\\\\')  
        return f"""  
rule {intent.rule\_name} {{  
    meta:  
        author \= "{intent.author}"  
        description \= "{intent.description}"  
        mitre\_att\_id \= "{intent.mitre\_id or "N/A"}"  
    strings:  
        $ioc\_string \= "{sanitized\_ioc}" ascii wide nocase  
    condition:  
        any of them  
}}  
"""

    @staticmethod  
    def construct\_sigma(intent: RuleInputIntent) \-\> str:  
        sanitized\_ioc \= intent.ioc\_indicator.replace("'", "''")  
        return f"""  
title: {intent.rule\_name}  
id: {intent.rule\_name.lower().replace('\_', '-')}  
status: experimental  
description: {intent.description}  
author: {intent.author}  
logsource:  
    category: file\_event  
    product: linux  
detection:  
    selection:  
        target\_path|contains: '{sanitized\_ioc}'  
    condition: selection  
fields:  
    \- target\_path  
    \- pid  
    \- process\_comm  
falsepositives:  
    \- Unknown  
level: medium  
"""

### **ephem\_compile: Ephemeral Verification Loop**

To prevent syntactic anomalies or compiler-level exploits from affecting the primary host process, the daemon runs a test compilation inside an unprivileged systemd sandbox container5. This verification sandbox runs as a dynamically generated user, has no network access, and maps directories read-only to evaluate the candidate ruleset against a local benign test corpus3.

Python  
import subprocess  
import tempfile

class EphemeralValidator:  
    """Compiles and verifies rules inside a systemd-run sandbox \[cite: 1, 34\]."""  
    def \_\_init\_\_(self, benign\_corpus\_dir: str \= "/var/lib/secdaemon/benign\_test\_set"):  
        self.benign\_corpus\_dir \= benign\_corpus\_dir

    def run\_sandbox\_validation(self, signature\_text: str) \-\> bool:  
        """Compiles the rule inside an isolated dynamic-user container \[cite: 34, 35\]."""  
        with tempfile.NamedTemporaryFile(suffix=".yar", mode="w", delete=False) as temp\_file:  
            temp\_file.write(signature\_text)  
            temp\_rule\_path \= temp\_file.name

        try:  
            \# Build an isolated systemd sandbox wrapper \[cite: 34, 35\]  
            sandbox\_command \= \[  
                "/usr/bin/systemd-run",  
                "--quiet",  
                "--wait",  
                "--user=false", \# Execute inside system manager context \[cite: 1, 36\]  
                "--property=PrivateTmp=yes", \[cite: 34, 37\]  
                "--property=ProtectSystem=strict", \[cite: 34\]  
                "--property=ProtectHome=yes", \[cite: 34\]  
                "--property=PrivateNetwork=yes", \[cite: 34\]  
                "--property=DynamicUser=yes", \[cite: 34, 35\]  
                "--property=NoNewPrivileges=yes", \[cite: 34, 37\]  
                "--property=CapabilityBoundingSet=", \[cite: 35, 36\]  
                "--property=SystemCallFilter=@system-service", \[cite: 38\]  
                "--property=RestrictAddressFamilies=none", \[cite: 34, 38\]  
                f"--property=BindReadOnlyPaths={temp\_rule\_path}:{temp\_rule\_path}", \[cite: 35\]  
                f"--property=BindReadOnlyPaths={self.benign\_corpus\_dir}:/mnt/benign\_corpus", \[cite: 35\]  
                "/usr/bin/yara",  
                temp\_rule\_path,  
                "/mnt/benign\_corpus"  
            \]

            \# Execute validation run  
            exec\_result \= subprocess.run(  
                sandbox\_command,  
                capture\_output=True,  
                text=True,  
                timeout=10  
            )

            \# Rule validity validation rules:  
            \# 1\. Compiler must exit cleanly with status 0  
            \# 2\. Rule must not generate false matches against the benign corpus  
            if exec\_result.returncode \== 0:  
                if not exec\_result.stdout.strip():  
                    return True  \# Rule is valid and passed safety checks  
                else:  
                    sys.stderr.write("Verification failed: Rule matched benign asset baseline.\\n")  
            else:  
                sys.stderr.write(f"Compiler syntax fault inside sandbox context: {exec\_result.stderr}\\n")

        except subprocess.TimeoutExpired:  
            sys.stderr.write("Compilation execution exceeded target timeout limit.\\n")  
        finally:  
            if os.path.exists(temp\_rule\_path):  
                os.remove(temp\_rule\_path)

        return False

## **5\. Security Orchestration & Flow Framework**

                 \+-------------------------------------------------+  
                 |             KERNEL SPACE HOOKPOINT              |  
                 | (sys\_enter\_openat / vfs\_open / \_\_sys\_connect)   |  
                 \+------------------------+------------------------+  
                                          |  
                        event (ring buffer | perf channel)  
                                          v  
                 \+-------------------------------------------------+  
                 |            SYSTEM DAEMON POLL ENGINE            |  
                 |  (Processes Raw Bytes / Matches Telemetry)      |  
                 \+------------------------+------------------------+  
                                          |  
                        valid telemetry   |  (Pydantic payload)  
                                          v  
                 \+-------------------------------------------------+  
                 |              APPEND-ONLY SECURITY               |  
                 |               LOGGING REPOSITORY                |  
                 \+------------------------+------------------------+  
                                          |  
                        rule proposal     |  (Structured JSON)  
                                          v  
                 \+-------------------------------------------------+  
                 |               RULE SYNTAX ENGINE                |  
                 |  (Sanitizes and Generates YARA/SIGMA Targets)   |  
                 \+------------------------+------------------------+  
                                          |  
                        sandbox execute   |  (systemd-run validation)  
                                          v  
                 \+-------------------------------------------------+  
                 |          EPHEMERAL VERIFICATION SYSTEM          |  
                 |  (Evaluates Signatures over Benign Corpuses)    |  
                 \+-------------------------------------------------+

### **Rate-Limiting Metrics for Discovery Events**

To prevent event starvation or resource exhaustion attacks against the database writer loop, packet-sniffing streams and telemetry alert channels run under a strict token bucket rate-limiting algorithm:

$$T(t) \= \\min\\left(B, T\_{prev} \+ r \\cdot (t \- t\_{prev})\\right)$$  
where $B$ is the maximum bucket capacity (e.g., 100 tokens), $r$ is the recovery generation rate (e.g., 5 tokens per second), and $t \- t\_{prev}$ is the delta time calculation. Telemetry logs are suppressed when $T(t) \< 1$, preventing disk thrashing during high-volume network activity or system events14.

### **systemd Service Blueprint**

Deploy this unit definition to /etc/systemd/system/secdaemon.service to enforce sandbox constraints on the primary host daemon34:

Ini, TOML  
\[Unit\]  
Description=Principal Endpoint Detection and Mitigation Daemon  
After=network.target \[cite: 1\]  
StartLimitIntervalSec=0

\[Service\]  
Type=simple  
ExecStart=/usr/bin/python3 /usr/bin/secdaemon.py  
Restart=always  
RestartSec=1

\# Sandboxing and System Hardening  
NoNewPrivileges=true \[cite: 34, 37\]  
ProtectSystem=strict \[cite: 34\]  
ProtectHome=true \[cite: 34\]  
PrivateTmp=true \[cite: 34, 37\]  
PrivateDevices=true \[cite: 34, 35\]  
ProtectKernelTunables=true \[cite: 35\]  
ProtectControlGroups=true \[cite: 35\]  
ProtectKernelModules=true \[cite: 35\]  
RestrictNamespaces=yes

\# Paths are mapped read-only, except for writing log trails and DB states  
ReadWritePaths=/var/lib/secdaemon/ /var/log/  
ReadOnlyPaths=/etc/ /usr/bin/

\# Limit raw capabilities to the minimum needed for sockets and tracing \[cite: 36\]  
CapabilityBoundingSet=CAP\_NET\_RAW CAP\_SYS\_ADMIN  
AmbientCapabilities=CAP\_NET\_RAW CAP\_SYS\_ADMIN \[cite: 36\]

\# Fallback actions for execution faults  
\# Under a fail-closed architecture, system faults trigger immediate lockdown  
OnFailure=emergency-lockdown.service

\[Install\]  
WantedBy=multi-user.target

### **Emergency Lockdown Service**

Deploy this fallback unit definition to /etc/systemd/system/emergency-lockdown.service to handle system faults or service crashes3:

Ini, TOML  
\[Unit\]  
Description=Emergency Fail-Closed System Isolation Service  
DefaultDependencies=no  
Before=shutdown.target

\[Service\]  
Type=oneshot  
\# Clear existing rules and drop all incoming, outgoing, and forward traffic \[cite: 31, 32\]  
ExecStart=/usr/bin/nft flush ruleset \[cite: 31, 32\]  
ExecStartPost=/usr/bin/nft add table inet filter \[cite: 32\]  
ExecStartPost=/usr/bin/nft add chain inet filter input { type filter hook input priority 0; policy drop; } \[cite: 32\]  
ExecStartPost=/usr/bin/nft add chain inet filter forward { type filter hook forward priority 0; policy drop; } \[cite: 32\]  
ExecStartPost=/usr/bin/nft add chain inet filter output { type filter hook output priority 0; policy drop; }  
ExecStartPost=/usr/bin/systemctl isolate rescue.target

#### **Works cited**

1. Using systemd features to secure services \- Red Hat, [https://www.redhat.com/en/blog/systemd-secure-services](https://www.redhat.com/en/blog/systemd-secure-services)  
2. Systemd Service Hardening | Linux Journal, [https://www.linuxjournal.com/content/systemd-service-strengthening](https://www.linuxjournal.com/content/systemd-service-strengthening)  
3. What is eBPF? An Introduction and Deep Dive into the eBPF Technology, [https://ebpf.io/what-is-ebpf/](https://ebpf.io/what-is-ebpf/)  
4. Runtime Isolation \- Model Context Protocol Security, [https://modelcontextprotocol-security.io/build/runtime-isolation/](https://modelcontextprotocol-security.io/build/runtime-isolation/)  
5. Systemd service sandboxing and security hardening (2020) | Hacker News, [https://news.ycombinator.com/item?id=29977271](https://news.ycombinator.com/item?id=29977271)  
6. How to Implement Effective Linux Directory Monitoring | LabEx, [https://labex.io/tutorials/linux-how-to-implement-effective-linux-directory-monitoring-415799](https://labex.io/tutorials/linux-how-to-implement-effective-linux-directory-monitoring-415799)  
7. File Monitoring with eBPF and Tetragon (Part 1\) \- Isovalent, [https://isovalent.com/blog/post/file-monitoring-with-ebpf-and-tetragon-part-1/](https://isovalent.com/blog/post/file-monitoring-with-ebpf-and-tetragon-part-1/)  
8. butter \- PyPI, [https://pypi.org/project/butter/](https://pypi.org/project/butter/)  
9. python-fanotify/fanotify.c at master · google/python-fanotify \- GitHub, [https://github.com/google/python-fanotify/blob/master/fanotify.c](https://github.com/google/python-fanotify/blob/master/fanotify.c)  
10. yara-ctypes Documentation, [https://yara-ctypes.readthedocs.io/\_/downloads/en/latest/pdf/](https://yara-ctypes.readthedocs.io/_/downloads/en/latest/pdf/)  
11. Linux manual pages: directory by project \- man7.org, [https://man7.org/linux/man-pages/dir\_by\_project.html](https://man7.org/linux/man-pages/dir_by_project.html)  
12. Packet Sniffing — Python Cheat Sheet, [https://www.pythonsheets.com/notes/network/python-socket-sniffer.html](https://www.pythonsheets.com/notes/network/python-socket-sniffer.html)  
13. The C API — yara 4.4.0 documentation, [https://yara.readthedocs.io/en/stable/capi.html](https://yara.readthedocs.io/en/stable/capi.html)  
14. tjnull/leetha: Passive network fingerprinting and analysis engine \- GitHub, [https://github.com/tjnull/leetha](https://github.com/tjnull/leetha)  
15. socket — Low-level networking interface — Python 3.14.6 documentation, [https://docs.python.org/3/library/socket.html](https://docs.python.org/3/library/socket.html)  
16. Raw\_Sock,Struct,Arp requests : r/Python \- Reddit, [https://www.reddit.com/r/Python/comments/34ln7l/raw\_sockstructarp\_requests/](https://www.reddit.com/r/Python/comments/34ln7l/raw_sockstructarp_requests/)  
17. Observability With eBPF \- DZone, [https://dzone.com/articles/observability-with-ebpf](https://dzone.com/articles/observability-with-ebpf)  
18. How to Monitor File Access with eBPF \- OneUptime, [https://oneuptime.com/blog/post/2026-01-07-ebpf-file-access-monitoring/view](https://oneuptime.com/blog/post/2026-01-07-ebpf-file-access-monitoring/view)  
19. GitHub \- amirhnajafiz/syscall-blocker: Using eBPF to block system calls in Linux., [https://github.com/amirhnajafiz/kafka](https://github.com/amirhnajafiz/kafka)  
20. eBPF Tutorial: Privilege Escalation via File Content Manipulation \- DEV Community, [https://dev.to/yunwei37/ebpf-tutorial-privilege-escalation-via-file-content-manipulation-1797](https://dev.to/yunwei37/ebpf-tutorial-privilege-escalation-via-file-content-manipulation-1797)  
21. pyroute2 0.4.13.post7 documentation, [https://pyroute2.org/pyroute2-0.4.13.post7/general.html](https://pyroute2.org/pyroute2-0.4.13.post7/general.html)  
22. pyroute2 0.9.3rc1 documentation, [https://docs.pyroute2.org/general.html](https://docs.pyroute2.org/general.html)  
23. NFTables module — pyroute2 0.6.9.post99 documentation, [https://pyroute2.org/pyroute2-0.6.9.post99/nftables.html](https://pyroute2.org/pyroute2-0.6.9.post99/nftables.html)  
24. nftables \- ArchWiki, [https://wiki.archlinux.org/title/Nftables](https://wiki.archlinux.org/title/Nftables)  
25. How to Block an IP Address with nftables \- OneUptime, [https://oneuptime.com/blog/post/2026-03-20-block-ip-nftables/view](https://oneuptime.com/blog/post/2026-03-20-block-ip-nftables/view)  
26. How to Configure systemd Service Sandboxing on Ubuntu \- OneUptime, [https://oneuptime.com/blog/post/2026-03-02-configure-systemd-service-sandboxing-ubuntu/view](https://oneuptime.com/blog/post/2026-03-02-configure-systemd-service-sandboxing-ubuntu/view)  
27. Sandboxing Services with Systemd | Lincoln Loop, [https://lincolnloop.com/blog/sandboxing-services-systemd/](https://lincolnloop.com/blog/sandboxing-services-systemd/)