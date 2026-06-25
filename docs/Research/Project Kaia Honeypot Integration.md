# **Hardware Honeypot Integration into Hardened AI SecOps Architecture (Project Kaia)**

Integrating network-adjacent hardware honeypot nodes directly into an isolated, hardened AI system administration and security operations layer (Project Kaia) expands localized threat intelligence capabilities. Operating on a zero-trust model on Arch Linux, the Kaia architecture strictly decouples the automated planning loops of its artificial intelligence orchestrator from host defensive barriers. By projecting passive decoy networks into adjacent Local Area Network (LAN) segments, the system can capture real-time adversary interaction signatures, identify lateral movement vectors, and perform deterministic, out-of-process policy modifications across the host. This technical engineering assessment evaluates target software ecosystems, design ingestion topologies, feedback loops, and integration requirements.

## **Target Software Ecosystem Feasibility Matrix**

Evaluating the feasibility of various honeypot frameworks on limited-resource single-board computers—specifically the Raspberry Pi family, ranging from the low-power Pi Zero 2 W to the high-performance Pi 5—requires a rigorous analysis of RAM utilization, CPU cycles, and structural integration ease. The selected daemon must emit structured telemetry that is easily parsed by Kaia’s deterministic boundaries without introducing operational overhead.  
The table below contrasts the technical parameters of the target honeypot frameworks:

| Honeypot Framework | Target Hardware Class | Idle RAM Footprint | Peak CPU Footprint (Active Brute-Force) | Native Log Output Structure | Structural Integration Path | Feasibility Rating |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| **OpenCanary** | Pi Zero 2 W / Pi 3 / CM4 | 22 MB – 35 MB | 2% – 8% (Single Core) | Structured JSON (single-line newline-delimited)1 | High (Direct tailing, Webhook, Python log handler)3 | **Highly Feasible** |
| **Cowrie** | Pi 3 / CM4 / Pi 4 | 95 MB – 140 MB | 25% – 60% (Cryptographic handshake)5 | Deeply nested JSON (cowrie.json)6 | Medium (Structured event IDs, custom parsers)8 | **Highly Feasible** |
| **DShield Honeypot** | Pi 3 / CM4 / Pi 4 | 120 MB – 180 MB | 15% – 45% (Parallel HTTP processing) | Structured JSON (tailored for SANS ISC REST APIs)10 | Low (Internet dependent, rigid network modifications)10 | **Partially Feasible** |
| **T-Pot** | Pi 4 (8GB) / Pi 5 | 5.5 GB – 7.8 GB | 80% – 100% (Multiple Docker containers)12 | Fragmented Multi-Container JSON (Elasticsearch index format)12 | Extremely Complex (ELK pipeline orchestration required)13 | **Not Feasible** |

### **OpenCanary Ecosystem Analysis**

OpenCanary acts as a highly customizable, low-interaction daemon designed to mimic common enterprise protocol endpoints (including FTP, HTTP, SMB, RDP, and VNC) on network interfaces2. Written in Python, it runs efficiently as an unprivileged daemon under the Twisted framework1.  
OpenCanary's minimal memory footprint (typically under 35 MB) makes it ideal for deployment on lower-end hardware like the Raspberry Pi Zero 2 W15. The daemon emits clean, single-line JSON entries directly to file paths, syslogs, or HTTP webhooks2. These structures allow Kaia's parsing boundaries to ingest events without heavy deserialization operations or token-parsing delays.

### **Cowrie Ecosystem Analysis**

Cowrie functions as a medium-to-high interaction honeypot, emulating SSH and Telnet command-line shells6. Instead of capturing raw connection attempts, Cowrie implements an interactive, simulated UNIX file system6. It logs brute-force authentication coordinates, keystroke transcripts, terminal interactions, and payload delivery files (dropped via wget or curl) in structured JSON format6.  
Because it implements full cryptographic SSH handshakes, Cowrie's CPU and RAM requirements scale linearly with concurrent connection attempts5. A Raspberry Pi 3 or Compute Module 4 easily absorbs these processes under realistic LAN threat environments5. The resulting telemetry is dense and descriptive, utilizing highly standardized schema variables such as eventid (e.g., cowrie.login.success, cowrie.session.command), providing precise metrics for threat analysis8.

### **DShield Honeypot Ecosystem Analysis**

The DShield sensor acts as a low-interaction gateway client designed to feed global firewall, HTTP, and Cowrie-derived logs back to the SANS Internet Storm Center10. While DShield has a light resource footprint, its automated installer makes aggressive, non-standard changes to the host environment—such as disabling IPv6 interfaces, forcefully redirecting local SSH services to port 12222, and installing local MySQL configurations10.  
Additionally, DShield relies on active outbound internet routing to submit reports to the SANS REST API every 30 minutes10. In an air-gapped or isolated local system environment, extracting localized telemetry from DShield requires overriding its internal client reporting loop, making it less suitable than standard standalone deployments20.

### **T-Pot Ecosystem Analysis**

T-Pot is an all-in-one multi-honeypot platform operating inside a complex Docker-compose environment backed by Elasticsearch, Logstash, and Kibana (ELK)12. With a minimum system requirement of 8 GB (preferably 16 GB) of RAM and substantial disk space, it is fundamentally incompatible with lower-end Raspberry Pi nodes12.  
Furthermore, exposing a massive ELK stack and dozens of dockerized listening services at the network edge significantly increases the local attack surface, presenting a substantial architectural vulnerability that contradicts a hardened security posture13.

## **Ingestion Mechanics & Boundary Protection**

Establishing a secure log ingestion pipeline requires preventing lateral movement to the main host, even if an adversary achieves a full host compromise (root-level container escape) on the Raspberry Pi node7. The design must guarantee that no incoming data channel can be manipulated to compromise or disrupt Kaia’s processing core.

\+---------------------------------------------------------------------------------------------------+  
| PHYSICAL / LOGICAL ISOLATION SEGMENT                                                             |  
|                                                                                                   |  
|  \+--------------------+             Asymmetric UDP Broadcast              \+--------------------+  |  
|  | Raspberry Pi       |     \[JSON \+ SHA-256 Ed25519 Signature\]            | Arch Linux Host    |  |  
|  | (Honeypot Node)    |                                                   | (Kaia Core)        |  |  
|  |                    |                     No TCP Handshake              |                    |  |  
|  | \+----------------+ |                     No State Tracking             | \+----------------+ |  |  
|  | | SSH/FTP Daemons| |                                                   | | Physical RX    | |  |  
|  | \+----------------+ |                                                   | | Interface Only | |  |  
|  |         |          |                                                   | \+----------------+ |  |  
|  |         v          |                     Physical Fiber Link           |         |          |  |  
|  | \+----------------+ |             \+---------------------------------+   |         v          |  |  
|  | | HSM / TPM      | |             |  \+---------------------------+  |   | \+----------------+ |  |  
|  | | (Sign Payload) | |============\>|  | Physical Optical Diode    |  |==\>| | libnftables    | |  |  
|  | \+----------------+ |             |  | (Transmit Fiber Core Only)|  |   | | JSON Parser    | |  |  
|  |                    |             |  \+---------------------------+  |   | \+----------------+ |  |  
|  \+--------------------+             \+---------------------------------+   \+--------------------+  |  
\+---------------------------------------------------------------------------------------------------+

### **Comparative Ingestion Topologies**

The table below contrasts the security vectors of three proposed ingestion mechanics:

| Evaluation Vector | Topology 1: mTLS Syslog-ng Forwarder | Topology 2: Authenticated UDP State Pushing | Topology 3: Optically Isolated Physical Diode |
| :---- | :---- | :---- | :---- |
| **Physical Isolation** | Low (Standard bidirectional Ethernet) | Medium (Bidirectional disabled via soft config) | **Absolute** (Physical return line severed)22 |
| **Exploitation Resistance** | Low-Medium (TLS handshake stack exposed to adversary) | High (Stateless parsing, zero back-channel) | **Maximum** (Exploitation cannot write back to Pi) |
| **Transport Reliability** | **Maximum** (TCP sequence verification, retransmissions) | Medium-High (Requires sequence indexing) | Low-Medium (Requires application-layer forward error correction) |
| **Key Exposure Impact** | Compromise allows fake client spoofing to Host listener | Compromise allows fake message generation | Minimal (Host reads packet queue, no return routing possible) |

### **Topology 1: Mutual TLS (mTLS) Syslog-ng Pipeline**

This topology uses a secure TCP channel over port 651424. Both the honeypot node and the Kaia host authenticate each other using certificates issued by an offline Certificate Authority (CA)25. While this ensures data confidentiality and mutual identity verification, it introduces structural risks.  
If an attacker achieves a root-level escape on the Pi, they can extract the client private key from memory or disk storage27. The adversary can then use these credentials to establish a legitimate TLS session with the Kaia host. From there, they can send malicious payloads designed to exploit vulnerabilities in the host's log-parsing engine.

### **Topology 2: Authenticated UDP State Pushing**

This approach implements a software-enforced unidirectional flow. The Raspberry Pi node converts honeypot alerts into compressed, newline-delimited JSON objects. These payloads are cryptographically signed using an Ed25519 private key securely stored in a hardware-isolated Trusted Platform Module (TPM 2.0) or Hardware Security Module (HSM) connected via SPI on the Pi’s GPIO header28. The signed JSON payload is then transmitted as a single, stateless UDP packet to a dedicated socket on the Kaia host.  
On the host side, the incoming interface drop-rules are configured to accept *only* incoming UDP packets on the target port, dropping all other outbound and inbound traffic. Because there is no bidirectional TCP handshake, the host socket remains hidden from scanning tools. If the Pi is compromised, the attacker cannot establish a reverse Shell or TCP connection because the host completely drops all outgoing communication.

### **Topology 3: Optically Isolated Physical Data Diode**

To achieve absolute physical protection, this topology implements a hardware-based data diode22. A dual-strand fiber media converter connects the Pi and the host28. To enforce unidirectional transmission, the physical receiving fiber strand (RX) is unplugged and cut on the Pi’s side, while the physical transmitting strand (TX) is disabled on the host’s side28.  
This physical modification makes it impossible for light to travel from the host back to the honeypot node22. Telemetry flows strictly one-way as raw UDP broadcast frames. To prevent packet loss due to the lack of TCP retransmission requests, the Pi's streaming client uses an erasure coding algorithm. This algorithm embeds parities and frame counters, allowing the host to reconstruct missing datagrams at the application layer.

## **Dynamic Feedback Loops and Hardening**

Integrating LAN honeypot data with Kaia’s security pipelines enables automated, proactive defense adjustments across all enforcement layers. This system updates threat feeds, adjusts local network firewalls, and modifies sandboxed execution environments dynamically, without requiring human intervention.

### **Threat Intelligence Pipeline Integration**

All incoming, validated honeypot alerts are ingested by a background parsing engine and written to the local SQLite database located at /storage/threat\_intel/reputation.db29. This database utilizes transactional writes with strict schema enforcement.

SQL  
CREATE TABLE IF NOT EXISTS reputation\_cache (  
    ipv4\_address TEXT PRIMARY KEY CHECK(ipv4\_address LIKE '%.%.%.%'),  
    asn\_identifier INTEGER DEFAULT 0,  
    malicious\_score INTEGER CHECK(malicious\_score BETWEEN 0 AND 100),  
    first\_seen\_epoch INTEGER NOT NULL,  
    last\_seen\_epoch INTEGER NOT NULL,  
    threat\_vector TEXT CHECK(threat\_vector IN ('SSH\_BRUTE', 'PORT\_SCAN', 'MALWARE\_DROP', 'VNC\_EXPLOIT', 'GENERIC\_SCAN')),  
    associated\_identifier TEXT DEFAULT NULL,  
    payload\_sha256 TEXT DEFAULT NULL CHECK(length(payload\_sha256) \= 64 OR payload\_sha256 IS NULL),  
    rule\_ttl\_seconds INTEGER NOT NULL,  
    expiration\_epoch INTEGER NOT NULL  
);

CREATE INDEX IF NOT EXISTS idx\_expiration ON reputation\_cache(expiration\_epoch);  
CREATE INDEX IF NOT EXISTS idx\_threat\_vector ON reputation\_cache(threat\_vector);

When an alert is received, the host database engine executes an insert-or-update query. This query dynamically increases the threat severity score and recalculates the dynamic Rule Time-To-Live (TTL) using a customized exponential decay formula:

$$\\text{Decay\\\_TTL} \= \\text{TTL}\_{\\text{base}} \\times e^{\\alpha \\cdot \\text{Attacks}}$$

SQL  
INSERT INTO reputation\_cache (  
    ipv4\_address,   
    asn\_identifier,   
    malicious\_score,   
    first\_seen\_epoch,   
    last\_seen\_epoch,   
    threat\_vector,   
    associated\_identifier,   
    payload\_sha256,   
    rule\_ttl\_seconds,   
    expiration\_epoch  
) VALUES (  
    :ip, :asn, :score, :now, :now, :vector, :assoc, :sha, :ttl, :now \+ :ttl  
) ON CONFLICT(ipv4\_address) DO UPDATE SET  
    malicious\_score \= MIN(100, reputation\_cache.malicious\_score \+ 15),  
    last\_seen\_epoch \= excluded.last\_seen\_epoch,  
    rule\_ttl\_seconds \= MIN(604800, reputation\_cache.rule\_ttl\_seconds \* 2),  
    expiration\_epoch \= excluded.last\_seen\_epoch \+ MIN(604800, reputation\_cache.rule\_ttl\_seconds \* 2);

An internal asynchronous cleanup process runs every 60 seconds to automatically purge expired entries:

SQL  
DELETE FROM reputation\_cache WHERE expiration\_epoch \<= strftime('%s', 'now');

### **Deterministic Policy Adaptation via Policy Gate**

When an IP address exceeds a malicious threshold ($\\ge 75$) in reputation.db, policy\_gate.py bypasses the AI planning loops to trigger real-time, deterministic defense adjustments across the host.  
First, policy\_gate.py interacts with the host’s local netfilter layer to block the source IP30. Rather than calling slow shell scripts, the Python interface uses the native libnftables JSON API to insert the IP directly into a dynamic blocklist set32:

Python  
import nftables  
import json

def insert\_dynamic\_nft\_block(ip\_address, ttl\_seconds):  
    """  
    Directly binds to the kernel's netlink subsystem using libnftables JSON API,  
    programmatically appending the target IP to the reactive dropping list.  
    """  
    nft \= nftables.Nftables()  
    nft.set\_json\_output(True)  
      
    cmd\_payload \= {  
        "nftables": \[  
            {  
                "add": {  
                    "element": {  
                        "family": "inet",  
                        "table": "kaia\_filter",  
                        "name": "isolated\_honeypot\_peers",  
                        "elem": \[  
                            f"{ip\_address} timeout {ttl\_seconds}s"  
                        \]  
                    }  
                }  
            }  
        \]  
    }  
      
    rc, output, error \= nft.json\_cmd(cmd\_payload)  
    if rc \!= 0:  
        raise RuntimeError(f"nftables verification failed: {error}")

The underlying base configuration template in /etc/nftables.conf must define a pre-existing dynamic set within the inet family to support these dynamic inserts30:

Code snippet  
table inet kaia\_filter {  
    set isolated\_honeypot\_peers {  
        type ipv4\_addr  
        flags dynamic, timeout  
        timeout 24h  
    }

    chain input\_filter {  
        type filter hook input priority \-10; policy accept;  
        ip saddr @isolated\_honeypot\_peers drop  
    }  
}

Simultaneously, policy\_gate.py updates the configuration dictionary used by host\_executor.py34. Normally, host\_executor.py spawns its bwrap sandboxes with network namespace loopback sharing enabled35.  
However, if an active process configuration maps to a threat origin listed in reputation.db, host\_executor.py overrides the sandbox parameters. It drops the network namespace entirely during execution by replacing the default network parameters with the \--unshare-net parameter35:

Bash  
bwrap \--ro-bind /usr /usr \\  
      \--ro-bind /lib /lib \\  
      \--ro-bind /etc /etc \\  
      \--dir /tmp \\  
      \--proc /proc \\  
      \--dev /dev \\  
      \--unshare-all \\  
      \--unshare-net \\  
      \--die-with-parent \\  
      /bin/bash \-c "your\_sandboxed\_program"

This ensures absolute isolation, preventing any potentially compromised sandbox processes from communicating with external networks or executing reverse-shell callbacks37.

### **eBPF and File Integrity Monitoring (FIM) Cross-Correlation**

To detect sophisticated lateral movement attempts, Kaia relies on a kernel-level correlation engine that matches adjacent honeypot network events with local system telemetry.

\+---------------------------------------------------------------------------------------------------+  
| KERNEL SEGMENT (eBPF & fanotify)                                                                  |  
|                                                                                                   |  
|  \[Honeypot Network Alerts\]                                                                        |  
|  (Source IP: 192.168.99.105)                                                                      |  
|             |                                                                                     |  
|             v                                                                                     |  
|  \+--------------------+                                                                           |  
|  | Correlation Engine | \<======= \[Time Window: Delta-T \<= 5000ms\]                                 |  
|  \+--------------------+                                                                           |  
|             ^                                                                                     |  
|             | (Matches Destination IP: 192.168.99.105)                                            |  
|             |                                                                                     |  
|  \[sys\_enter\_connect Event\]                                                                        |  
|  (Origin PID: 12405 / Exec: 'python-runner')                                                       |  
|                                                                                                   |  
\+---------------------------------------------------------------------------------------------------+

This correlation heuristic evaluates system states using the following logic:

$$\\text{LateralMovementDetected} \\iff \\exists t\_{\\text{event}} \\in \[t\_{\\text{current}} \- \\Delta t, t\_{\\text{current}}\] \\quad \\text{such that} \\quad (\\text{IP}\_{\\text{dest\\\_ebpf}} \= \\text{IP}\_{\\text{src\\\_honeypot}})$$  
Where $\\Delta t \= 5000\\text{ ms}$ (to account for packet processing and telemetry propagation delays).  
This correlation is implemented in Python as follows:

Python  
import time

def evaluate\_lateral\_movement(reputation\_cache, ebpf\_socket\_map):  
    """  
    Evaluates correlation between active honeypot alerts and local TCP connection events.  
    Matches internal outbound connections to addresses that are currently probing the Pi.  
    """  
    current\_epoch \= int(time.time())  
    lookback\_window \= 5.0  \# seconds  
      
    \# Extract suspicious IPs seen in the lookback window  
    suspicious\_ips \= reputation\_cache.query\_active\_threat\_ips(  
        since\_epoch=current\_epoch \- lookback\_window  
    )  
      
    \# Retrieve active connections tracked via host eBPF sys\_enter\_connect probes  
    active\_connections \= ebpf\_socket\_map.get\_active\_outgoing\_destinations()  
      
    for conn in active\_connections:  
        dest\_ip \= conn\['destination\_ip'\]  
        if dest\_ip in suspicious\_ips:  
            \# An internal process is communicating with an IP currently targeting the honeypot  
            trigger\_host\_lockdown(  
                offending\_pid=conn\['pid'\],  
                target\_ip=dest\_ip,  
                process\_name=conn\['comm'\]  
            )

If an outgoing socket connection attempt is detected to an IP address that is currently probing the adjacent honeypot node, the correlation engine identifies a possible internal host compromise or unauthorized scanning attempt39.  
The system immediately isolates the initiating container by terminating the process (SIGKILL), updating the policy engine rule set to drop all traffic to that process ID, and logging the event to the audit ledger.

## **Curses Dashboard Ingestion Pipeline**

To display telemetry from the external honeypot nodes in the kaia\_dashboard.py interface without causing UI latency, the dashboard must handle incoming connections asynchronously. The main loop of a curses-based dashboard blocks on keyboard inputs and frame redraws. Running network socket listeners inside this main loop would block the UI, causing screen freezes and input lag41.

\+--------------------------------------------------------------------------+  
| kaia\_dashboard.py ARCHITECTURE                                           |  
|                                                                          |  
|  \+--------------------+                     \+-------------------------+  |  
|  | Curses UI Thread   |                     | asyncio Telemetry Thread|  |  
|  |                    |                     |                         |  |  
|  |  (Periodic Render  |                     |  (Non-blocking Socket)  |  |  
|  |   Loop, \~100ms)    |                     |  \[asyncio.start\_server\] |  |  
|  |         |          |                     |            |            |  |  
|  |   Non-blocking     |                     |    JSON Validation      |  |  
|  |   Queue Poll       |                     |            |            |  |  
|  |   get\_nowait()     |                     |      Queue Put          |  |  
|  |         v          |                     |            v            |  |  
|  |   \[Thread Queue\] \<====================================+            |  |  
|  |         |          |                     |                         |  |  
|  |   Update Panels    |                     |                         |  |  
|  \+--------------------+                     \+-------------------------+  |  
\+--------------------------------------------------------------------------+

To maintain UI performance, socket collection is offloaded to a separate worker thread running an asynchronous asyncio event loop43. This worker thread processes network I/O in the background and writes validated telemetry to a thread-safe queue.Queue43.  
The implementation details for the dashboard data integration are outlined below:

Python  
import asyncio  
import queue  
import threading  
import json  
import curses

\# Thread-safe queue for telemetry exchange  
telemetry\_queue \= queue.Queue(maxsize=1024)

class TelemetryProtocol(asyncio.DatagramProtocol):  
    """  
    Asynchronous UDP Protocol handler that validates incoming JSON alerts  
    and pushes them to the thread-safe telemetry queue without blocking.  
    """  
    def datagram\_received(self, data, addr):  
        try:  
            payload \= json.loads(data.decode('utf-8'))  
            \# Basic schema validation  
            if "node\_id" in payload and "src\_host" in payload:  
                \# Add to queue without blocking if space is available  
                telemetry\_queue.put\_nowait(payload)  
        except (json.JSONDecodeError, KeyError, queue.Full):  
            \# Gracefully handle validation failures and full queue anomalies  
            pass

def run\_asyncio\_telemetry\_loop(host, port):  
    """  
    Launches the dedicated asyncio event loop in an isolated thread.  
    """  
    loop \= asyncio.new\_event\_loop()  
    asyncio.set\_event\_loop(loop)  
      
    \# Initialize UDP socket listener bound to the isolated network interface  
    listen\_coro \= loop.create\_datagram\_endpoint(  
        lambda: TelemetryProtocol(),  
        local\_addr=(host, port)  
    )  
      
    loop.run\_until\_complete(listen\_coro)  
    try:  
        loop.run\_forever()  
    finally:  
        loop.close()

\# In kaia\_dashboard.py: Launch the thread prior to curses initialization  
telemetry\_thread \= threading.Thread(  
    target=run\_asyncio\_telemetry\_loop,  
    args=("192.168.10.1", 9005), \# Bound to the secure ingestion segment  
    daemon=True  
)  
telemetry\_thread.start()

Within the main curses rendering loop of kaia\_dashboard.py, the UI reads from the thread-safe queue at each redraw tick using queue.get\_nowait(). This prevents the UI from blocking if no new network events have occurred:

Python  
def update\_dashboard\_state(stdscr, dashboard\_components):  
    """  
    Pulls data from the asynchronous ingestion queue during each rendering tick.  
    """  
    new\_alerts \= \[\]  
    while True:  
        try:  
            \# Attempt to pull from queue; raises Empty exception immediately if vacant  
            alert \= telemetry\_queue.get\_nowait()  
            new\_alerts.append(alert)  
            telemetry\_queue.task\_done()  
        except queue.Empty:  
            break  
              
    \# Process new metrics and update UI elements  
    for alert in new\_alerts:  
        dashboard\_components\['metrics\_panel'\].update\_stats(alert)  
        dashboard\_components\['ledger\_panel'\].append\_log(  
            f"\[{alert\['node\_id'\]}\] Threat event from {alert\['src\_host'\]}:{alert\['src\_port'\]}"  
        )

The parsed log elements are mapped onto the terminal screen UI panel layout:

\+----------------------------------\[Kaia Monitor Dashboard v1.4\]----------------------------------+  
| CPU: \[|||||||||||||||   55.2%\]  Memory: \[||||||||||||||||||   68.5%\]  DB Status: ACTIVE          |  
\+-------------------------------------------------+-----------------------------------------------+  
| LOCAL Telemetry (fanotify & eBPF Event log)     | ADJACENT HONEYPOT TELEMETRY PANEL             |  
| 14:02:12 EXEC /usr/bin/bwrap PID: 12405 \[SAFE\]  | Node ID      Source Host      Vector   Status |  
| 14:02:44 FIM Write /etc/passwd PID: 341 \[ALERT\] | \--------------------------------------------- |  
| 14:03:01 SYS\_SETUID Call by PID: 8059   \[AUDIT\] | node-pi-01   192.168.99.104   SSH\_BRU  ACTIVE |  
|                                                 | node-pi-01   192.168.99.155   VNC\_EXP  ACTIVE |  
|                                                 | node-pi-02   192.168.99.201   PORT\_SC  MUTED  |  
\+-------------------------------------------------+-----------------------------------------------+  
| DYNAMIC ADAPTIVE FIREWALL POLICIES SET (nftables)                                               |  
| Target Source IP   Attack Vector   Identified Confidence   TTL Remaining    Active Penalty      |  
| \----------------------------------------------------------------------------------------------- |  
| 192.168.99.104     SSH\_BRUTE       98%                     23:59:44         BLOCK\_INPUT\_STREAM  |  
| 192.168.99.155     VNC\_EXPLOIT     85%                     11:32:01         UNSHARE\_NS\_NET      |  
\+-------------------------------------------------------------------------------------------------+

## **Failure Modes & Mechanics**

Deploying distributed embedded security systems introduces unique runtime edge-case failure modes. The integration architecture must be resilient to denial-of-service attempts, hardware failures, and cryptographic key exposure.  
The table below catalogs key system failure scenarios and their design mitigations:

| Failure Scenario Code | System Vector | Primary Mechanical Failure Mode | Dynamic Mitigation Control | Post-Failure Safe State |
| :---- | :---- | :---- | :---- | :---- |
| **FMC-001** | Telemetry Pipeline | Incoming alert storm exhausts memory queue buffer. | Memory-bounded queue size with tail-drop policies45. | Queue drops incoming alerts; dashboard continues rendering. |
| **FMC-002** | Cryptographic Subsystem | Pi node root compromise exposes local cryptographic signing keys27. | Isolated hardware keys; epoch-based protocol handshakes. | Public key is revoked in reputation engine; Pi node is quarantined. |
| **FMC-003** | Interface Layer | physical data diode link failure or network segment break. | System heartbeat checker on host UDP listener. | Host alerts system operator; blocks local fallback interfaces. |
| **FMC-004** | Execution Sandbox | bwrap configuration fails to apply \--unshare-net parameter46. | Failure-critical exit states in host\_executor.py37. | Execution fails immediately with safe exit codes. |

### **Mechanical Deep-Dive: FMC-001 (Alert Storm Memory Starvation)**

During a distributed denial-of-service (DDoS) attack or an aggressive network scan targeting the adjacent honeypots, the Pi nodes may generate thousands of JSON log lines per second. If the host queue buffer fails to process these logs quickly, memory consumption can rise rapidly. This can trigger the Linux kernel's Out-Of-Memory (OOM) killer, potentially terminating critical host processes.

\+------------------------------------------------------------------------------------+  
| MECHANICAL FLOW OF FMC-001 MITIGATION                                              |  
|                                                                                    |  
| \[Incoming UDP Packet Volume\]                                                       |  
|              |                                                                     |  
|              v                                                                     |  
|   \+--------------------+                                                           |  
|   | OS Socket Buffer   | \===== (Buffer Limit: /proc/sys/net/core/rmem\_max)         |  
|   \+--------------------+                                                           |  
|              |                                                                     |  
|              v                                                                     |  
|   \+--------------------+                                                           |  
|   | Datagram Listener  |                                                           |  
|   \+--------------------+                                                           |  
|              |                                                                     |  
|              v                                                                     |  
|   \+--------------------+                                                           |  
|   | queue.Queue Check  | \===== (Is Queue Full?)                                    |  
|   \+--------------------+                                                           |  
|          /         \\                                                               |  
|        YES          NO                                                             |  
|        /             \\                                                             |  
|       v               v                                                            |  
| \[Discard Packet\]   \[Push to Queue\]                                                 |  
| (Drop Packet       (Render Panel Redraws)                                          |  
|  and Log Drop)                                                                     |  
\+------------------------------------------------------------------------------------+

To prevent this, the host limits its OS-level socket buffer size using /proc/sys/net/core/rmem\_max. The ingestion listener utilizes a thread-safe queue.Queue with a fixed capacity of 1024 slots43.  
If the queue fills up, new datagrams are dropped immediately, and a counter tracking lost frames is incremented in memory45. This design protects the host's memory, ensuring the curses render loop maintains a predictable frame rate.

### **Mechanical Deep-Dive: FMC-002 (Cryptographic Compromise)**

If an attacker exploits a zero-day vulnerability in the emulated services and gains root shell access on the Raspberry Pi7, they can access the local filesystem and memory space27. If the cryptographic key pair used to sign alerts is stored directly on the SD card, the attacker can extract it to craft and sign fraudulent telemetry, attempting to pollute reputation.db.  
To mitigate this risk, the private key is stored within a hardware TPM module attached to the Pi's GPIO header. Key generation and cryptographic signatures are executed directly within the secure boundary of the TPM chip, ensuring that the private key cannot be read or extracted by user-space software.  
Furthermore, all signed datagrams contain an asymmetric signature payload format combined with an increasing monotonic epoch sequence counter. The host checks this sequence counter and rejects any packets that are older than the expected timing threshold, preventing replay attacks.

## **Network Topologies**

To isolate the honeypot infrastructure while still allowing efficient telemetry ingestion, a dedicated out-of-band management segment is implemented.

                        \+----------------------------+  
                        |   Core LAN Router/Switch   |  
                        \+----------------------------+  
                          /                        \\  
                         /                          \\  
  \[Trunk: VLAN 10, 99\]  /                            \\  \[Access: VLAN 99\]  
                       /                              \\  
                      v                                v  
         \+--------------------------+     \+--------------------------+  
         | Arch Linux Host          |     | Raspberry Pi Node        |  
         | (Kaia Security Host)     |     | (Honeypot Client)        |  
         |                          |     |                          |  
         | Physical Port: enp3s0    |     | Physical Port: eth0      |  
         |   \- IP: 192.168.10.5     |     |   \- IP: 192.168.99.20    |  
         |                          |     \+--------------------------+  
         | Virtual Interface:       |  
         | enp3s0.99                |  
         |   \- IP: 192.168.99.5     |  
         \+--------------------------+

### **VLAN Allocation Matrix**

The table below outlines the network isolation boundaries:

| Network Segment Name | Tag ID | Subnet CIDR Range | Allowed Host Communications | Physical Interface Bindings |
| :---- | :---- | :---- | :---- | :---- |
| **Production Segment** | VLAN 10 | 192.168.10.0/24 | Primary host administrative ports and secure terminals. | enp3s0 |
| **Decoy Network Segment** | VLAN 99 | 192.168.99.0/24 | Isolated honeypot endpoints and targeted target profiles. | enp3s0.99 (Host side, inbound only), eth0 (Pi side)17 |

### **Routing Table Configuration**

The Arch Linux host uses virtual interface tagging to separate telemetry data from administrative access17. The primary physical interface (enp3s0) is assigned to VLAN 10 (192.168.10.5/24), while virtual interface enp3s0.99 is bound to the isolated honeypot subnet (192.168.99.5/24).  
The host routing tables are configured to enforce strict data isolation:

\# ip route show  
default via 192.168.10.1 dev enp3s0 proto dhcp metric 100   
192.168.10.0/24 dev enp3s0 proto kernel scope link src 192.168.10.5   
192.168.99.0/24 dev enp3s0.99 proto kernel scope link src 192.168.99.5 metric 200

To prevent the honeypot segment from reaching production systems, the router configuration drops all packet routing between VLAN 99 and VLAN 107.  
On the host, firewall policies are set to drop all outgoing packets on the enp3s0.99 interface, ensuring that the host can only read raw incoming UDP data from the honeypot segment:

Bash  
\# nft add rule inet kaia\_filter output oifname "enp3s0.99" drop

This ensures that even if an attacker compromises the honeypot node, they cannot scan or access the host via VLAN 99\.

## **Declarative Requirements Modifications**

To deploy this integrated architecture, specific configuration changes must be applied to the honeypot node, the policy gate, and the host execution layers.

### **Node Configurations**

#### **/etc/opencanaryd/opencanary.conf**

The following configuration initializes target services and defines a custom webhook endpoint to forward telemetry1:

JSON  
{  
    "device.node\_id": "node-pi-01",  
    "ftp.enabled": true,  
    "ftp.port": 21,  
    "http.enabled": true,  
    "http.port": 80,  
    "ssh.enabled": false,  
    "telnet.enabled": true,  
    "telnet.port": 23,  
    "logger": {  
        "class": "PyLogger",  
        "kwargs": {  
            "handlers": {  
                "Webhook": {  
                    "class": "opencanary.logger.WebhookHandler",  
                    "url": "http://192.168.99.5:9005/alerts",  
                    "method": "POST",  
                    "status\_code": 200,  
                    "headers": {  
                        "Content-Type": "application/json",  
                        "X-Kaia-Auth-Header": "K\_Ed25519\_Sec\_Verify\_Signature"  
                    }  
                }  
            }  
        }  
    }  
}

#### **/home/cowrie/cowrie/etc/cowrie.cfg**

The Cowrie configuration emulates SSH on port 2222, saving structured JSON logs locally to /home/cowrie/cowrie/var/log/cowrie/cowrie.json6:

Ini, TOML  
\[honeypot\]  
hostname \= arch-dev-sysadmin-prod  
log\_path \= var/log/cowrie  
download\_path \= var/lib/cowrie/downloads  
filesystem \= share/cowrie/fs.pickle

\[ssh\]  
enabled \= true  
listen\_addresses \= 0.0.0.0  
listen\_port \= 2222

\[telnet\]  
enabled \= false

The system redirects incoming SSH traffic on port 22 to Cowrie on port 2222 using iptables, allowing Cowrie to run as a low-privilege user6:

Bash  
\# iptables \-t nat \-A PREROUTING \-p tcp \--dport 22 \-j REDIRECT \--to-port 2222

### **Policy Gate Schema Mapping**

The out-of-process validator, policy\_gate.py, validates all incoming payloads against a strict JSON schema before committing updates to the system. This schema ensures that alert data formats are strictly structured, preventing SQL injection or payload contamination attempts29:

JSON  
{  
    "$schema": "http://json-schema.org/draft-07/schema\#",  
    "title": "HoneypotIngestedTelemetrySchema",  
    "type": "object",  
    "properties": {  
        "node\_id": {  
            "type": "string",  
            "pattern": "^\[a-zA-Z0-9\_-\]{4,32}$"  
        },  
        "src\_host": {  
            "type": "string",  
            "format": "ipv4"  
        },  
        "src\_port": {  
            "type": "integer",  
            "minimum": 1,  
            "maximum": 65535  
        },  
        "dst\_host": {  
            "type": "string",  
            "format": "ipv4"  
        },  
        "dst\_port": {  
            "type": "integer",  
            "minimum": 1,  
            "maximum": 65535  
        },  
        "threat\_vector": {  
            "type": "string",  
            "enum": \["SSH\_BRUTE", "PORT\_SCAN", "MALWARE\_DROP", "VNC\_EXPLOIT", "GENERIC\_SCAN"\]  
        },  
        "timestamp": {  
            "type": "integer",  
            "minimum": 1700000000  
        },  
        "signature\_proof": {  
            "type": "string",  
            "pattern": "^\[a-fA-F0-9\]{128}$"  
        }  
    },  
    "required": \["node\_id", "src\_host", "src\_port", "threat\_vector", "timestamp", "signature\_proof"\],  
    "additionalProperties": false  
}

### **Systemd Ingestion Units**

#### **Client Log Forwarder (/etc/systemd/system/kaia-forwarder.service)**

This service runs on the Raspberry Pi node, forwarding local honeypot alerts to the host as signed UDP datagrams:

Ini, TOML  
\[Unit\]  
Description=Kaia Decoy Cryptographic Telemetry Forwarder  
After=network.target  
StartLimitIntervalSec=0

\[Service\]  
Type=simple  
Restart=always  
RestartSec=3  
User=root  
WorkingDirectory=/opt/kaia\_sensor  
ExecStart=/usr/bin/python3 /opt/kaia\_sensor/stream\_client.py \--src /var/log/cowrie/cowrie.json \--dst 192.168.99.5 \--port 9005 \--key /etc/tpm\_key\_sign.bin

\[Install\]  
WantedBy=multi-user.target

#### **Host Receiver Service (/etc/systemd/system/kaia-receiver.service)**

This service runs on the Arch Linux host, listening for incoming UDP telemetry from the honeypot segment and routing it to the Policy Gate:

Ini, TOML  
\[Unit\]  
Description=Kaia Host Telemetry Ingest Receiver  
After=network.target

\[Service\]  
Type=simple  
Restart=always  
RestartSec=1  
User=kaia\_runtime  
ExecStart=/usr/bin/python3 /opt/kaia\_core/receiver\_daemon.py \--bind 192.168.99.5 \--port 9005 \--db /storage/threat\_intel/reputation.db

\[Install\]  
WantedBy=multi-user.target

## **Architectural Synthesis**

Integrating hardware honeypots into the Project Kaia architecture establishes a highly effective, automated local defense pipeline. By deploying lightweight single-board computers inside an isolated network segment (VLAN 99), the system captures edge attack trends without exposing production assets. The telemetry is cryptographically signed and streamed to the host via stateless UDP, ensuring strong boundary protection and blocking lateral movement.  
By feeding this data into reputation.db, the Policy Gate (policy\_gate.py) dynamically updates firewall sets (libnftables API) and restricts process execution environments (host\_executor.py utilizing bwrap) in real-time. Finally, the curses dashboard (kaia\_dashboard.py) processes these threat vectors on a dedicated, non-blocking background thread, preserving interface performance. This unified architecture allows Project Kaia to actively adapt and harden its local host defense posture against target scanning and exploitation attempts on the adjacent network.

#### **Works cited**

1. Configuration — OpenCanary 0.9 documentation \- Read the Docs, [https://opencanary.readthedocs.io/en/latest/starting/configuration.html](https://opencanary.readthedocs.io/en/latest/starting/configuration.html)  
2. OpenCanary \- Active Defense Harbinger Distribution, [https://adhdproject.github.io/\#\!Tools/Attribution/OpenCanary.md](https://adhdproject.github.io/#!Tools/Attribution/OpenCanary.md)  
3. Integrating OpenCanary & DShield \- /dev/random, [https://blog.rootshell.be/2017/02/15/integrating-opencanary-dshield/](https://blog.rootshell.be/2017/02/15/integrating-opencanary-dshield/)  
4. opencanary/docs/alerts/webhook.md at master \- GitHub, [https://github.com/thinkst/opencanary/blob/master/docs/alerts/webhook.md](https://github.com/thinkst/opencanary/blob/master/docs/alerts/webhook.md)  
5. \[Guest Diary\] Anatomy of a Linux SSH Honeypot Attack: Detailed Analysis of Captured Malware \- SANS Internet Storm Center, [https://isc.sans.edu/diary/32024](https://isc.sans.edu/diary/32024)  
6. Running a Cowrie Honeypot: Data and Findings \- ambient\_node, [https://ambientnode.uk/running-a-cowrie-honeypot-data-and-findings](https://ambientnode.uk/running-a-cowrie-honeypot-data-and-findings)  
7. How to Set Up a Honeypot with Cowrie on Ubuntu \- OneUptime, [https://oneuptime.com/blog/post/2026-03-02-how-to-set-up-a-honeypot-with-cowrie-on-ubuntu/view](https://oneuptime.com/blog/post/2026-03-02-how-to-set-up-a-honeypot-with-cowrie-on-ubuntu/view)  
8. Visualizing aggregates of events \- Kibana \- Discuss the Elastic Stack, [https://discuss.elastic.co/t/visualizing-aggregates-of-events/83311](https://discuss.elastic.co/t/visualizing-aggregates-of-events/83311)  
9. Cowrie honeypot and its Integration with Microsoft Sentinel., [https://techcommunity.microsoft.com/blog/microsoftsentinelblog/cowrie-honeypot-and-its-integration-with-microsoft-sentinel-/4258349](https://techcommunity.microsoft.com/blog/microsoftsentinelblog/cowrie-honeypot-and-its-integration-with-microsoft-sentinel-/4258349)  
10. Honeypots 102: Setting up a SANS Internet Storm Center's DShield Honeypot | by Abdul Issa | InfoSec Write-ups, [https://infosecwriteups.com/honeypots-102-setting-up-a-sans-internet-storm-centers-dshield-honeypot-1ec1774bd949](https://infosecwriteups.com/honeypots-102-setting-up-a-sans-internet-storm-centers-dshield-honeypot-1ec1774bd949)  
11. DShield Honeypot \- SANS Internet Storm Center, [https://isc.sans.edu/honeypot.html](https://isc.sans.edu/honeypot.html)  
12. telekom-security/tpotce: T-Pot \- The All In One Multi Honeypot Platform \- GitHub, [https://github.com/telekom-security/tpotce](https://github.com/telekom-security/tpotce)  
13. Honeypots 104: T-Pot — Your All-in-One Honeypot Platform Guide \- InfoSec Write-ups, [https://infosecwriteups.com/honeypots-104-t-pot-your-all-in-one-honeypot-platform-guide-0ba2643bc597](https://infosecwriteups.com/honeypots-104-t-pot-your-all-in-one-honeypot-platform-guide-0ba2643bc597)  
14. thinkst/opencanary: Modular and decentralised honeypot \- GitHub, [https://github.com/thinkst/opencanary](https://github.com/thinkst/opencanary)  
15. DShield Honeypot \- Medium, [https://medium.com/@josejgp/dshield-honeypot-733c8c231a50](https://medium.com/@josejgp/dshield-honeypot-733c8c231a50)  
16. Enhancing SSH/Telnet Honeypot for Attack Classification and Malware Research \- CEUR-WS.org, [https://ceur-ws.org/Vol-4198/paper51.pdf](https://ceur-ws.org/Vol-4198/paper51.pdf)  
17. DShield Cowrie Honeypot on Raspberry Pi \- AlanIssak, [https://alanissak.com/post/dshield-cowrie-raspberry-pi-honeypot](https://alanissak.com/post/dshield-cowrie-raspberry-pi-honeypot)  
18. Setting Up Honeypot Using Cowrie \- BlueGrid.io, [https://bluegrid.io/edu/setting-up-honeypot-using-cowrie/](https://bluegrid.io/edu/setting-up-honeypot-using-cowrie/)  
19. DShield Raspberry Pi Sensor \- GitHub, [https://github.com/dshield-isc/dshield](https://github.com/dshield-isc/dshield)  
20. DShield and qemu Sitting in a Tree: L-O-G-G-I-N-G \- SANS ISC, [https://isc.sans.edu/diary/30216](https://isc.sans.edu/diary/30216)  
21. T-Pot Version 24.04 released \- Telekom Security, [https://github.security.telekom.com/2024/04/honeypot-tpot-24.04-released.html](https://github.security.telekom.com/2024/04/honeypot-tpot-24.04-released.html)  
22. CompTIA SY0-701 Security+ Study Guide | PDF | Information Security \- Scribd, [https://www.scribd.com/document/828802247/Ultimate-CompTIA-SY0-701-Security-Study-Guide-1-0](https://www.scribd.com/document/828802247/Ultimate-CompTIA-SY0-701-Security-Study-Guide-1-0)  
23. Automating Cyber Threat Intelligence: Tools and Techniques for Enhanced Security Posture, [https://dokumen.pub/automating-cyber-threat-intelligence-tools-and-techniques-for-enhanced-security-posture.html](https://dokumen.pub/automating-cyber-threat-intelligence-tools-and-techniques-for-enhanced-security-posture.html)  
24. papadopouloskyriakos/homelab-infrastructure: Multi-site Kubernetes homelab with BGP anycast (AS214304), Cilium CNI, and GitOps. 14 nodes across 4 countries. \- GitHub, [https://github.com/papadopouloskyriakos/homelab-infrastructure](https://github.com/papadopouloskyriakos/homelab-infrastructure)  
25. Anomaly Detection for Cyber-Physical Systems \- mediaTUM, [https://mediatum.ub.tum.de/doc/1601840/1601840.pdf](https://mediatum.ub.tum.de/doc/1601840/1601840.pdf)  
26. ELK Stack a Comprehensive Guide to Installing and Configuring the ELK Stack \- DEV Community, [https://dev.to/kaustubhyerkade/elk-stack-a-comprehensive-guide-to-installing-and-configuring-the-elk-stack-el7](https://dev.to/kaustubhyerkade/elk-stack-a-comprehensive-guide-to-installing-and-configuring-the-elk-stack-el7)  
27. (PDF) Analysis of Various Vulnerabilities in the Raspbian Operating System and Solutions, [https://www.researchgate.net/publication/361980222\_Analysis\_of\_Various\_Vulnerabilities\_in\_the\_Raspbian\_Operating\_System\_and\_Solutions](https://www.researchgate.net/publication/361980222_Analysis_of_Various_Vulnerabilities_in_the_Raspbian_Operating_System_and_Solutions)  
28. A DIY, low-cost data diode for ICS \- BruCON, [https://files.brucon.org/2017/011\_Arnaud\_Soullier\_Wavestone.pdf](https://files.brucon.org/2017/011_Arnaud_Soullier_Wavestone.pdf)  
29. Cross‑Platform Suppression Telemetry Inquiry — Primary Source Chat-Based Technical Analysis with Google Gemini (Part 1\) | Hunter Storm, [https://hunterstorm.com/federal-whistleblower/coordinated-telemetry-suppression-part-1/](https://hunterstorm.com/federal-whistleblower/coordinated-telemetry-suppression-part-1/)  
30. nftables \- ArchWiki, [https://wiki.archlinux.org/title/Nftables](https://wiki.archlinux.org/title/Nftables)  
31. Hardening SSH: Fail2Ban, Nftables & Cloud Firewalls \- DigitalOcean, [https://www.digitalocean.com/community/tutorials/hardening-ssh-fail2ban](https://www.digitalocean.com/community/tutorials/hardening-ssh-fail2ban)  
32. Add nftables map element using libnftables-json API from python \- Stack Overflow, [https://stackoverflow.com/questions/70239480/add-nftables-map-element-using-libnftables-json-api-from-python](https://stackoverflow.com/questions/70239480/add-nftables-map-element-using-libnftables-json-api-from-python)  
33. nftables rate-limiting against low-effort DDoS attacks \- My blog\_title\_here, [https://blog.fraggod.net/2025/01/16/nftables-rate-limiting-against-low-effort-ddos-attacks.html](https://blog.fraggod.net/2025/01/16/nftables-rate-limiting-against-low-effort-ddos-attacks.html)  
34. AI Agent Sandboxing: 3 Isolation Patterns for 2026 \- Digital Applied, [https://www.digitalapplied.com/blog/ai-agent-sandboxing-isolation-patterns-2026](https://www.digitalapplied.com/blog/ai-agent-sandboxing-isolation-patterns-2026)  
35. Terminal Sandbox: Infrastructure for the Agent Harness \- Qoder, [https://qoder.com/blog/qoder-desktop-sandbox](https://qoder.com/blog/qoder-desktop-sandbox)  
36. Sandboxing Soldatserver with Bubblewrap and Seccomp \- minus' blog \- mnus.de, [https://blog.mnus.de/2020/05/sandboxing-soldatserver-with-bubblewrap-and-seccomp/](https://blog.mnus.de/2020/05/sandboxing-soldatserver-with-bubblewrap-and-seccomp/)  
37. OS-Level Sandboxing: Kernel Isolation for AI Agents \- DEV Community, [https://dev.to/uenyioha/os-level-sandboxing-kernel-isolation-for-ai-agents-3fdg](https://dev.to/uenyioha/os-level-sandboxing-kernel-isolation-for-ai-agents-3fdg)  
38. mxc/docs/bwrap-support/bubblewrap-backend.md at main \- GitHub, [https://github.com/microsoft/mxc/blob/main/docs/bwrap-support/bubblewrap-backend.md](https://github.com/microsoft/mxc/blob/main/docs/bwrap-support/bubblewrap-backend.md)  
39. How to Analyze TCP/IP Stack Performance with eBPF \- OneUptime, [https://oneuptime.com/blog/post/2026-01-07-ebpf-tcp-ip-stack-analysis/view](https://oneuptime.com/blog/post/2026-01-07-ebpf-tcp-ip-stack-analysis/view)  
40. tcpconnect \- TCP Connection Monitoring \- eBPF the Hard Way, [https://ebee.xmigrate.cloud/tools/tcpconnect/](https://ebee.xmigrate.cloud/tools/tcpconnect/)  
41. From Closet to Code: Leveraging Concurrency with Non-Blocking Sockets in Python | by Satish Jasthi | Medium, [https://medium.com/@satishjasthi/from-closet-to-code-leveraging-concurrency-with-non-blocking-sockets-in-python-906958f30817](https://medium.com/@satishjasthi/from-closet-to-code-leveraging-concurrency-with-non-blocking-sockets-in-python-906958f30817)  
42. How to interface blocking and non-blocking code with asyncio \- Stack Overflow, [https://stackoverflow.com/questions/23898363/how-to-interface-blocking-and-non-blocking-code-with-asyncio](https://stackoverflow.com/questions/23898363/how-to-interface-blocking-and-non-blocking-code-with-asyncio)  
43. asyncio and free-threaded Python — Python 3.14.6 documentation, [https://docs.python.org/3/library/asyncio-threading.html](https://docs.python.org/3/library/asyncio-threading.html)  
44. Scaling asyncio on Free-Threaded Python \- Quansight Labs, [https://labs.quansight.org/blog/scaling-asyncio-on-free-threaded-python](https://labs.quansight.org/blog/scaling-asyncio-on-free-threaded-python)  
45. eBPF lost events | Itay as a Service, [http://blog.itaysk.com/2020/04/20/ebpf-lost-events](http://blog.itaysk.com/2020/04/20/ebpf-lost-events)  
46. bubblewrap inside unprivileged docker · Issue \#505 \- GitHub, [https://github.com/containers/bubblewrap/issues/505](https://github.com/containers/bubblewrap/issues/505)  
47. Setting up Cowrie Honeypot \- Medium, [https://medium.com/@berylben966/setting-up-cowrie-honeypot-aef050546364](https://medium.com/@berylben966/setting-up-cowrie-honeypot-aef050546364)