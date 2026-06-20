# The InfoSec Knowledge Base

This guide provides a comprehensive overview of essential InfoSec tools and foundational concepts, covering their primary functions, advanced use cases, and detailed commands.

---

## Table of Contents
1.  [Core Information Security Concepts](#1-core-information-security-concepts)
2.  [Defensive Security & Blue Teaming](#2-defensive-security--blue-teaming)
3.  [Offensive Security & Red Teaming](#3-offensive-security--red-teaming)
4.  [Digital Forensics & Incident Response (DFIR)](#4-digital-forensics--incident-response-dfir)
5.  [Tool Deep Dive: Network & Asset Management](#5-tool-deep-dive-network--asset-management)
6.  [Tool Deep Dive: Vulnerability & Exploitation](#6-tool-deep-dive-vulnerability--exploitation)
7.  [Tool Deep Dive: Monitoring & Forensics](#7-tool-deep-dive-monitoring--forensics)
8.  [Tool Deep Dive: Web & Password Attacks](#8-tool-deep-dive-web--password-attacks)

---

## 1. Core Information Security Concepts

### **The CIA Triad**
The foundational model for information security.
*   **Confidentiality:** Ensuring that data is accessible only to authorized individuals. *Controls: Encryption, Access Control Lists (ACLs), Data Classification.*
*   **Integrity:** Maintaining the consistency, accuracy, and trustworthiness of data over its entire lifecycle. Data must not be changed in transit or altered by unauthorized people. *Controls: Hashing (e.g., SHA-256), Digital Signatures, File Integrity Monitoring (FIM).*
*   **Availability:** Ensuring that data and services are accessible to authorized users when they are needed. *Controls: Redundancy (RAID, server clusters), Backups, Disaster Recovery Plans.*

### **Threat vs. Vulnerability vs. Risk**
*   **Threat:** A potential for harm. It's the "what" could happen (e.g., a malicious actor, a natural disaster).
*   **Vulnerability:** A weakness or flaw in a system, process, or control that could be exploited by a threat. It's the "how" an attack could succeed (e.g., an unpatched server, a weak password).
*   **Risk:** The potential for loss or damage when a threat exploits a vulnerability. It's the intersection of the two. `Risk = Threat x Vulnerability`.

### **The Cyber Kill Chain®**
A 7-stage model of a cyberattack, useful for understanding the adversary's process.
1.  **Reconnaissance:** Gathering information on the target.
2.  **Weaponization:** Creating a malicious payload.
3.  **Delivery:** Transmitting the payload (e.g., phishing email).
4.  **Exploitation:** Triggering the payload to exploit a vulnerability.
5.  **Installation:** Establishing persistence on the victim's system.
6.  **Command & Control (C2):** Creating a channel back to the attacker.
7.  **Actions on Objectives:** The attacker achieves their goal (e.g., data theft).

### **MITRE ATT&CK® Framework**
A globally accessible knowledge base of adversary tactics and techniques. It provides a granular view of attacker behavior, organized into:
*   **Tactics:** The adversary's technical goal (e.g., `Privilege Escalation`, `Lateral Movement`).
*   **Techniques:** How the tactic is achieved (e.g., `Privilege Escalation` via `Sudo Caching`).
*   **Sub-techniques:** A more specific description of the technique.

---

## 2. Defensive Security & Blue Teaming

### **Defense in Depth**
A strategy of using multiple, layered security controls. If one layer fails, another is in place to stop an attack. Layers include:
*   **Perimeter:** Firewalls, email filtering.
*   **Network:** Intrusion Detection/Prevention Systems (IDS/IPS), network segmentation.
*   **Endpoint:** Antivirus, Endpoint Detection & Response (EDR), HIDS (e.g., **Wazuh**).
*   **Application:** Secure coding, Web Application Firewalls (WAF).
*   **Data:** Encryption, access controls.
*   **Human:** Security awareness training.

### **Firewalls**
*   **Stateless:** Filters traffic based on source/destination IP and port. Fast but basic.
*   **Stateful:** Tracks the state of active connections, offering better security than stateless firewalls.
*   **Next-Gen Firewall (NGFW):** Combines traditional firewalling with advanced features like deep packet inspection, application awareness, and integrated IDS/IPS.
*   **Web Application Firewall (WAF):** Specifically designed to protect web applications by filtering and monitoring HTTP traffic between the application and the internet.

### **IDS/IPS**
*   **Intrusion Detection System (IDS):** Monitors network traffic or system activity for malicious signatures or anomalies and logs alerts. It is a passive system.
*   **Intrusion Prevention System (IPS):** An active system that can block or prevent detected intrusions in real-time.
*   **Common Tools:** Snort, Suricata, Zeek (formerly Bro).

---

## 3. Offensive Security & Red Teaming

### **Phases of a Penetration Test**
1.  **Scoping:** Defining the rules of engagement, objectives, and boundaries of the test.
2.  **Reconnaissance (OSINT):** Gathering public information about the target.
3.  **Scanning:** Using tools like **Nmap** to discover live hosts, open ports, and services.
4.  **Gaining Access (Exploitation):** Using tools like **Metasploit** to exploit a vulnerability and get a foothold.
5.  **Maintaining Access:** Establishing persistence with backdoors or C2 channels.
6.  **Covering Tracks:** Removing evidence of the intrusion.
7.  **Reporting:** Documenting findings, risk ratings, and remediation advice.

### **Post-Exploitation**
The phase after gaining initial access.
*   **Privilege Escalation:** Elevating permissions from a low-level user to root or Administrator.
*   **Lateral Movement:** Moving from the initially compromised machine to other machines on the same network.
*   **Pivoting:** Using a compromised host to attack other systems that are not directly accessible.

### **Command & Control (C2) Frameworks**
Toolkits used by attackers to manage compromised machines.
*   **Cobalt Strike:** A popular commercial framework, widely used by both red teams and real-world adversaries.
*   **Sliver / Covenant:** Open-source alternatives that provide similar C2 capabilities.

---

## 4. Digital Forensics & Incident Response (DFIR)

### **Order of Volatility**
The principle of collecting evidence in order from most volatile to least volatile.
1.  CPU Registers, Cache
2.  Routing Tables, ARP Cache, Process Table
3.  RAM / System Memory (**Volatility**'s domain)
4.  Temporary System Files
5.  Disk / Storage Media
6.  Backups, Archives

### **Chain of Custody**
A chronological documentation trail showing the seizure, custody, control, transfer, analysis, and disposition of evidence. It is critical for maintaining the integrity of evidence for legal proceedings.

### **Disk Imaging**
Creating a bit-for-bit copy of a storage device.
*   **Why:** To perform analysis on a copy without altering the original evidence.
*   **Tools:** `dd` (Linux), FTK Imager, EnCase.
*   **Write Blockers:** Hardware or software tools that prevent any writes to the source drive during the imaging process.

---

## 5. Tool Deep Dive: Network & Asset Management

### **RunZero**
*   **Primary Use:** Comprehensive asset inventory and network mapping.
*   **Key Features:** Unauthenticated scanning, service identification, network topology mapping.
*   **Synergy:** Provides the "what to scan" for **Nmap** and **Nuclei**.

### **Nmap**
*   **Primary Use:** The "gold standard" for network discovery and port scanning.
*   **Advanced Commands:**
    *   **OS Detection:** `nmap -O <target>`
    *   **Aggressive Scan (OS, Version, Scripts, Traceroute):** `nmap -A <target>`
    *   **Scan All TCP Ports:** `nmap -p- <target>`
    *   **UDP Scan:** `nmap -sU <target>` (Note: Very slow).
    *   **Using NSE Scripts:** `nmap --script=vuln <target>` - Runs all scripts in the "vuln" category to check for known vulnerabilities.
    *   **Output Formats:** `nmap -oA <basename> <target>` - Outputs in all formats (Normal, XML, and Grepable).

### **Wireshark**
*   **Primary Use:** Deep packet inspection and network protocol analysis.
*   **Advanced Usage:**
    *   **Capture vs. Display Filters:** Capture filters (e.g., `tcp port 80`) are set before starting a capture to reduce the size of the capture file. Display filters are applied after the capture to analyze the data.
    *   **tshark (CLI Wireshark):** `tshark -i eth0 -Y "http.request" -T fields -e http.host -e http.request.uri` - Captures on `eth0` and prints the Host and URI of any HTTP requests.
    *   **Expert Information:** Wireshark can analyze a capture and point out potential issues like retransmissions, duplicate ACKs, and window size problems.

---

## 6. Tool Deep Dive: Vulnerability & Exploitation

### **Nuclei**
*   **Primary Use:** Fast, template-based vulnerability scanning.
*   **Advanced Usage:**
    *   **Updating Templates:** `nuclei -update-templates`
    *   **Targeting Specific Templates:** `nuclei -l <list_of_urls.txt> -t technologies/wordpress/` - Runs only WordPress-related templates against a list of URLs.
    *   **Template Structure:** Templates consist of an `info` block (metadata) and protocol blocks (`http`, `dns`, etc.) that define `requests`, `matchers`, and `extractors`.

### **Metasploit Framework**
*   **Primary Use:** A comprehensive framework for penetration testing.
*   **Module Types:**
    *   `exploit`: Modules that execute a payload on a target.
    *   `auxiliary`: Modules for scanning, fuzzing, and enumeration.
    *   `post`: Modules that run on a compromised target after exploitation.
    *   `payload`: The code that runs on the target after a successful exploit (e.g., a reverse shell).
    *   `encoder`: Obfuscates payloads to avoid detection by AV/IDS.
*   **Payloads:**
    *   **Reverse Shell:** The compromised target connects *back* to the attacker's machine. More firewall-friendly.
    *   **Bind Shell:** The exploit opens a listener on the target machine, and the attacker connects *to* it.

---

## 7. Tool Deep Dive: Monitoring & Forensics

### **Wazuh**
*   **Primary Use:** A SIEM (Security Information and Event Management) and HIDS (Host-based Intrusion Detection System).
*   **Key Features:** Log analysis, file integrity monitoring (FIM), vulnerability detection, and active response.
*   **Synergy:** Acts as the central hub for alerts from nearly every other tool in this list.

### **Volatility Framework**
*   **Primary Use:** Gold standard for memory forensics.
*   **Key Plugins:**
    *   `imageinfo`: Suggests the OS profile.
    *   `pslist` / `pstree`: Lists running processes and shows parent/child relationships.
    *   `netscan`: Shows network connections active at the time of capture.
    *   `filescan`: Scans for file objects in memory.
    *   `cmdline`: Shows the command line arguments for processes.
    *   `malfind`: A powerful plugin to find injected code and other common malware artifacts.

### **Ghidra**
*   **Primary Use:** A full-featured software reverse engineering (SRE) suite.
*   **Key Windows/Features:**
    *   **Code Browser:** The main window for analysis.
    *   **Listing (Disassembly):** Shows the assembly code.
    *   **Decompiler:** Translates assembly to C-like pseudo-code. This is its most powerful feature.
    *   **Symbol Tree:** Shows all identified functions, labels, and namespaces.
    *   **Function Graph:** Visualizes the control flow of a single function, making it easy to understand loops and conditional branches.

---

## 8. Tool Deep Dive: Web & Password Attacks

### **OWASP ZAP & Burp Suite**
*   **Primary Use:** Web application security testing.
*   **Advanced Features (Burp):**
    *   **Decoder:** A utility for transforming data (e.g., URL decoding, Base64 encoding/decoding).
    *   **Comparer:** A visual diff utility to compare two pieces of data (e.g., two responses from a server).
    *   **Extensions:** The BApp Store allows for the installation of community-created extensions that add significant functionality.

### **Password Cracking: John the Ripper & Hashcat**
*   **Use Case:** Recovering passwords from captured hashes.
*   **Attack Modes:**
    *   **Wordlist Attack:** Uses a list of potential passwords (e.g., `rockyou.txt`).
    *   **Mask Attack (Hashcat):** Tries combinations based on a defined pattern (e.g., `?u?l?l?l?l?d?d?d` for an uppercase letter, 4 lowercase, and 3 digits).
    *   **Brute-Force:** Tries every possible combination of characters.
*   **Hash Identification:** Before cracking, you must know the hash type. Use tools like `hashid` or online identifiers.

### **Privilege Escalation Enumeration**
*   **Use Case:** Finding pathways to elevate privileges on a compromised system.
*   **Tools:**
    *   **LinPEAS (Linux):** A script that automates the discovery of misconfigurations.
    *   **WinPEAS (Windows):** The Windows equivalent.
    *   **GTFOBins:** A curated list of Unix binaries that can be used to bypass local security restrictions.

### **Open-Source Intelligence (OSINT)**
*   **Use Case:** Gathering information from public sources.
*   **Tools:**
    *   **theHarvester:** Gathers emails, subdomains, hosts, and names.
    *   **Maltego:** Visualizes relationships between data points.
    *   **Shodan:** A search engine for internet-connected devices. Can find exposed servers, webcams, and industrial control systems.
