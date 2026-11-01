import os
import stat
import sys
import time
import socket
import struct
import ctypes
import logging
import threading
from collections import deque

logger = logging.getLogger(__name__)

# Bound pending honeypot detections separately from the bounded dashboard histories.
HONEYPOT_HITS_MAX_ENTRIES = 1000

HAS_BCC = False
try:
    from bcc import BPF
    HAS_BCC = True
except ImportError:
    logger.warning("BCC/BPF compiler collection python bindings not found. eBPF telemetry is disabled (falling back to psutil).")

# BPF C code
bpf_text = """
#include <uapi/linux/ptrace.h>
#include <linux/sched.h>
#include <linux/fs.h>
// struct sock (tcp_v4_connect / tcp_retransmit_skb probes). Without it newer
// kernels leave the type incomplete and the WHOLE program fails to compile,
// which silently disabled every probe, honeypots included.
#include <net/sock.h>

struct exec_event_t {
    u32 pid;
    u32 uid;
    char comm[16];
    char filename[256];
};
BPF_PERF_OUTPUT(exec_events);

struct tcp_connect_event_t {
    u32 pid;
    char comm[16];
    u32 daddr;
    u16 dport;
};
BPF_PERF_OUTPUT(tcp_connect_events);

struct tcp_retransmit_event_t {
    u32 saddr;
    u32 daddr;
    u16 sport;
    u16 dport;
    u8 state;
};
BPF_PERF_OUTPUT(tcp_retransmit_events);

struct privilege_event_t {
    u32 pid;
    char comm[16];
    u32 uid;
};
BPF_PERF_OUTPUT(privilege_events);

struct open_event_t {
    u32 pid;
    char comm[16];
    char filename[256];
    u32 flags;
};
BPF_PERF_OUTPUT(openat2_events);
BPF_PERF_OUTPUT(honeypot_events);

struct delete_event_t {
    u32 pid;
    char comm[16];
    char filename[256];
};
BPF_PERF_OUTPUT(delete_events);

struct honeypot_key_t {
    char path[256];
};
BPF_HASH(honeypot_paths, struct honeypot_key_t, u32);

// Path matching above sees only the exact string handed to open(): "cat flag.txt"
// after a cd, "./flag.txt" or "grep -r" never equal an absolute decoy path.
// Matching the opened file's (device, inode) at the LSM open hook catches every
// route to it. The map value is the decoy's index; Python maps it back to a path.
struct honeypot_inode_t {
    u64 dev;
    u64 ino;
};
BPF_HASH(honeypot_inodes, struct honeypot_inode_t, u32);

int kprobe__security_file_open(struct pt_regs *ctx, struct file *file) {
    struct honeypot_inode_t key = {};
    key.ino = file->f_inode->i_ino;
    key.dev = file->f_inode->i_sb->s_dev;
    u32 *idx = honeypot_inodes.lookup(&key);
    if (!idx) {
        return 0;
    }
    struct open_event_t data = {};
    data.pid = bpf_get_current_pid_tgid() >> 32;
    bpf_get_current_comm(&data.comm, sizeof(data.comm));
    data.flags = *idx | 0x80000000;
    honeypot_events.perf_submit(ctx, &data, sizeof(data));
    return 0;
}

TRACEPOINT_PROBE(syscalls, sys_enter_execve) {
    struct exec_event_t data = {};
    data.pid = bpf_get_current_pid_tgid() >> 32;
    data.uid = bpf_get_current_uid_gid();
    bpf_get_current_comm(&data.comm, sizeof(data.comm));
    bpf_probe_read_user_str(&data.filename, sizeof(data.filename), args->filename);
    exec_events.perf_submit(args, &data, sizeof(data));
    return 0;
}

int kprobe__tcp_v4_connect(struct pt_regs *ctx, struct sock *sk) {
    struct tcp_connect_event_t data = {};
    data.pid = bpf_get_current_pid_tgid() >> 32;
    bpf_get_current_comm(&data.comm, sizeof(data.comm));
    
    u32 daddr = 0;
    u16 dport = 0;
    bpf_probe_read_kernel(&daddr, sizeof(daddr), &sk->__sk_common.skc_daddr);
    bpf_probe_read_kernel(&dport, sizeof(dport), &sk->__sk_common.skc_dport);
    
    data.daddr = daddr;
    data.dport = bpf_ntohs(dport);
    
    tcp_connect_events.perf_submit(ctx, &data, sizeof(data));
    return 0;
}

int kprobe__tcp_retransmit_skb(struct pt_regs *ctx, struct sock *sk, struct sk_buff *skb) {
    struct tcp_retransmit_event_t data = {};
    u32 saddr = 0, daddr = 0;
    u16 sport = 0, dport = 0;
    u8 state = 0;
    
    bpf_probe_read_kernel(&saddr, sizeof(saddr), &sk->__sk_common.skc_rcv_saddr);
    bpf_probe_read_kernel(&daddr, sizeof(daddr), &sk->__sk_common.skc_daddr);
    bpf_probe_read_kernel(&sport, sizeof(sport), &sk->__sk_common.skc_num);
    bpf_probe_read_kernel(&dport, sizeof(dport), &sk->__sk_common.skc_dport);
    bpf_probe_read_kernel(&state, sizeof(state), &sk->sk_state);
    
    data.saddr = saddr;
    data.daddr = daddr;
    data.sport = sport;
    data.dport = bpf_ntohs(dport);
    data.state = state;
    
    tcp_retransmit_events.perf_submit(ctx, &data, sizeof(data));
    return 0;
}

TRACEPOINT_PROBE(syscalls, sys_enter_setuid) {
    struct privilege_event_t data = {};
    data.pid = bpf_get_current_pid_tgid() >> 32;
    bpf_get_current_comm(&data.comm, sizeof(data.comm));
    data.uid = args->uid;
    privilege_events.perf_submit(args, &data, sizeof(data));
    return 0;
}

TRACEPOINT_PROBE(syscalls, sys_enter_setreuid) {
    struct privilege_event_t data = {};
    data.pid = bpf_get_current_pid_tgid() >> 32;
    bpf_get_current_comm(&data.comm, sizeof(data.comm));
    data.uid = args->ruid;
    privilege_events.perf_submit(args, &data, sizeof(data));
    return 0;
}

TRACEPOINT_PROBE(syscalls, sys_enter_openat) {
    struct open_event_t data = {};
    data.pid = bpf_get_current_pid_tgid() >> 32;
    bpf_get_current_comm(&data.comm, sizeof(data.comm));
    bpf_probe_read_user_str(&data.filename, sizeof(data.filename), args->filename);
    data.flags = args->flags;

    // Look up with data.filename itself (same 256-byte layout as the key).
    // Copying it into a second 256-byte key put ~540 bytes on the 512-byte BPF
    // stack, and the verifier rejected the whole program.
    u32 *val = honeypot_paths.lookup((struct honeypot_key_t *)data.filename);
    if (val) {
        honeypot_events.perf_submit(args, &data, sizeof(data));
    }
    
    // Do not stream every openat call through Python: that event is very high
    // volume. The dashboard specifically needs openat2, while honeypot matches
    // from openat continue to use the dedicated honeypot_events buffer.
    return 0;
}

TRACEPOINT_PROBE(syscalls, sys_enter_openat2) {
    struct open_event_t data = {};
    data.pid = bpf_get_current_pid_tgid() >> 32;
    bpf_get_current_comm(&data.comm, sizeof(data.comm));
    bpf_probe_read_user_str(&data.filename, sizeof(data.filename), args->filename);
    data.flags = 0;  // openat2 flags are in a struct, not a flat int — set 0 as safe default

    // Look up with data.filename itself (same 256-byte layout as the key).
    // Copying it into a second 256-byte key put ~540 bytes on the 512-byte BPF
    // stack, and the verifier rejected the whole program.
    u32 *val = honeypot_paths.lookup((struct honeypot_key_t *)data.filename);
    if (val) {
        honeypot_events.perf_submit(args, &data, sizeof(data));
    }

    openat2_events.perf_submit(args, &data, sizeof(data));
    return 0;
}

TRACEPOINT_PROBE(syscalls, sys_enter_unlinkat) {
    struct delete_event_t data = {};
    data.pid = bpf_get_current_pid_tgid() >> 32;
    bpf_get_current_comm(&data.comm, sizeof(data.comm));
    bpf_probe_read_user_str(&data.filename, sizeof(data.filename), args->pathname);
    delete_events.perf_submit(args, &data, sizeof(data));
    return 0;
}
"""

class EBPFTelemetryEngine:
    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def __init__(self):
        self.bpf = None
        self.polling_thread = None
        self.stop_event = threading.Event()
        
        self.exec_deque = deque(maxlen=500)
        self.open_deque = deque(maxlen=500)
        self.delete_deque = deque(maxlen=500)
        self.conn_deque = deque(maxlen=500)
        self.retrans_deque = deque(maxlen=500)
        self.priv_deque = deque(maxlen=500)
        self.honeypot_hits = deque(maxlen=HONEYPOT_HITS_MAX_ENTRIES)
        # Index -> decoy path for inode-matched hits (map value in honeypot_inodes).
        self.honeypot_inode_paths = []
        # (pid, path) -> last hit time; one open can match both probes.
        self._recent_honeypot_hits = {}
        self.honeypot_dropped_events = 0
        self.honeypot_lock = threading.Lock()
        
        self.lock = threading.Lock()
        
    def start(self):
        if not HAS_BCC:
            logger.warning("BCC not installed, cannot start eBPF engine.")
            return False
            
        try:
            self.bpf = BPF(text=bpf_text)
            
            # Register perf buffer callbacks
            self.bpf["exec_events"].open_perf_buffer(self._handle_exec_event)
            self.bpf["openat2_events"].open_perf_buffer(self._handle_openat2_event)
            self.bpf["delete_events"].open_perf_buffer(self._handle_delete_event)
            self.bpf["tcp_connect_events"].open_perf_buffer(self._handle_tcp_connect)
            self.bpf["tcp_retransmit_events"].open_perf_buffer(self._handle_tcp_retransmit)
            self.bpf["privilege_events"].open_perf_buffer(self._handle_privilege_event)
            self.bpf["honeypot_events"].open_perf_buffer(self._handle_honeypot_event)
            
            self.polling_thread = threading.Thread(target=self._poll_perf_buffers, daemon=True, name="ebpf-perf-poll")
            self.polling_thread.start()
            logger.info("eBPF Telemetry engine loaded and polling perf buffers.")
            return True
        except Exception as e:
            logger.error(f"Failed to load or compile eBPF probes: {e}")
            self.bpf = None
            return False

    def stop(self):
        self.stop_event.set()
        
    def register_honeypot_paths(self, paths: list):
        if not self.bpf or not HAS_BCC:
            return
        tbl = self.bpf["honeypot_paths"]
        for p in paths:
            # BCC map keys must be the table's ctypes Key type, not raw bytes
            # (raw bytes raised "byref() argument 1 must be _ctypes._CData").
            try:
                key = tbl.Key()
                key.path = p.encode("utf-8")[:255]
                tbl[key] = ctypes.c_uint(1)
            except Exception as e:
                logger.error(f"Error registering honeypot path {p} in BPF map: {e}")

    @staticmethod
    def _kernel_dev(st_dev: int) -> int:
        """Userspace st_dev -> the kernel's internal dev_t (MKDEV: major<<20 | minor)."""
        return (os.major(st_dev) << 20) | os.minor(st_dev)

    def register_honeypot_inodes(self, paths: list) -> int:
        """Watch decoys by (device, inode) so any path form that opens them is caught.

        Uses lstat: a symlink planted at a decoy path is never followed or watched.
        Returns how many paths were registered.
        """
        if not self.bpf or not HAS_BCC:
            return 0
        tbl = self.bpf["honeypot_inodes"]
        registered = 0
        for p in paths:
            try:
                st = os.lstat(p)
            except OSError as e:
                logger.warning(f"Honeypot inode watch skipped for {p}: {e}")
                continue
            if not stat.S_ISREG(st.st_mode):
                logger.warning(f"Honeypot inode watch skipped for non-regular file {p}")
                continue
            idx = len(self.honeypot_inode_paths)
            self.honeypot_inode_paths.append(p)
            key = tbl.Key(self._kernel_dev(st.st_dev), st.st_ino)
            try:
                tbl[key] = ctypes.c_uint(idx)
                registered += 1
            except Exception as e:
                logger.error(f"Error registering honeypot inode for {p}: {e}")
        return registered

    def _poll_perf_buffers(self):
        while not self.stop_event.is_set():
            try:
                self.bpf.perf_buffer_poll(timeout=100)
            except Exception as e:
                logger.error(f"Error polling BPF perf buffers: {e}")
                time.sleep(0.5)

    # --- Perf Buffer Event Callbacks ---

    def _handle_exec_event(self, cpu, data, size):
        event = self.bpf["exec_events"].event(data)
        filename = event.filename.decode("utf-8", "ignore")
        comm = event.comm.decode("utf-8", "ignore")
        from security.telemetry_sanitizer import sanitize_telemetry
        raw = {"pid": str(event.pid), "comm": comm, "path": filename}
        clean = sanitize_telemetry(raw)
        clean_pid = int(clean.get("pid")) if clean.get("pid") else event.pid
        clean_comm = clean.get("comm", "")
        clean_filename = clean.get("path", "")
        with self.lock:
            self.exec_deque.append({
                "pid": clean_pid,
                "uid": event.uid,
                "comm": clean_comm,
                "filename": clean_filename,
                "timestamp": time.time()
            })

    def _handle_openat2_event(self, cpu, data, size):
        event = self.bpf["openat2_events"].event(data)
        filename = event.filename.decode("utf-8", "ignore")
        comm = event.comm.decode("utf-8", "ignore")
        from security.telemetry_sanitizer import sanitize_telemetry
        clean = sanitize_telemetry({
            "pid": str(event.pid),
            "comm": comm,
            "path": filename,
        })
        with self.lock:
            self.open_deque.append({
                "syscall": "sys_enter_openat2",
                "pid": int(clean.get("pid") or event.pid),
                "comm": clean.get("comm", ""),
                "filename": clean.get("path", ""),
                "timestamp": time.time(),
            })

    def _handle_delete_event(self, cpu, data, size):
        """Record an unlinkat syscall attempt; sys_enter does not prove deletion succeeded."""
        event = self.bpf["delete_events"].event(data)
        filename = event.filename.decode("utf-8", "ignore")
        comm = event.comm.decode("utf-8", "ignore")
        from security.telemetry_sanitizer import sanitize_telemetry
        clean = sanitize_telemetry({
            "pid": str(event.pid),
            "comm": comm,
            "path": filename,
        })
        record = {
            "syscall": "sys_enter_unlinkat",
            "pid": int(clean.get("pid") or event.pid),
            "comm": clean.get("comm", ""),
            "filename": clean.get("path", ""),
            "disposition": "observed",
            "timestamp": time.time(),
        }
        with self.lock:
            self.delete_deque.append(record)

    def _handle_tcp_connect(self, cpu, data, size):
        event = self.bpf["tcp_connect_events"].event(data)
        daddr_str = socket.inet_ntoa(struct.pack("<I", event.daddr))
        comm = event.comm.decode("utf-8", "ignore")
        from security.telemetry_sanitizer import sanitize_telemetry
        raw = {"pid": str(event.pid), "comm": comm, "ip": daddr_str, "port": str(event.dport)}
        clean = sanitize_telemetry(raw)
        clean_pid = int(clean.get("pid")) if clean.get("pid") else event.pid
        clean_comm = clean.get("comm", "")
        clean_daddr = clean.get("ip", "")
        clean_dport = int(clean.get("port")) if clean.get("port") else event.dport
        with self.lock:
            self.conn_deque.append({
                "pid": clean_pid,
                "comm": clean_comm,
                "daddr": clean_daddr,
                "dport": clean_dport,
                "timestamp": time.time()
            })

    def _handle_tcp_retransmit(self, cpu, data, size):
        event = self.bpf["tcp_retransmit_events"].event(data)
        saddr_str = socket.inet_ntoa(struct.pack("<I", event.saddr))
        daddr_str = socket.inet_ntoa(struct.pack("<I", event.daddr))
        from security.telemetry_sanitizer import sanitize_telemetry
        raw = {
            "ip": saddr_str, "port": str(event.sport),
        }
        clean_src = sanitize_telemetry(raw)
        raw2 = {"ip": daddr_str, "port": str(event.dport)}
        clean_dst = sanitize_telemetry(raw2)
        with self.lock:
            self.retrans_deque.append({
                "saddr": clean_src.get("ip", ""),
                "daddr": clean_dst.get("ip", ""),
                "sport": int(clean_src.get("port", 0) or 0),
                "dport": int(clean_dst.get("port", 0) or 0),
                "state": event.state,
                "timestamp": time.time()
            })

    def _handle_privilege_event(self, cpu, data, size):
        event = self.bpf["privilege_events"].event(data)
        comm = event.comm.decode("utf-8", "ignore")
        from security.telemetry_sanitizer import sanitize_telemetry
        raw_pid = {"pid": str(event.pid)}
        raw_comm = {"comm": comm}
        raw_uid = {"uid": str(event.uid)}
        clean_pid = sanitize_telemetry(raw_pid).get("pid", "")
        clean_comm = sanitize_telemetry(raw_comm).get("comm", "")
        clean_uid = sanitize_telemetry(raw_uid).get("uid", "")
        clean_pid_int = int(clean_pid) if clean_pid else event.pid
        clean_uid_int = int(clean_uid) if clean_uid else event.uid
        with self.lock:
            self.priv_deque.append({
                "pid": clean_pid_int,
                "comm": clean_comm,
                "uid": clean_uid_int,
                "timestamp": time.time()
            })
        try:
            from security.db import log_security_event
            log_security_event(
                event_type="privilege_escalation",
                source="ebpf_telemetry",
                actor=f"{clean_comm} (PID {clean_pid_int})",
                payload_hash=f"uid={clean_uid_int}",
                disposition="observed",
                session_id="system_protection"
            )
        except Exception:
            pass

    def _handle_honeypot_event(self, cpu, data, size):
        event = self.bpf["honeypot_events"].event(data)
        filename = event.filename.decode("utf-8", "ignore")
        flags = getattr(event, "flags", 0) or 0
        if flags & 0x80000000:
            # Inode match from security_file_open: translate index -> decoy path.
            idx = flags & 0x7FFFFFFF
            if idx < len(self.honeypot_inode_paths):
                filename = self.honeypot_inode_paths[idx]
        comm = event.comm.decode("utf-8", "ignore")
        from security.telemetry_sanitizer import sanitize_telemetry
        raw = {"pid": str(event.pid), "comm": comm, "path": filename}
        clean = sanitize_telemetry(raw)
        clean_pid = int(clean.get("pid")) if clean.get("pid") else event.pid
        clean_comm = clean.get("comm", "")
        clean_filename = clean.get("path", "")
        with self.honeypot_lock:
            # An absolute-path open matches both the path probe and the inode
            # probe; report it once.
            now = time.time()
            key = (clean_pid, clean_filename)
            last = self._recent_honeypot_hits.get(key)
            if last is not None and now - last < 2.0:
                return
            self._recent_honeypot_hits[key] = now
            if len(self._recent_honeypot_hits) > HONEYPOT_HITS_MAX_ENTRIES:
                cutoff = now - 2.0
                self._recent_honeypot_hits = {
                    k: t for k, t in self._recent_honeypot_hits.items() if t >= cutoff
                }
            # Keep the newest bounded window. Record overflow so an event burst
            # cannot silently convert into unbounded root-process memory use.
            if len(self.honeypot_hits) == HONEYPOT_HITS_MAX_ENTRIES:
                self.honeypot_dropped_events += 1
            self.honeypot_hits.append({
                "pid": clean_pid,
                "comm": clean_comm,
                "filename": clean_filename,
                "timestamp": time.time()
            })

    # --- Data Retrieval API ---

    def get_recent_connections(self, n=20) -> list:
        with self.lock:
            return list(self.conn_deque)[-n:]

    def get_recent_execs(self, n=20) -> list:
        with self.lock:
            return list(self.exec_deque)[-n:]

    def get_recent_opens(self, n=20) -> list:
        with self.lock:
            return list(self.open_deque)[-n:]

    def get_recent_deletes(self, n=20) -> list:
        with self.lock:
            return list(self.delete_deque)[-n:]

    def get_privilege_escalations(self, n=10) -> list:
        with self.lock:
            return list(self.priv_deque)[-n:]

    def get_honeypot_hits(self) -> list:
        with self.honeypot_lock:
            hits = list(self.honeypot_hits)
            dropped = self.honeypot_dropped_events
            self.honeypot_hits.clear()
            self.honeypot_dropped_events = 0
        if dropped:
            logger.warning(
                "Honeypot telemetry queue overflowed; %d older hit(s) were dropped before consumption.",
                dropped,
            )
        return hits
