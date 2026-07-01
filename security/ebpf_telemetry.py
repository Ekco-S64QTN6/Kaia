import os
import sys
import time
import socket
import struct
import logging
import threading
from collections import deque

logger = logging.getLogger(__name__)

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
BPF_PERF_OUTPUT(open_events);
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
    
    struct honeypot_key_t key = {};
    __builtin_memcpy(key.path, data.filename, sizeof(key.path));
    u32 *val = honeypot_paths.lookup(&key);
    if (val) {
        honeypot_events.perf_submit(args, &data, sizeof(data));
    }
    
    open_events.perf_submit(args, &data, sizeof(data));
    return 0;
}

TRACEPOINT_PROBE(syscalls, sys_enter_openat2) {
    struct open_event_t data = {};
    data.pid = bpf_get_current_pid_tgid() >> 32;
    bpf_get_current_comm(&data.comm, sizeof(data.comm));
    bpf_probe_read_user_str(&data.filename, sizeof(data.filename), args->filename);
    data.flags = 0;  // openat2 flags are in a struct, not a flat int — set 0 as safe default

    struct honeypot_key_t key = {};
    __builtin_memcpy(key.path, data.filename, sizeof(key.path));
    u32 *val = honeypot_paths.lookup(&key);
    if (val) {
        honeypot_events.perf_submit(args, &data, sizeof(data));
    }

    open_events.perf_submit(args, &data, sizeof(data));
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
        self.conn_deque = deque(maxlen=500)
        self.retrans_deque = deque(maxlen=500)
        self.priv_deque = deque(maxlen=500)
        self.honeypot_hits = []
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
            # Map path to key
            p_bytes = p.encode("utf-8")[:255] + b"\x00"
            # In BCC, we can set key directly
            try:
                # Pad to 256 bytes
                padded = p_bytes.ljust(256, b"\x00")
                tbl[padded] = ctypes.c_uint(1)
            except Exception as e:
                logger.error(f"Error registering honeypot path {p} in BPF map: {e}")

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
                disposition="approved",
                session_id="system_protection"
            )
        except Exception:
            pass

    def _handle_honeypot_event(self, cpu, data, size):
        event = self.bpf["honeypot_events"].event(data)
        filename = event.filename.decode("utf-8", "ignore")
        comm = event.comm.decode("utf-8", "ignore")
        from security.telemetry_sanitizer import sanitize_telemetry
        raw = {"pid": str(event.pid), "comm": comm, "path": filename}
        clean = sanitize_telemetry(raw)
        clean_pid = int(clean.get("pid")) if clean.get("pid") else event.pid
        clean_comm = clean.get("comm", "")
        clean_filename = clean.get("path", "")
        with self.honeypot_lock:
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

    def get_privilege_escalations(self, n=10) -> list:
        with self.lock:
            return list(self.priv_deque)[-n:]

    def get_honeypot_hits(self) -> list:
        with self.honeypot_lock:
            hits = list(self.honeypot_hits)
            self.honeypot_hits.clear()
            return hits
