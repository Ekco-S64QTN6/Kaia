import os
import re
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

# Order matches the count_drop() indexes in bpf_text.
RINGBUF_STREAMS = (
    "exec_events", "tcp_connect_events", "tcp_retransmit_events", "privilege_events",
    "openat2_events", "honeypot_events", "delete_events",
)
RINGBUF_DROP_REPORT_INTERVAL = 30.0

# privilege_event_t.kind values (PRIV_KIND_* in bpf_text).
PRIV_KIND_SETUID_EXEC = 1
PRIV_KIND_NO_PRIOR_ROOT = 2

# Security event types for a gain of root seen at commit_creds().
PRIV_EVENT_SETUID_EXEC = "privilege_setuid_exec"
PRIV_EVENT_SETUID_EXEC_UNUSUAL = "privilege_setuid_exec_unusual"
PRIV_EVENT_UNEXPECTED = "privilege_escalation_unexpected"
PRIV_ALERT_EVENTS = (PRIV_EVENT_UNEXPECTED, PRIV_EVENT_SETUID_EXEC_UNUSUAL)

# Where Arch installs its set-user-ID programs (sudo, su, passwd, pkexec,
# fusermount3, the polkit and dbus helpers). bprm->filename is the path as
# passed to execve, so a relative path is never "usual".
SYSTEM_SETUID_PREFIXES = ("/usr/bin/", "/usr/lib/", "/usr/sbin/", "/usr/libexec/")
_CLEAN_EXEC_PATH = re.compile(r"[A-Za-z0-9_./+-]{1,200}")


def classify_privilege_gain(kind: int, exec_path: str) -> str:
    """Event type for a root gain reported by the commit_creds probe.

    A set-user-ID exec is how sudo, su and pkexec work; it is recorded, and only
    flagged when the binary is outside the system directories. A gain outside
    execve by a task that held root in none of uid/euid/suid cannot come from
    the setuid family without CAP_SETUID: it is the shape of a kernel privilege
    escalation (prepare_kernel_cred/commit_creds), and of nothing routine.
    Unknown kinds are treated as unexpected, never as routine.
    """
    if kind == PRIV_KIND_SETUID_EXEC:
        # Judge the raw path: sanitizing first could strip characters out of an
        # unusual path until it looked like a system one.
        if (
            _CLEAN_EXEC_PATH.fullmatch(exec_path or "")
            and exec_path.startswith(SYSTEM_SETUID_PREFIXES)
            and "/." not in exec_path
        ):
            return PRIV_EVENT_SETUID_EXEC
        return PRIV_EVENT_SETUID_EXEC_UNUSUAL
    return PRIV_EVENT_UNEXPECTED

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
// struct cred / kuid_t and struct linux_binprm, for the commit_creds probe.
#include <linux/cred.h>
#include <linux/binfmts.h>

// Events travel through BPF ring buffers: one buffer per stream shared by all
// CPUs, charged to the memory cgroup rather than RLIMIT_MEMLOCK. The old
// per-CPU perf buffers (BCC default: 8 pages x 12 CPUs) overflowed in every
// exec/unlink burst (package builds, pip installs) and BCC only printed an
// anonymous "Possibly lost N samples". A full ring now increments a per-stream
// counter that Python reports by name. Index order = RINGBUF_STREAMS in Python.
BPF_PERCPU_ARRAY(ringbuf_drops, u64, 7);

static __always_inline void count_drop(u32 stream) {
    u64 *n = ringbuf_drops.lookup(&stream);
    if (n) {
        (*n)++;
    }
}

struct exec_event_t {
    u32 pid;
    u32 uid;
    char comm[16];
    char filename[256];
};
BPF_RINGBUF_OUTPUT(exec_events, 256);  // 1024 KiB, shared by all CPUs

struct tcp_connect_event_t {
    u32 pid;
    char comm[16];
    u32 daddr;
    u16 dport;
};
BPF_RINGBUF_OUTPUT(tcp_connect_events, 64);  // 256 KiB, shared by all CPUs

struct tcp_retransmit_event_t {
    u32 saddr;
    u32 daddr;
    u16 sport;
    u16 dport;
    u8 state;
};
BPF_RINGBUF_OUTPUT(tcp_retransmit_events, 16);  // 64 KiB, shared by all CPUs

// Privilege transitions are taken from commit_creds(), where every credential
// change is installed, not from setuid()/setreuid() syscall entry. Syscall
// entry saw only the requested uid: privilege *drops* (pacman -> alpm), calls
// that fail with EPERM, and processes that already were root, while missing
// setresuid() and every in-kernel change. Only a gain of root is reported:
// the old credential had no effective root, the new one has root uid or euid.
#define PRIV_KIND_SETUID_EXEC 1  // installed by execve of a set-user-ID binary
#define PRIV_KIND_NO_PRIOR_ROOT 2  // outside execve, and no root uid/euid/suid before
struct privilege_event_t {
    u32 pid;
    u32 kind;
    u32 old_uid;
    u32 old_euid;
    u32 old_suid;
    u32 new_uid;
    u32 new_euid;
    char comm[16];
    char filename[256];  // the executed binary, for PRIV_KIND_SETUID_EXEC
};
BPF_RINGBUF_OUTPUT(privilege_events, 16);  // 64 KiB, shared by all CPUs

// begin_new_exec() calls security_bprm_committing_creds(bprm) and then
// commit_creds(bprm->cred). The hook records which cred the exec is about to
// install; commit_creds() treats the change as an exec only when it installs
// exactly that cred. A leftover entry can never mark a later change as exec.
struct exec_cred_t {
    u64 bprm;
    u64 cred;
};
BPF_HASH(exec_creds, u32, struct exec_cred_t, 4096);

struct open_event_t {
    u32 pid;
    char comm[16];
    char filename[256];
    u32 flags;
};
BPF_RINGBUF_OUTPUT(openat2_events, 64);  // 256 KiB, shared by all CPUs
BPF_RINGBUF_OUTPUT(honeypot_events, 64);  // 256 KiB, shared by all CPUs

struct delete_event_t {
    u32 pid;
    char comm[16];
    char filename[256];
};
BPF_RINGBUF_OUTPUT(delete_events, 256);  // 1024 KiB, shared by all CPUs

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
    if (honeypot_events.ringbuf_output(&data, sizeof(data), 0) != 0) {
        count_drop(5);  // honeypot_events
    }
    return 0;
}

TRACEPOINT_PROBE(syscalls, sys_enter_execve) {
    struct exec_event_t data = {};
    data.pid = bpf_get_current_pid_tgid() >> 32;
    data.uid = bpf_get_current_uid_gid();
    bpf_get_current_comm(&data.comm, sizeof(data.comm));
    bpf_probe_read_user_str(&data.filename, sizeof(data.filename), args->filename);
    if (exec_events.ringbuf_output(&data, sizeof(data), 0) != 0) {
        count_drop(0);  // exec_events
    }
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
    
    if (tcp_connect_events.ringbuf_output(&data, sizeof(data), 0) != 0) {
        count_drop(1);  // tcp_connect_events
    }
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
    
    if (tcp_retransmit_events.ringbuf_output(&data, sizeof(data), 0) != 0) {
        count_drop(2);  // tcp_retransmit_events
    }
    return 0;
}

int kprobe__security_bprm_committing_creds(struct pt_regs *ctx, struct linux_binprm *bprm) {
    u32 tid = (u32)bpf_get_current_pid_tgid();
    struct exec_cred_t ec = {};
    struct cred *cred = NULL;
    bpf_probe_read_kernel(&cred, sizeof(cred), &bprm->cred);
    ec.bprm = (u64)bprm;
    ec.cred = (u64)cred;
    exec_creds.update(&tid, &ec);
    return 0;
}

int kprobe__commit_creds(struct pt_regs *ctx, struct cred *new) {
    u32 tid = (u32)bpf_get_current_pid_tgid();
    struct exec_cred_t *ec = exec_creds.lookup(&tid);
    u64 exec_bprm = 0;
    if (ec) {
        if (ec->cred == (u64)new) {
            exec_bprm = ec->bprm;
        }
        exec_creds.delete(&tid);
    }

    struct task_struct *task = (struct task_struct *)bpf_get_current_task();
    const struct cred *old = NULL;
    bpf_probe_read_kernel(&old, sizeof(old), &task->real_cred);
    if (!old) {
        return 0;
    }
    kuid_t ou = {}, oe = {}, os = {}, nu = {}, ne = {};
    bpf_probe_read_kernel(&ou, sizeof(ou), &old->uid);
    bpf_probe_read_kernel(&oe, sizeof(oe), &old->euid);
    bpf_probe_read_kernel(&os, sizeof(os), &old->suid);
    bpf_probe_read_kernel(&nu, sizeof(nu), &new->uid);
    bpf_probe_read_kernel(&ne, sizeof(ne), &new->euid);

    // Not a gain: already effectively root, or the new cred has no root uid/euid.
    if (oe.val == 0 || (nu.val != 0 && ne.val != 0)) {
        return 0;
    }
    u32 kind = PRIV_KIND_SETUID_EXEC;
    if (!exec_bprm) {
        // A set-user-ID program switching back to the root it holds in its
        // real or saved uid (sudo does this repeatedly) is not a gain.
        if (ou.val == 0 || os.val == 0) {
            return 0;
        }
        kind = PRIV_KIND_NO_PRIOR_ROOT;
    }

    struct privilege_event_t data = {};
    data.pid = bpf_get_current_pid_tgid() >> 32;
    data.kind = kind;
    data.old_uid = ou.val;
    data.old_euid = oe.val;
    data.old_suid = os.val;
    data.new_uid = nu.val;
    data.new_euid = ne.val;
    bpf_get_current_comm(&data.comm, sizeof(data.comm));
    if (exec_bprm) {
        const char *fn = NULL;
        bpf_probe_read_kernel(&fn, sizeof(fn), &((struct linux_binprm *)exec_bprm)->filename);
        if (fn) {
            bpf_probe_read_kernel_str(&data.filename, sizeof(data.filename), fn);
        }
    }
    if (privilege_events.ringbuf_output(&data, sizeof(data), 0) != 0) {
        count_drop(3);  // privilege_events
    }
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
        if (honeypot_events.ringbuf_output(&data, sizeof(data), 0) != 0) {
        count_drop(5);  // honeypot_events
    }
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
        if (honeypot_events.ringbuf_output(&data, sizeof(data), 0) != 0) {
        count_drop(5);  // honeypot_events
    }
    }

    if (openat2_events.ringbuf_output(&data, sizeof(data), 0) != 0) {
        count_drop(4);  // openat2_events
    }
    return 0;
}

TRACEPOINT_PROBE(syscalls, sys_enter_unlinkat) {
    struct delete_event_t data = {};
    data.pid = bpf_get_current_pid_tgid() >> 32;
    bpf_get_current_comm(&data.comm, sizeof(data.comm));
    bpf_probe_read_user_str(&data.filename, sizeof(data.filename), args->pathname);
    if (delete_events.ringbuf_output(&data, sizeof(data), 0) != 0) {
        count_drop(6);  // delete_events
    }
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

        # Events the kernel could not queue because a ring buffer was full.
        self.ringbuf_drops = {name: 0 for name in RINGBUF_STREAMS}
        self._last_drop_report = 0.0
        self._drop_read_failed = False

    def start(self):
        if not HAS_BCC:
            logger.warning("BCC not installed, cannot start eBPF engine.")
            return False
            
        try:
            self.bpf = BPF(text=bpf_text)
            
            # Register ring buffer callbacks (one shared poller for all of them)
            self.bpf["exec_events"].open_ring_buffer(self._handle_exec_event)
            self.bpf["openat2_events"].open_ring_buffer(self._handle_openat2_event)
            self.bpf["delete_events"].open_ring_buffer(self._handle_delete_event)
            self.bpf["tcp_connect_events"].open_ring_buffer(self._handle_tcp_connect)
            self.bpf["tcp_retransmit_events"].open_ring_buffer(self._handle_tcp_retransmit)
            self.bpf["privilege_events"].open_ring_buffer(self._handle_privilege_event)
            self.bpf["honeypot_events"].open_ring_buffer(self._handle_honeypot_event)
            
            self.polling_thread = threading.Thread(target=self._poll_ring_buffers, daemon=True, name="ebpf-ringbuf-poll")
            self.polling_thread.start()
            logger.info("eBPF Telemetry engine loaded and polling ring buffers.")
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

    def _poll_ring_buffers(self):
        while not self.stop_event.is_set():
            try:
                self.bpf.ring_buffer_poll(timeout=100)
                self._report_ringbuf_drops()
            except Exception as e:
                logger.error(f"Error polling BPF ring buffers: {e}")
                time.sleep(0.5)

    def _read_ringbuf_drop_totals(self) -> dict:
        """Kernel-side drop counters, summed over CPUs: {stream: total since load}."""
        tbl = self.bpf["ringbuf_drops"]
        totals = {}
        for i, name in enumerate(RINGBUF_STREAMS):
            totals[name] = int(sum(tbl[tbl.Key(i)]))
        return totals

    def _report_ringbuf_drops(self, now=None):
        """Every RINGBUF_DROP_REPORT_INTERVAL s, log new drops once, by stream name."""
        now = time.time() if now is None else now
        if now - self._last_drop_report < RINGBUF_DROP_REPORT_INTERVAL:
            return
        self._last_drop_report = now
        try:
            totals = self._read_ringbuf_drop_totals()
        except Exception as e:
            if not self._drop_read_failed:
                logger.warning(f"Could not read ring buffer drop counters: {e}")
                self._drop_read_failed = True
            return
        new = {n: t - self.ringbuf_drops.get(n, 0) for n, t in totals.items() if t > self.ringbuf_drops.get(n, 0)}
        self.ringbuf_drops.update(totals)
        if not new:
            return
        summary = ", ".join(f"{n}={c}" for n, c in sorted(new.items()))
        if "honeypot_events" in new:
            # A dropped decoy read is a missed intrusion signal, not just telemetry.
            logger.error(f"eBPF ring buffer full: dropped events: {summary}")
        else:
            logger.warning(f"eBPF ring buffer full: dropped events: {summary}")

    def get_ringbuf_drops(self) -> dict:
        """Cumulative dropped-event counts per stream, as of the last report."""
        return dict(self.ringbuf_drops)

    # --- Ring Buffer Event Callbacks ---

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
        self._record_privilege_event(
            pid=event.pid,
            comm=event.comm.decode("utf-8", "ignore"),
            kind=event.kind,
            old_ids=(event.old_uid, event.old_euid, event.old_suid),
            new_ids=(event.new_uid, event.new_euid),
            filename=event.filename.decode("utf-8", "ignore"),
        )

    def _record_privilege_event(self, *, pid, comm, kind, old_ids, new_ids, filename):
        """Classify and record one root gain reported by the commit_creds probe."""
        from security.telemetry_sanitizer import sanitize_telemetry
        clean = sanitize_telemetry({"pid": str(pid), "comm": comm, "path": filename})
        clean_pid = int(clean.get("pid") or 0) or int(pid)
        clean_comm = clean.get("comm", "")
        clean_path = clean.get("path", "")
        o_uid, o_euid, o_suid = (int(x) for x in old_ids)
        n_uid, n_euid = (int(x) for x in new_ids)
        event_type = classify_privilege_gain(kind, filename)
        details = f"uid {o_uid}->{n_uid} euid {o_euid}->{n_euid} suid {o_suid}"
        if clean_path:
            details += f" exec={clean_path}"

        with self.lock:
            self.priv_deque.append({
                "pid": clean_pid,
                "comm": clean_comm,
                "type": event_type,
                "uid": n_uid,
                "euid": n_euid,
                "old_uid": o_uid,
                "old_euid": o_euid,
                "filename": clean_path,
                "timestamp": time.time(),
            })

        if event_type == PRIV_EVENT_UNEXPECTED:
            # Alert only (axiom 5): a heuristic never triggers a response itself.
            logger.critical(
                "Unexpected root gain outside execve with no prior root identity: "
                "%s (PID %d) %s", clean_comm, clean_pid, details,
            )
        elif event_type == PRIV_EVENT_SETUID_EXEC_UNUSUAL:
            logger.warning(
                "Set-user-ID root exec from outside the system directories: "
                "%s (PID %d) %s", clean_comm, clean_pid, details,
            )
        try:
            from security.db import log_security_event
            log_security_event(
                event_type=event_type,
                source="ebpf_telemetry",
                actor=f"{clean_comm} (PID {clean_pid})",
                payload_hash=details,
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
