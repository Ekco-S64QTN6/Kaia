"""Compile the real eBPF program with the local kernel headers.

From 2026-10-25 to 2026-11-01 one missing header (`<net/sock.h>`) made the whole
program fail to compile, and every probe, honeypot watching included, was off
without a test noticing: the other eBPF tests only read the C source as text.

This test runs BCC's clang front end over `bpf_text` and fails on any compiler
error. It skips where BCC or kernel headers are absent (GitHub's runners), so it
guards local development, not CI.

Unprivileged, BCC cannot read the tracepoint formats under tracefs, so
`TRACEPOINT_PROBE` argument types would stay incomplete. The test renames those
probes to plain functions over a stand-in argument struct with the fields the
program uses; every probe body is then type-checked against the real kernel
headers. BCC then stops at map creation (EPERM), before code generation: the
512-byte stack limit, the verifier and attachment are still proven only by
loading on the host (DEPLOYMENT_CHECKLIST step 3).
"""

import os
import platform
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

CHILD = r"""
import re, sys
sys.path[:0] = [sys.argv[1], sys.argv[1] + "/core"]
from bcc import BPF
from security.ebpf_telemetry import bpf_text

stub = '''
struct kaia_tp_args {
    u64 __pad; int __syscall_nr; int dfd;
    const char *filename; const char *pathname; int flags;
};
'''
src = re.sub(r"TRACEPOINT_PROBE\((\w+), (\w+)\)",
             r"int kaia_tp__\1__\2(struct kaia_tp_args *args)", bpf_text)
marker = "#include <linux/binfmts.h>\n"
assert marker in src
src = src.replace(marker, marker + stub, 1)
try:
    BPF(text=src)
    print("BCC-RESULT: loaded")
except Exception as e:
    print("BCC-RESULT:", e)
"""

STUB_FIELDS = {"filename", "pathname", "flags"}


def _bcc_available() -> bool:
    try:
        r = subprocess.run([sys.executable, "-c", "import bcc"], capture_output=True, timeout=30)
    except Exception:
        return False
    return r.returncode == 0


def test_stub_covers_every_tracepoint_argument_the_program_reads():
    from security.ebpf_telemetry import bpf_text
    used = set(re.findall(r"args->(\w+)", bpf_text))
    assert used <= STUB_FIELDS, f"add {used - STUB_FIELDS} to the stand-in struct in CHILD"


@pytest.mark.skipif(not _bcc_available(), reason="BCC python bindings not installed")
@pytest.mark.skipif(
    not os.path.isdir(f"/lib/modules/{platform.release()}/build"),
    reason="kernel headers for the running kernel not installed",
)
@pytest.mark.skipif(os.geteuid() == 0, reason="as root this would load and attach the probes")
def test_bpf_program_compiles_against_local_kernel_headers():
    env = dict(os.environ, KAIA_CAPABILITY_TOKEN_SECRET="ci-test-secret")
    r = subprocess.run(
        [sys.executable, "-c", CHILD, str(REPO)],
        capture_output=True, text=True, timeout=300, env=env,
    )
    out = r.stdout + r.stderr
    errors = re.findall(r"^\S+:\d+:\d+: error: .*$", out, re.MULTILINE)
    assert not errors, "eBPF program does not compile:\n" + "\n".join(errors) + "\n\n" + out[-4000:]

    # Past the front end, an unprivileged BCC fails creating the first map.
    assert "BCC-RESULT:" in out, out[-4000:]
    assert re.search(r"could not open bpf map: \w+, error: Operation not permitted", out), out[-4000:]
