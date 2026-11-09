"""A block can never cut the host off from itself, its gateway or its resolver.

A block is `ip saddr X drop` on input. Targets came only from schema checks
(any syntactically valid address), so an injected LLM proposal, honeypot
attribution or a replayed ledger record could block loopback (every local
service), the gateway or the DNS resolver. Fixtures are the host's own
/proc/net/route and /proc/net/ipv6_route lines.
"""

import os

os.environ.setdefault("KAIA_CAPABILITY_TOKEN_SECRET", "ci-test-secret")

import pytest

from security import host_executor as he
from security.host_executor import HostExecutor, protected_block_reason

ROUTE = (
    "Iface\tDestination\tGateway \tFlags\tRefCnt\tUse\tMetric\tMask\t\tMTU\tWindow\tIRTT\n"
    "wlan1\t00000000\tFE01A8C0\t0003\t0\t0\t600\t00000000\t0\t0\t0\n"
    "wlan1\t0001A8C0\t00000000\t0001\t0\t0\t600\t00FFFFFF\t0\t0\t0\n"
)
ROUTE6 = (
    "260017026f2056b00000000000000000 40 00000000000000000000000000000000 00 "
    "00000000000000000000000000000000 00000258 00000001 00000000 00000001    wlan1\n"
    "00000000000000000000000000000000 00 00000000000000000000000000000000 00 "
    "20010db8000000000000000000000001 00000400 00000001 00000000 00000003    wlan1\n"
)


def test_gateways_are_read_from_the_kernel_wire_format(tmp_path):
    (tmp_path / "route").write_text(ROUTE)
    (tmp_path / "route6").write_text(ROUTE6)
    assert he.route_gateways(str(tmp_path / "route"), str(tmp_path / "route6")) == {"192.168.1.254", "2001:db8::1"}


def test_resolvers_are_read(tmp_path):
    (tmp_path / "r").write_text("# x\nnameserver 9.9.9.9\nnameserver fe80::1%wlan1\nsearch lan\n")
    assert he.resolver_addresses([str(tmp_path / "r"), str(tmp_path / "missing")]) == {"9.9.9.9", "fe80::1"}


HOST = lambda: ({"192.168.1.50", "2600:1702:6f20:56b0::5"}, {"192.168.1.254"}, {"9.9.9.9"})


@pytest.mark.parametrize("target,why", [
    ("127.0.0.1", "loopback"), ("127.8.8.8", "loopback"), ("::1", "loopback"),
    ("::ffff:127.0.0.1", "loopback"), ("0.0.0.0", "unspecified"), ("::", "unspecified"),
    ("224.0.0.251", "multicast"), ("ff02::1", "multicast"), ("255.255.255.255", "broadcast"),
    ("169.254.10.1", "link-local"), ("fe80::2a74:f5ff:fea3:a752", "link-local"),
    ("192.168.1.50", "this host"), ("2600:1702:6f20:56b0::5", "this host"),
    ("192.168.1.254", "gateway"), ("::ffff:192.168.1.254", "gateway"), ("9.9.9.9", "resolver"),
])
def test_protected_targets_are_refused(target, why):
    assert why in protected_block_reason(target, discover=HOST)


def test_ordinary_targets_are_allowed():
    for target in ("203.0.113.9", "198.51.100.7", "192.168.1.77", "2001:db8::99"):
        assert protected_block_reason(target, discover=HOST) == ""


def test_undiscoverable_host_refuses_rather_than_guesses():
    def broken():
        raise OSError("no /proc")
    assert "cannot determine" in protected_block_reason("203.0.113.9", discover=broken)


def test_executor_never_runs_nft_for_a_protected_target(monkeypatch):
    monkeypatch.setattr(he, "protected_block_reason", lambda a, discover=None: "a gateway of this host")
    monkeypatch.setattr(HostExecutor, "_run_cmd", staticmethod(lambda cmd: pytest.fail(f"ran {cmd}")))
    ok, _, err = HostExecutor.execute_mitigation("192.168.1.254", "all")
    assert ok is False and "gateway" in err


def test_unblock_is_never_guarded(monkeypatch):
    """Removing an existing bad block must always remain possible."""
    import inspect
    assert "protected_block_reason" not in inspect.getsource(HostExecutor.execute_unmitigation)
