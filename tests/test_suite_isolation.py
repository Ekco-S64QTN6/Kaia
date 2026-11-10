"""The test suite never talks to the production Policy Gate (tests/conftest.py)."""

import config


def test_production_socket_is_redirected():
    assert config.POLICY_GATE_SOCKET != "/run/kaiacord/policy_gate.sock"
    assert not os_path_exists(config.POLICY_GATE_SOCKET)


def test_console_status_does_not_reach_a_real_gate():
    from core.web_bridge import dispatch_operator_command
    out = dispatch_operator_command("status")
    assert "OFFLINE" in " ".join(out["lines"])


def os_path_exists(p):
    import os
    return os.path.exists(p)
