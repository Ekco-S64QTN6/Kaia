import os
os.environ.setdefault("KAIA_CAPABILITY_TOKEN_SECRET", "test_signing_secret_key_2026")

import json
import socket

import pytest

from security.policy_gate import PolicyGate


def _frame(payload: bytes) -> bytes:
    return len(payload).to_bytes(4, "big") + payload


def _recv_frame(sock: socket.socket) -> dict:
    """Read one complete length-prefixed response from a test socket."""
    header = bytearray()
    while len(header) < 4:
        packet = sock.recv(4 - len(header))
        if not packet:
            raise AssertionError("IPC peer closed before response header")
        header.extend(packet)

    length = int.from_bytes(header, "big")
    payload = bytearray()
    while len(payload) < length:
        packet = sock.recv(length - len(payload))
        if not packet:
            raise AssertionError("IPC peer closed before complete response")
        payload.extend(packet)
    return json.loads(payload.decode("utf-8"))


class _FragmentedSocket:
    """Socket-shaped test double that returns a frame in deliberate fragments."""

    def __init__(self, fragments):
        self.fragments = list(fragments)
        self.timeouts = []

    def settimeout(self, timeout):
        self.timeouts.append(timeout)

    def recv(self, size):
        if not self.fragments:
            return b""
        fragment = self.fragments[0]
        packet = fragment[:size]
        remainder = fragment[size:]
        if remainder:
            self.fragments[0] = remainder
        else:
            self.fragments.pop(0)
        return packet


def test_ipc_reads_fragmented_header_and_payload_exactly():
    gate = PolicyGate()
    expected = {
        "request_id": "fragmented",
        "action": "diagnostics",
        "payload": {"query_type": "ss"},
    }
    wire = _frame(json.dumps(expected).encode("utf-8"))
    # Split every part of the 4-byte header and several parts of the JSON body.
    sock = _FragmentedSocket([
        wire[:1],
        wire[1:3],
        wire[3:4],
        wire[4:5],
        wire[5:11],
        wire[11:],
    ])

    assert gate._read_frame(sock) == expected
    assert len(sock.timeouts) >= 6
    assert sock.fragments == []


@pytest.mark.parametrize(
    "truncated_frame",
    [
        pytest.param(bytes([0, 0]), id="partial-header"),
        pytest.param((8).to_bytes(4, "big") + b'{"a"', id="partial-payload"),
    ],
)
def test_ipc_closes_truncated_frame_without_dispatching(truncated_frame):
    gate = PolicyGate()
    left, right = socket.socketpair()
    gate.evaluate_and_execute = lambda payload: pytest.fail(
        "truncated IPC frames must never reach policy evaluation"
    )
    try:
        left.sendall(truncated_frame)
        left.shutdown(socket.SHUT_WR)

        gate._handle_client(right)

        assert right.fileno() == -1, "client socket must close after a truncated frame"
        assert left.recv(1) == b"", "truncated frames must not receive a success or error response"
    finally:
        left.close()
        right.close()


def test_ipc_rejects_oversized_frame_before_reading_payload():
    gate = PolicyGate()
    left, right = socket.socketpair()
    try:
        left.sendall((gate.MAX_IPC_FRAME_BYTES + 1).to_bytes(4, "big"))
        with pytest.raises(ValueError, match="Invalid IPC frame length"):
            gate._read_frame(right)
    finally:
        left.close()
        right.close()


def test_ipc_client_closes_connection_after_invalid_frame():
    gate = PolicyGate()
    left, right = socket.socketpair()
    try:
        left.sendall((gate.MAX_IPC_FRAME_BYTES + 1).to_bytes(4, "big"))
        gate._handle_client(right)

        response = _recv_frame(left)
        assert response["approved"] is False
        assert response["executor_response"]["status"] == "error"
        assert "Invalid IPC frame length" in response["executor_response"]["message"]
        assert right.fileno() == -1, "client socket must close after a protocol error"
    finally:
        left.close()
        right.close()


def test_ipc_rejects_non_object_json():
    gate = PolicyGate()
    left, right = socket.socketpair()
    try:
        left.sendall(_frame(json.dumps(["not", "an", "envelope"]).encode()))
        with pytest.raises(ValueError, match="JSON object"):
            gate._read_frame(right)
    finally:
        left.close()
        right.close()


def test_ipc_accepts_valid_object_frame():
    gate = PolicyGate()
    left, right = socket.socketpair()
    try:
        expected = {"action": "diagnostics", "payload": {"query_type": "ss"}}
        left.sendall(_frame(json.dumps(expected).encode()))
        assert gate._read_frame(right) == expected
    finally:
        left.close()
        right.close()


def test_ipc_approval_flag_is_false_when_executor_errors():
    gate = PolicyGate()
    left, right = socket.socketpair()
    gate.evaluate_and_execute = lambda payload: {"status": "error", "message": "failed"}
    try:
        left.sendall(_frame(json.dumps({
            "request_id": "test",
            "action": "diagnostics",
            "payload": {},
        }).encode()))
        gate._handle_client(right)
        response = _recv_frame(left)
        assert response["approved"] is False
        assert response["executor_response"]["status"] == "error"
    finally:
        left.close()


def test_ipc_payload_cannot_override_envelope_fields():
    gate = PolicyGate()
    left, right = socket.socketpair()
    try:
        left.sendall(_frame(json.dumps({
            "request_id": "test",
            "action": "diagnostics",
            "capability_token": None,
            "payload": {"action": "lockdown", "capability_token": "attacker-value"},
        }).encode()))
        gate.evaluate_and_execute = lambda payload: pytest.fail("invalid envelope must not be evaluated")
        gate._handle_client(right)
        response = _recv_frame(left)
        assert response["approved"] is False
        assert "may not override envelope fields" in response["executor_response"]["message"]
    finally:
        left.close()


def test_unblock_ip_requires_a_valid_scoped_capability(monkeypatch):
    import os
    os.environ.setdefault("KAIA_CAPABILITY_TOKEN_SECRET", "test_signing_secret_key_2026")
    import config
    import security.policy_gate as policy_gate_module
    from security.policy_gate import generate_capability_token

    monkeypatch.setattr(config, "GLOBAL_PERMISSIONS", {"unblock_ip"})
    monkeypatch.setattr(config, "WORKSPACE_PERMISSIONS", {"unblock_ip"})
    calls = []
    monkeypatch.setattr(
        policy_gate_module.HostExecutor,
        "execute_unmitigation",
        lambda *args: calls.append(args) or (True, "removed", ""),
    )
    monkeypatch.setattr(PolicyGate, "_log_audit", lambda *args, **kwargs: None)
    monkeypatch.setattr(policy_gate_module, "log_security_event", lambda *args, **kwargs: None)

    gate = PolicyGate()
    target = "192.0.2.55"
    token = generate_capability_token("unblock_ip", target, duration_seconds=30)
    result = gate.evaluate_and_execute({
        "action": "unblock_ip", "target_ip": target, "protocol": "all", "port": None,
        "justification": "test unblocking", "capability_token": token, "session_id": "unit_test",
    })

    assert result["status"] == "success"
    assert calls == [(target, "all", None)]


def test_unblock_ip_rejects_invalid_capability_without_executor_call(monkeypatch):
    import config
    import security.policy_gate as policy_gate_module

    monkeypatch.setattr(config, "GLOBAL_PERMISSIONS", {"unblock_ip"})
    monkeypatch.setattr(config, "WORKSPACE_PERMISSIONS", {"unblock_ip"})
    monkeypatch.setattr(PolicyGate, "_log_audit", lambda *args, **kwargs: None)
    monkeypatch.setattr(policy_gate_module, "log_security_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        policy_gate_module.HostExecutor,
        "execute_unmitigation",
        lambda *args: pytest.fail("invalid tokens must not reach the executor"),
    )
    result = PolicyGate().evaluate_and_execute({
        "action": "unblock_ip", "target_ip": "192.0.2.55", "protocol": "all", "port": None,
        "justification": "test unblocking", "capability_token": "invalid", "session_id": "unit_test",
    })
    assert result["status"] == "denied"


def test_block_ip_schema_rejects_invalid_ip_and_ambiguous_port_scope(monkeypatch):
    import os
    os.environ.setdefault("KAIA_CAPABILITY_TOKEN_SECRET", "test_signing_secret_key_2026")
    from pydantic import ValidationError
    from security.schemas import MitigationRequest

    base = {
        "target_ip": "192.0.2.55",
        "protocol": "all",
        "port": None,
        "justification": "unit-test block request",
        "capability_token": "ck_test",
        "session_id": "unit_test",
    }
    assert MitigationRequest(**base).target_ip == "192.0.2.55"

    for changes in [
        {"target_ip": "192.0.2.1/24"},
        {"protocol": "all", "port": 53},
        {"protocol": "tcp", "port": None},
        {"protocol": "udp", "port": None},
    ]:
        with pytest.raises(ValidationError):
            MitigationRequest(**{**base, **changes})


def test_mitigation_and_unblock_schemas_reject_boolean_ports():
    from pydantic import ValidationError
    from security.schemas import MitigationRequest, UnblockRequest

    for model in (MitigationRequest, UnblockRequest):
        with pytest.raises(ValidationError):
            model(
                target_ip="192.0.2.9",
                protocol="tcp",
                port=True,
                justification="strict numeric port test",
                capability_token="test-token",
                session_id="unit_test",
            )

def test_root_policy_gate_refuses_tmp_fallback_when_runtime_directory_is_unavailable(monkeypatch):
    import config
    import security.policy_gate as policy_gate_module

    gate = PolicyGate()
    gate.is_running = True
    runtime_dir = os.path.dirname(config.POLICY_GATE_SOCKET)
    real_makedirs = os.makedirs

    def deny_runtime_dir(path, *args, **kwargs):
        if path == runtime_dir:
            raise PermissionError("simulated runtime directory failure")
        return real_makedirs(path, *args, **kwargs)

    monkeypatch.setattr(policy_gate_module.os, "makedirs", deny_runtime_dir)
    monkeypatch.setattr(policy_gate_module.os, "geteuid", lambda: 0)
    monkeypatch.setattr(
        policy_gate_module.socket,
        "socket",
        lambda *args, **kwargs: pytest.fail("root must not bind a fallback socket"),
    )

    gate._run_server()

    assert gate.is_running is False
    assert gate.socket_path == config.POLICY_GATE_SOCKET

def test_policy_gate_fallback_path_is_scoped_to_current_uid():
    import config

    expected_parent = os.path.join("/tmp", f"kaia-policy-gate-{os.getuid()}")
    assert os.path.dirname(config.POLICY_GATE_SOCKET_FALLBACK) == expected_parent

def test_ipc_payload_uses_one_absolute_deadline_across_partial_reads(monkeypatch):
    import security.policy_gate as policy_gate_module

    gate = PolicyGate()
    gate.MAX_IPC_FRAME_READ_SECONDS = 5.0
    left, right = socket.socketpair()
    try:
        # Advertise a longer frame, but send only its first payload byte. Fake
        # monotonic time crosses the one deadline between payload recv attempts;
        # a timeout-resetting-per-recv implementation would not reject this yet.
        left.sendall((100).to_bytes(4, "big") + b"{")
        clock = iter([0.0, 0.0, 0.0, 6.0])
        monkeypatch.setattr(policy_gate_module.time, "monotonic", lambda: next(clock))

        with pytest.raises(TimeoutError, match="frame read deadline"):
            gate._read_frame(right)
    finally:
        left.close()
        right.close()


def test_ipc_frame_timeout_is_bounded_by_a_single_deadline():
    gate = PolicyGate()
    assert gate.MAX_IPC_FRAME_READ_SECONDS > 0
    assert gate.MAX_IPC_FRAME_READ_SECONDS <= 10.0


def test_ipc_writer_replaces_oversized_response_with_bounded_error_frame():
    gate = PolicyGate()
    gate.MAX_IPC_RESPONSE_BYTES = 1024
    left, right = socket.socketpair()
    try:
        gate._write_frame(right, {
            "request_id": "large-response",
            "approved": True,
            "executor_response": {"status": "success", "stdout": "x" * 10000},
            "audit_id": "audit_test",
        })

        header = bytearray()
        while len(header) < 4:
            header.extend(left.recv(4 - len(header)))
        length = int.from_bytes(header, "big")
        assert 0 < length <= gate.MAX_IPC_RESPONSE_BYTES

        body = bytearray()
        while len(body) < length:
            body.extend(left.recv(length - len(body)))
        response = json.loads(body.decode("utf-8"))
        assert response["approved"] is False
        assert "exceeded" in response["executor_response"]["message"].lower()
    finally:
        left.close()
        right.close()
