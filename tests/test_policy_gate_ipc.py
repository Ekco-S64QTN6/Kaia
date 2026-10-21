import json
import socket

import pytest

from security.policy_gate import PolicyGate


def _frame(payload: bytes) -> bytes:
    return len(payload).to_bytes(4, "big") + payload


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
        header = bytearray()
        while len(header) < 4:
            header.extend(left.recv(4 - len(header)))
        length = int.from_bytes(header, "big")
        body = bytearray()
        while len(body) < length:
            body.extend(left.recv(length - len(body)))
        response = json.loads(body.decode())
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
        header = bytearray()
        while len(header) < 4:
            header.extend(left.recv(4 - len(header)))
        length = int.from_bytes(header, "big")
        body = bytearray()
        while len(body) < length:
            body.extend(left.recv(length - len(body)))
        response = json.loads(body.decode())
        assert response["approved"] is False
        assert "may not override envelope fields" in response["executor_response"]["message"]
    finally:
        left.close()
