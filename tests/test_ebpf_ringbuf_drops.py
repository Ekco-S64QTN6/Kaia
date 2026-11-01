"""Ring buffer transport and per-stream drop accounting for the eBPF engine.

Per-CPU perf buffers at BCC's default size overflowed in every exec/unlink burst
and BCC logged only an anonymous "Possibly lost N samples". Streams now use ring
buffers and the kernel counts drops per stream (count_drop index -> name).
"""

import logging
import os
import re

os.environ.setdefault("KAIA_CAPABILITY_TOKEN_SECRET", "ci-test-secret")

from security import ebpf_telemetry
from security.ebpf_telemetry import EBPFTelemetryEngine, RINGBUF_STREAMS


def test_every_stream_is_a_ring_buffer_with_power_of_two_pages():
    text = ebpf_telemetry.bpf_text
    assert "BPF_PERF_OUTPUT" not in text
    assert "perf_submit" not in text
    rings = dict(re.findall(r"BPF_RINGBUF_OUTPUT\((\w+), (\d+)\)", text))
    assert set(rings) == set(RINGBUF_STREAMS)
    for name, pages in rings.items():
        n = int(pages)
        assert n > 0 and n & (n - 1) == 0, f"{name}: ring size must be a power of two pages"


def test_drop_counter_indexes_match_stream_names():
    text = ebpf_telemetry.bpf_text
    sites = re.findall(r"if \((\w+)\.ringbuf_output\(&data, sizeof\(data\), 0\) != 0\) \{\s*count_drop\((\d+)\);", text)
    assert len(sites) == text.count("ringbuf_output(")
    for name, idx in sites:
        assert RINGBUF_STREAMS[int(idx)] == name
    assert f"BPF_PERCPU_ARRAY(ringbuf_drops, u64, {len(RINGBUF_STREAMS)})" in text


def test_python_side_opens_ring_buffers_and_polls_them():
    src = open(ebpf_telemetry.__file__).read()
    assert "open_perf_buffer" not in src and "perf_buffer_poll" not in src
    for name in RINGBUF_STREAMS:
        assert f'self.bpf["{name}"].open_ring_buffer(' in src
    assert "self.bpf.ring_buffer_poll(timeout=100)" in src


class FakeDropTable:
    def __init__(self, per_cpu):
        self.per_cpu = per_cpu  # stream index -> list of per-CPU counts

    def Key(self, i):
        return i

    def __getitem__(self, key):
        return self.per_cpu.get(key, [0, 0])


def _engine_with_drops(per_cpu):
    engine = EBPFTelemetryEngine()
    engine.bpf = {"ringbuf_drops": FakeDropTable(per_cpu)}
    return engine


def test_drops_are_reported_once_by_name_and_rate_limited(caplog):
    exec_i = RINGBUF_STREAMS.index("exec_events")
    engine = _engine_with_drops({exec_i: [3, 4]})
    with caplog.at_level(logging.WARNING, logger="security.ebpf_telemetry"):
        engine._report_ringbuf_drops(now=1000.0)
        engine._report_ringbuf_drops(now=1001.0)  # inside the interval: silent
        engine._report_ringbuf_drops(now=1100.0)  # no new drops: silent
    msgs = [r.getMessage() for r in caplog.records]
    assert msgs == ["eBPF ring buffer full: dropped events: exec_events=7"]
    assert engine.get_ringbuf_drops()["exec_events"] == 7


def test_only_new_drops_are_reported():
    exec_i = RINGBUF_STREAMS.index("exec_events")
    table = FakeDropTable({exec_i: [5, 0]})
    engine = EBPFTelemetryEngine()
    engine.bpf = {"ringbuf_drops": table}
    engine._report_ringbuf_drops(now=1000.0)
    table.per_cpu[exec_i] = [5, 2]
    engine._report_ringbuf_drops(now=1100.0)
    assert engine.get_ringbuf_drops()["exec_events"] == 7


def test_honeypot_drops_are_errors(caplog):
    hp = RINGBUF_STREAMS.index("honeypot_events")
    engine = _engine_with_drops({hp: [1, 0]})
    with caplog.at_level(logging.WARNING, logger="security.ebpf_telemetry"):
        engine._report_ringbuf_drops(now=1000.0)
    assert [r.levelno for r in caplog.records] == [logging.ERROR]
    assert "honeypot_events=1" in caplog.records[0].getMessage()
