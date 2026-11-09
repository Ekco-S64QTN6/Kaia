"""Control health: every control says what it is doing, and silence is noticed.

On this host the eBPF program did not compile for a week, kaia-restore-blocks
failed on every boot, and no lockdown ever reached the HUD; none of it was
reported. The gate now publishes each control's state with heartbeats, the HUD
shows it, and kaia_health_check.py (a timer, outside the gate) flags a gate
that stopped publishing or a control whose thread stopped beating.
"""

import importlib.util
import json
import logging
import os
import stat
import threading
import types
from pathlib import Path

os.environ.setdefault("KAIA_CAPABILITY_TOKEN_SECRET", "ci-test-secret")

import pytest

from security import control_health
from security.control_health import ControlHealth, evaluate, stale_after

ROOT = Path(__file__).resolve().parent.parent


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


def _registry():
    clock = Clock()
    return ControlHealth(clock=clock), clock


def _by_name(report):
    return {c["name"]: c for c in report["controls"]}


# --- registry ---

def test_since_moves_only_when_the_state_changes():
    reg, clock = _registry()
    reg.report("ebpf", "active", "loaded")
    clock.t = 1100
    reg.report("ebpf", "active", "drops: openat2_events=5")
    entry = reg.snapshot()["controls"]["ebpf"]
    assert entry["since"] == 1000 and entry["detail"] == "drops: openat2_events=5"
    clock.t = 1200
    reg.report("ebpf", "failed", "gone")
    assert reg.snapshot()["controls"]["ebpf"]["since"] == 1200


def test_unknown_state_is_rejected_but_module_report_never_raises(caplog):
    reg, _ = _registry()
    with pytest.raises(ValueError):
        reg.report("x", "fine")
    with caplog.at_level(logging.ERROR, logger="security.control_health"):
        control_health.report("x", "fine")  # must not take a control down
    assert any("failed" in r.getMessage() for r in caplog.records)


def test_heartbeat_for_an_unregistered_control_is_ignored():
    reg, _ = _registry()
    reg.heartbeat("nobody")
    assert reg.snapshot()["controls"] == {}


def test_detail_is_bounded():
    reg, _ = _registry()
    reg.report("x", "degraded", "y" * 5000)
    assert len(reg.snapshot()["controls"]["x"]["detail"]) == control_health.MAX_DETAIL_CHARS


# --- evaluation ---

def test_beating_controls_are_active_and_off_is_not_an_alarm():
    reg, clock = _registry()
    reg.report("ebpf", "active", "", heartbeat_interval=5)
    reg.report("decoy_listeners", "off", "namespace not provisioned")
    clock.t += 4
    reg.heartbeat("ebpf")
    report = evaluate(reg.snapshot(), now=clock.t + 1)
    assert report["overall"] == "active" and report["fresh"]
    assert _by_name(report)["decoy_listeners"]["state"] == "off"


def test_a_control_that_stops_beating_is_stale_and_fails_overall():
    reg, clock = _registry()
    reg.report("tamper_detection", "active", "28 files", heartbeat_interval=30)
    clock.t += stale_after(30) + 1  # the thread died; the publisher still runs
    report = evaluate(reg.snapshot(), now=clock.t)
    c = _by_name(report)["tamper_detection"]
    assert c["state"] == "stale" and "no heartbeat for" in c["detail"] and "28 files" in c["detail"]
    assert report["overall"] == "failed"


def test_degraded_is_reported_as_degraded():
    reg, clock = _registry()
    reg.report("fim", "degraded", "fanotify unavailable")
    assert evaluate(reg.snapshot(), now=clock.t)["overall"] == "degraded"


def test_an_old_file_means_the_gate_is_down_or_hung():
    reg, clock = _registry()
    reg.report("ebpf", "active", "")
    report = evaluate(reg.snapshot(), now=clock.t + control_health.MAX_FILE_AGE + 1)
    assert report["overall"] == "failed" and not report["fresh"]
    assert "down or hung" in report["problem"]


def test_future_or_missing_timestamps_are_not_fresh():
    assert evaluate({"written_at": 2000.0, "controls": {}}, now=1000.0)["overall"] == "failed"
    assert evaluate({"controls": {}}, now=1000.0)["overall"] == "failed"
    assert evaluate({"written_at": True, "controls": {}}, now=1.0)["overall"] == "failed"


def test_nothing_published_is_failed():
    report = evaluate(None, now=1.0, error="missing")
    assert report == {"overall": "failed", "fresh": False, "age": None, "problem": "missing", "controls": []}


def test_malformed_entries_never_read_as_healthy():
    snap = {"written_at": 1000.0, "controls": {
        "a": {"state": "fine"}, "b": "not a dict", "c": {"state": "active"},
    }}
    report = evaluate(snap, now=1000.0)
    names = _by_name(report)
    assert names["a"]["state"] == "failed" and "b" not in names and names["c"]["state"] == "active"
    assert report["overall"] == "failed"


# --- file I/O ---

def test_written_file_is_atomic_group_readable_and_round_trips(tmp_path):
    reg, clock = _registry()
    reg.report("ebpf", "active", "loaded", heartbeat_interval=5)
    path = tmp_path / "control_health.json"
    control_health.write_snapshot(str(path), reg.snapshot())
    assert stat.S_IMODE(path.stat().st_mode) == 0o640
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".")]
    report = control_health.load_and_evaluate(str(path), now=clock.t)
    assert report["overall"] == "active"


def test_reader_refuses_symlinks_non_regular_files_and_oversize(tmp_path):
    real = tmp_path / "real.json"
    real.write_text(json.dumps({"written_at": 1.0, "controls": {}}))
    link = tmp_path / "link.json"
    link.symlink_to(real)
    assert "cannot read" in control_health.load_and_evaluate(str(link), now=1.0)["problem"]
    assert "cannot read" in control_health.load_and_evaluate(str(tmp_path), now=1.0)["problem"]
    big = tmp_path / "big.json"
    big.write_text(" " * (control_health.MAX_FILE_BYTES + 1))
    assert "too large" in control_health.load_and_evaluate(str(big), now=1.0)["problem"]
    missing = control_health.load_and_evaluate(str(tmp_path / "nope.json"), now=1.0)
    assert missing["overall"] == "failed" and "has not published" in missing["problem"]


def test_writer_does_not_follow_a_planted_temp_symlink(tmp_path):
    target = tmp_path / "victim"
    target.write_text("keep")
    path = tmp_path / "control_health.json"
    tmp = tmp_path / f".control_health.json.{os.getpid()}.tmp"
    tmp.symlink_to(target)
    control_health.write_snapshot(str(path), {"written_at": 1.0, "controls": {}})
    assert target.read_text() == "keep"
    assert json.loads(path.read_text())["written_at"] == 1.0


def test_publisher_logs_a_failure_once_and_recovers(tmp_path, caplog):
    reg, _ = _registry()
    pub = control_health.HealthPublisher(str(tmp_path / "missing-dir" / "h.json"), reg)
    with caplog.at_level(logging.INFO, logger="security.control_health"):
        assert pub.publish_once() is False
        assert pub.publish_once() is False
        pub.path = str(tmp_path / "h.json")
        assert pub.publish_once() is True
    msgs = [r.getMessage() for r in caplog.records]
    assert sum("Could not publish" in m for m in msgs) == 1
    assert any("recovered" in m for m in msgs)


# --- the independent checker ---

def _checker():
    spec = importlib.util.spec_from_file_location("kaia_health_check", ROOT / "scripts/kaia_health_check.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeSystemctl:
    def __init__(self, props):
        self.props = props  # unit -> "Key=Value\n..."

    def __call__(self, argv, **kwargs):
        unit = argv[2]
        return types.SimpleNamespace(returncode=0, stdout=self.props[unit], stderr="")


HEALTHY_UNITS = {
    "kaia-policy-gate.service": "LoadState=loaded\nActiveState=active\nSubState=running\nResult=success\n",
    "kaia-restore-blocks.service": "LoadState=loaded\nActiveState=active\nSubState=exited\nResult=success\n",
}


def _published(tmp_path, controls, written_at=1000.0):
    path = tmp_path / "control_health.json"
    path.write_text(json.dumps({"written_at": written_at, "controls": controls}))
    return str(path)


def test_checker_is_quiet_and_zero_when_everything_runs(tmp_path):
    checker = _checker()
    path = _published(tmp_path, {"ebpf": {"state": "active"}, "decoy_listeners": {"state": "off"}})
    report = checker.run(path, now=1001.0, runner=FakeSystemctl(HEALTHY_UNITS))
    assert report["exit_status"] == 0
    lines = checker.format_lines(report)
    assert len(lines) == 1 and lines[0].startswith("Kaia controls: ACTIVE")


def test_restore_blocks_failing_every_boot_fails_the_check(tmp_path):
    checker = _checker()
    units = dict(HEALTHY_UNITS)
    units["kaia-restore-blocks.service"] = "LoadState=loaded\nActiveState=failed\nSubState=failed\nResult=exit-code\n"
    path = _published(tmp_path, {"ebpf": {"state": "active"}})
    report = checker.run(path, now=1001.0, runner=FakeSystemctl(units))
    assert report["exit_status"] == 2
    assert any("kaia-restore-blocks.service: failed" in line for line in checker.format_lines(report))


def test_gate_not_running_or_not_installed_fails(tmp_path):
    checker = _checker()
    path = _published(tmp_path, {"ebpf": {"state": "active"}})
    for gate in ("LoadState=loaded\nActiveState=inactive\nSubState=dead\nResult=success\n",
                 "LoadState=not-found\nActiveState=inactive\nSubState=dead\nResult=success\n"):
        units = dict(HEALTHY_UNITS, **{"kaia-policy-gate.service": gate})
        assert checker.run(path, now=1001.0, runner=FakeSystemctl(units))["exit_status"] == 2


def test_restore_blocks_inactive_after_install_is_fine_but_disabled_is_not(tmp_path):
    """Host finding 2026-11-09: the installer stops the boot oneshot; until the next
    boot it is inactive with a successful result, which is not a fault."""
    checker = _checker()
    path = _published(tmp_path, {"ebpf": {"state": "active"}})
    after_install = "LoadState=loaded\nActiveState=inactive\nSubState=dead\nResult=success\nUnitFileState=enabled\n"
    units = dict(HEALTHY_UNITS, **{"kaia-restore-blocks.service": after_install})
    assert checker.run(path, now=1001.0, runner=FakeSystemctl(units))["exit_status"] == 0
    disabled = after_install.replace("UnitFileState=enabled", "UnitFileState=disabled")
    units = dict(HEALTHY_UNITS, **{"kaia-restore-blocks.service": disabled})
    report = checker.run(path, now=1001.0, runner=FakeSystemctl(units))
    assert report["exit_status"] == 1 and "will not run at boot" in checker.format_lines(report)[-1]


def test_timer_does_not_fire_at_enable_time():
    timer = (ROOT / "scripts/kaia-health-check.timer").read_text()
    assert "OnActiveSec=5min" in timer and "\nOnBootSec" not in timer


def test_restore_blocks_not_installed_is_off(tmp_path):
    checker = _checker()
    units = dict(HEALTHY_UNITS, **{"kaia-restore-blocks.service": "LoadState=not-found\nActiveState=inactive\nSubState=dead\nResult=success\n"})
    path = _published(tmp_path, {"ebpf": {"state": "active"}})
    report = checker.run(path, now=1001.0, runner=FakeSystemctl(units))
    assert report["exit_status"] == 0
    assert {u["name"]: u["state"] for u in report["units"]}["kaia-restore-blocks.service"] == "off"


def test_degraded_exits_one_and_journal_lines_carry_priorities(tmp_path):
    checker = _checker()
    path = _published(tmp_path, {"fim": {"state": "degraded", "detail": "fanotify unavailable"}})
    report = checker.run(path, now=1001.0, runner=FakeSystemctl(HEALTHY_UNITS))
    assert report["exit_status"] == 1
    lines = checker.format_lines(report, journal=True)
    assert lines[0].startswith("<4>Kaia controls: DEGRADED")
    assert "<4>  control fim: degraded: fanotify unavailable" in lines


def test_checker_main_reports_a_stale_gate(tmp_path, capsys):
    checker = _checker()
    path = _published(tmp_path, {"ebpf": {"state": "active"}}, written_at=1.0)
    assert checker.main(["--path", path, "--no-units"]) == 2
    out = capsys.readouterr().out
    assert "FAILED" in out and "down or hung" in out


def test_checker_does_not_import_the_gate_configuration():
    source = (ROOT / "scripts/kaia_health_check.py").read_text()
    assert "import config" not in source and "from core" not in source
    assert "import config" not in (ROOT / "security/control_health.py").read_text()


# --- wiring ---

def test_tamper_loop_survives_an_exception_and_keeps_beating(monkeypatch):
    from security import tamper_detection
    monkeypatch.setattr(tamper_detection, "TAMPER_CHECK_INTERVAL", 0.001)
    beats = []
    monkeypatch.setattr(control_health, "heartbeat", lambda name: beats.append(name))
    monkeypatch.setattr(control_health, "report", lambda *a, **k: None)
    calls = []
    stop = threading.Event()

    def check():
        calls.append(1)
        if len(calls) == 1:
            raise OSError("transient")
        if len(calls) >= 3:
            stop.set()

    fake = types.SimpleNamespace(_stop_event=stop, check_integrity=check)
    tamper_detection.TamperDetector._run(fake)
    assert len(calls) >= 3 and beats.count("tamper_detection") >= 1


def _report_capture(monkeypatch):
    seen = {}
    monkeypatch.setattr(control_health, "report",
                        lambda name, state, detail="", heartbeat_interval=None: seen.__setitem__(name, (state, detail)))
    return seen


def test_honeypot_health_counts_what_is_really_watched(monkeypatch):
    from security.honeypot import HoneypotCoordinator
    seen = _report_capture(monkeypatch)
    fake = types.SimpleNamespace(decoys={"a": "", "b": "", "c": ""}, canary_paths=["f", "r"])
    ebpf = types.SimpleNamespace(bpf=object())
    HoneypotCoordinator._report_health(fake, ebpf, ["a", "b", "f", "r"], 4)
    assert seen["honeypot"][0] == "degraded" and "4 of 5" in seen["honeypot"][1]
    HoneypotCoordinator._report_health(fake, ebpf, ["a", "b", "c", "f", "r"], 5)
    assert seen["honeypot"][0] == "active"
    HoneypotCoordinator._report_health(fake, types.SimpleNamespace(bpf=None), ["a"], None)
    assert seen["honeypot"][0] == "failed" and "eBPF not loaded" in seen["honeypot"][1]


def test_ebpf_load_failure_is_reported_failed(monkeypatch):
    from security import ebpf_telemetry
    seen = _report_capture(monkeypatch)
    monkeypatch.setattr(ebpf_telemetry, "HAS_BCC", True)

    def broken(text):
        raise Exception("Failed to compile BPF module <text>")

    monkeypatch.setattr(ebpf_telemetry, "BPF", broken, raising=False)
    assert ebpf_telemetry.EBPFTelemetryEngine().start() is False
    assert seen["ebpf"] == ("failed", "probes did not load: Failed to compile BPF module <text>")


def test_every_long_running_control_beats_in_its_loop():
    expect = {
        "security/ebpf_telemetry.py": 'control_health.heartbeat("ebpf")',
        "security/fim_daemon.py": 'control_health.heartbeat("fim")',
        "security/network_discovery.py": 'control_health.heartbeat("discovery")',
        "security/supply_chain_monitor.py": 'control_health.heartbeat("pacman_monitor")',
        "security/honeypot.py": 'control_health.heartbeat("honeypot")',
        "security/tamper_detection.py": 'control_health.heartbeat("tamper_detection")',
        "security/policy_gate.py": 'control_health.heartbeat("policy_gate")',
    }
    for path, call in expect.items():
        assert call in (ROOT / path).read_text(), path
    gate = (ROOT / "security/policy_gate.py").read_text()
    assert "HealthPublisher(config.CONTROL_HEALTH_PATH)" in gate
    assert '"fim", "degraded"' in gate


def test_health_check_unit_is_unprivileged_read_only_and_installed():
    unit = (ROOT / "scripts/kaia-health-check.service").read_text()
    for line in ("DynamicUser=yes", "SupplementaryGroups=kaiacord", "CapabilityBoundingSet=\n",
                 "NoNewPrivileges=true", "ReadOnlyPaths=__STATE_DIR__", "SuccessExitStatus=1",
                 "RestrictAddressFamilies=AF_UNIX", "-I -s __DEPLOY_DIR__/scripts/kaia_health_check.py"):
        assert line in unit, line
    assert "ReadWritePaths" not in unit
    timer = (ROOT / "scripts/kaia-health-check.timer").read_text()
    assert "OnUnitActiveSec=5min" in timer
    installer = (ROOT / "scripts/install_services.sh").read_text()
    assert "kaia-health-check.service kaia-health-check.timer" in installer
    assert "systemctl enable --now kaia-health-check.timer" in installer
    tamper = (ROOT / "security/tamper_detection.py").read_text()
    assert '"/etc/systemd/system/kaia-health-check.service"' in tamper


def test_posture_follows_failed_controls_but_not_standing_degradation():
    from kaia_dashboard import KaiamonSnapshot
    from core.web_bridge import serialize_snapshot

    def snap(overall):
        return KaiamonSnapshot(
            system_security={"gate_running": True, "lockdown_active": False,
                             "control_health": {"overall": overall, "fresh": True, "controls": []}},
            containment={"sentinel_alerts": 0},
        )

    assert serialize_snapshot(snap("failed"))["sentinel_state"] == "AnomalyDetected"
    assert serialize_snapshot(snap("degraded"))["sentinel_state"] != "AnomalyDetected"


def test_hud_escapes_control_details():
    js = (ROOT / "core/web/static/js/app.js").read_text()
    assert "function renderControls(health)" in js
    assert 'title="${esc(c.detail)}"' in js and "${esc(c.name)}" in js
    assert 'id="controls-strip"' in (ROOT / "core/web/static/index.html").read_text()


def test_curses_summary_names_what_is_not_active():
    from kaia_dashboard import controls_summary
    health = {"overall": "degraded", "fresh": True, "controls": [
        {"name": "fim", "state": "degraded"}, {"name": "ebpf", "state": "active"},
        {"name": "decoy_listeners", "state": "off"}]}
    assert controls_summary(health) == "Controls: DEGRADED (fim degraded)"
    assert controls_summary({"fresh": False, "problem": "old"}) == "Controls: NOT REPORTING (old)"
    assert controls_summary(None) == "Controls: unknown"


def test_every_gate_control_starts_failed_and_reports_for_itself():
    import re
    gate = (ROOT / "security/policy_gate.py").read_text()
    assert "for _name in control_health.GATE_CONTROLS:" in gate
    assert gate.index("control_health.GATE_CONTROLS") < gate.index("security.db.initialize_db()\n")
    sources = "".join((ROOT / "security" / p).read_text() for p in os.listdir(ROOT / "security") if p.endswith(".py"))
    reported = set(re.findall(r'control_health\.report\(\s*"(\w+)"', sources))
    assert set(control_health.GATE_CONTROLS) <= reported
    assert "health_publisher.start(publish_now=False)" in gate
