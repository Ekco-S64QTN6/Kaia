"""Alert-only watch on sudoers / passwd / group / ld.so.preload.

These files change legitimately, so the watch must only ever record: never
raise into the tamper detector, never trigger lockdown. It must still catch the
persistence moves that matter (a NOPASSWD rule, a uid-0 account, wheel
membership, a preload library), including changes made while Kaia was stopped.
"""

import logging
import os
import stat

os.environ.setdefault("KAIA_CAPABILITY_TOKEN_SECRET", "ci-test-secret")

import pytest

from security.privilege_watch import PrivilegeConfigWatcher, sudo_reads

PASSWD = "root:x:0:0::/root:/bin/bash\nekco:x:1000:1000::/home/ekco:/bin/bash\n"
GROUP = "root:x:0:root\nwheel:x:998:\n"
SUDOERS = "root ALL=(ALL:ALL) ALL\n@includedir /etc/sudoers.d\n"


@pytest.fixture
def host(tmp_path):
    etc = tmp_path / "etc"
    (etc / "sudoers.d").mkdir(parents=True)
    (etc / "sudoers").write_text(SUDOERS)
    (etc / "passwd").write_text(PASSWD)
    (etc / "group").write_text(GROUP)
    events = []

    def make():
        return PrivilegeConfigWatcher(
            tmp_path / "state" / "privilege_watch.json",
            files=(str(etc / "sudoers"), str(etc / "passwd"), str(etc / "group"), str(etc / "ld.so.preload")),
            dirs=(str(etc / "sudoers.d"),),
            log_event=lambda **kw: events.append(kw),
        )

    return etc, make, events, tmp_path / "state" / "privilege_watch.json"


def _types(events):
    return [e["event_type"] for e in events]


def test_sudo_reads_matches_sudoers_includedir_rules():
    assert sudo_reads("50-ekco-readonly")
    assert not sudo_reads("50-ekco-readonly.tmp")   # visudo temp file
    assert not sudo_reads("rule~")                  # editor backup
    assert not sudo_reads("README.md")


def test_first_run_and_unchanged_restart_are_silent(host):
    etc, make, events, state = host
    w = make()
    assert w.start() == []
    assert w.check() == []
    assert make().start() == []
    assert events == []
    assert stat.S_IMODE(os.stat(state).st_mode) == 0o600


def test_new_nopasswd_rule_is_critical_and_reported_once(host, caplog):
    etc, make, events, _ = host
    w = make(); w.start()
    (etc / "sudoers.d" / "99-backdoor").write_text("ekco ALL=(ALL) NOPASSWD: ALL\n")
    with caplog.at_level(logging.WARNING, logger="security.privilege_watch"):
        changes = w.check()
        assert w.check() == []  # same state: no second alert
    assert [c[1] for c in changes] == ["created"]
    assert _types(events) == ["privilege_config_critical"]
    assert events[0]["disposition"] == "observed"
    assert "new passwordless sudo rule" in caplog.text
    assert "NOPASSWD: ALL" in caplog.text


def test_commenting_out_a_rule_is_an_ordinary_change(host):
    etc, make, events, _ = host
    (etc / "sudoers.d" / "10-tools").write_text("ekco ALL=(ALL) NOPASSWD: /usr/bin/journalctl\n")
    w = make(); w.start()
    (etc / "sudoers.d" / "10-tools").write_text("# disabled: ekco ALL=(ALL) NOPASSWD: /usr/bin/journalctl\n")
    w.check()
    assert _types(events) == ["privilege_config_changed"]


def test_visudo_temp_files_are_ignored(host):
    etc, make, events, _ = host
    w = make(); w.start()
    (etc / "sudoers.d" / "50-rule.tmp").write_text("ekco ALL=(ALL) NOPASSWD: ALL\n")
    assert w.check() == []
    assert events == []


def test_change_while_stopped_is_reported_at_next_start(host, caplog):
    etc, make, events, _ = host
    make().start()
    (etc / "passwd").write_text(PASSWD + "toor:x:0:0::/root:/bin/bash\n")
    with caplog.at_level(logging.WARNING, logger="security.privilege_watch"):
        changes = make().start()
    assert changes and changes[0][0].endswith("/passwd")
    assert _types(events) == ["privilege_config_critical"]
    assert "while stopped" in events[0]["actor"]
    assert "new uid-0 account: toor" in caplog.text


def test_ordinary_system_user_is_not_critical(host):
    etc, make, events, _ = host
    w = make(); w.start()
    (etc / "passwd").write_text(PASSWD + "ollama:x:964:964::/var/lib/ollama:/usr/bin/nologin\n")
    w.check()
    assert _types(events) == ["privilege_config_changed"]


def test_user_added_to_wheel_is_critical(host, caplog):
    etc, make, events, _ = host
    w = make(); w.start()
    (etc / "group").write_text("root:x:0:root\nwheel:x:998:ekco\n")
    with caplog.at_level(logging.WARNING, logger="security.privilege_watch"):
        w.check()
    assert _types(events) == ["privilege_config_critical"]
    assert "privileged group wheel members now: ekco" in caplog.text


def test_ld_so_preload_appearing_is_critical(host):
    etc, make, events, _ = host
    w = make(); w.start()
    (etc / "ld.so.preload").write_text("/usr/lib/libevil.so\n")
    changes = w.check()
    assert changes[0][1] == "created"
    assert _types(events) == ["privilege_config_critical"]


def test_replacing_a_file_with_a_symlink_is_a_change(host, tmp_path):
    etc, make, events, _ = host
    w = make(); w.start()
    target = tmp_path / "elsewhere"
    target.write_text(SUDOERS)  # same content, different object
    (etc / "sudoers").unlink()
    (etc / "sudoers").symlink_to(target)
    changes = w.check()
    assert changes and changes[0][1] == "modified"


def test_corrupt_saved_state_starts_fresh_without_raising(host):
    etc, make, events, state = host
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text("{not json")
    assert make().start() == []
    assert events == []


def test_watch_never_triggers_lockdown_and_never_breaks_the_detector(monkeypatch, host):
    """Wired into TamperDetector: a critical change records, never locks down."""
    from security import tamper_detection as td
    import security.policy_gate as pg

    lockdowns = []
    monkeypatch.setattr(pg, "trigger_lockdown", lambda reason: lockdowns.append(reason))
    monkeypatch.setattr(td, "log_security_event", lambda **kw: None)
    etc, make, events, _ = host
    detector = td.TamperDetector()
    detector._baselines = {}
    detector.privilege_watch = make()
    detector.privilege_watch.start()
    (etc / "sudoers.d" / "99-backdoor").write_text("ekco ALL=(ALL) NOPASSWD: ALL\n")
    detector.check_integrity()
    assert _types(events) == ["privilege_config_critical"]
    assert lockdowns == []

    class Broken:
        def check(self):
            raise RuntimeError("boom")

    detector.privilege_watch = Broken()
    detector.check_integrity()  # must not raise


def test_start_failure_is_contained(monkeypatch):
    from security import tamper_detection as td
    import security.privilege_watch as pw

    def boom(*a, **k):
        raise RuntimeError("cannot start")

    monkeypatch.setattr(pw, "PrivilegeConfigWatcher", boom)
    detector = td.TamperDetector()
    detector._start_privilege_watch()  # must not raise
    assert detector.privilege_watch is None


def test_unreadable_file_is_labelled_and_warned(host, caplog):
    etc, make, events, _ = host
    if os.geteuid() == 0:
        pytest.skip("root can read mode-000 files")
    os.chmod(etc / "sudoers", 0)
    try:
        w = make()
        with caplog.at_level(logging.WARNING, logger="security.privilege_watch"):
            w.start()
        assert w.snapshot()[str(etc / "sudoers")]["sha256"].startswith("unreadable:")
        assert "cannot read" in caplog.text
    finally:
        os.chmod(etc / "sudoers", 0o644)


def test_hud_feed_names_the_changed_file(tmp_path):
    """The HUD's event feed shows the path for privilege-watch events, as ERROR when critical."""
    import sqlite3
    from kaia_dashboard import AuditLogCollector

    db = tmp_path / "security_events.db"
    conn = sqlite3.connect(db)
    conn.execute("""CREATE TABLE security_events (event_id TEXT PRIMARY KEY, timestamp TEXT NOT NULL,
        type TEXT NOT NULL, source TEXT NOT NULL, actor TEXT NOT NULL, payload_hash TEXT,
        disposition TEXT NOT NULL, session_id TEXT NOT NULL)""")
    conn.executemany("INSERT INTO security_events VALUES (?,?,?,?,?,?,?,?)", [
        ("e1", "2026-11-01T17:00:00Z", "privilege_config_critical", "privilege_watch",
         "/etc/sudoers.d/99-backdoor (created)", "abc", "observed", "system_protection"),
        ("e2", "2026-11-01T17:00:01Z", "privilege_config_changed", "privilege_watch",
         "/etc/passwd (modified)", "def", "observed", "system_protection"),
    ])
    conn.commit(); conn.close()

    collector = AuditLogCollector.__new__(AuditLogCollector)
    collector._security_db_path = str(db)
    collector._last_seen_rowid = 0
    entries = collector._poll_security_db()
    assert "/etc/sudoers.d/99-backdoor (created)" in entries[0].message
    assert entries[0].level == "ERROR"
    assert "/etc/passwd (modified)" in entries[1].message
    assert entries[1].level == "WARN"
