"""The HUD is a desktop application (GTK + WebKitGTK, as KaiaRX/KaiaBooth), the TUI is the default."""

import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "core" / "web"))
import app_window  # noqa: E402


def test_window_never_leaves_the_hud_origin():
    home = "http://127.0.0.1:37559/"
    assert app_window.same_origin("http://127.0.0.1:37559/static/js/app.js", home)
    for other in ("http://127.0.0.1:8080/", "https://127.0.0.1:37559/", "http://evil.example/",
                  "file:///etc/passwd", "http://localhost:37559/"):
        assert not app_window.same_origin(other, home), other


def test_window_refuses_a_non_loopback_url(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["app_window.py", "--url", "http://203.0.113.5:80/"])
    assert app_window._main() == 2


def _launcher():
    spec = importlib.util.spec_from_file_location("kaia_hud", ROOT / "kaia_hud.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_launcher_binds_loopback_and_reads_live_state(monkeypatch):
    m = _launcher()
    assert 1024 < m.free_loopback_port() < 65536
    monkeypatch.delenv("KAIA_STATE_DIR", raising=False)
    monkeypatch.setattr(m.os, "access", lambda p, mode: p == m.LIVE_STATE_DIR)
    assert m.bridge_env()["KAIA_STATE_DIR"] == "/var/lib/kaia"


def test_launcher_refuses_without_webkit_instead_of_opening_a_browser(monkeypatch, capsys):
    m = _launcher()
    monkeypatch.setattr(m.app_window, "available", lambda: False)
    assert m.main([]) == 2
    assert "WebKitGTK" in capsys.readouterr().err
    src = (ROOT / "kaia_hud.py").read_text() + (ROOT / "core/web/app_window.py").read_text()
    assert "chromium" not in src and "webbrowser" not in src and "--app=" not in src


def test_terminal_tui_is_the_default_entry_point():
    src = (ROOT / "kaia_dashboard.py").read_text()
    main = src[src.index("def main() -> None:"):]
    assert "if args.web:" in main and main.index("KaiamonUI()") > main.index("else:")
    assert os.access(ROOT / "kaia-hud", os.X_OK)


def test_avatar_rotates_through_the_shared_portraits():
    codec = (ROOT / "core/web/static/js/kaia/kaia-codec.js").read_text()
    assert "KaiaCodec.randomPortrait" in codec
    assert "|| 'random'" in (ROOT / "core/web/static/js/operative.js").read_text()
    for p in ("k28", "k59", "k60", "t1", "t2", "t3"):
        assert (ROOT / "core/web/static/assets/kaia" / p / "kaia_codec.webp").exists()
