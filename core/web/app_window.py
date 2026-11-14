"""The Kaia HUD as a desktop window, not a browser tab.

The same approach as KaiaRX and KaiaBooth (utils/infrastructure/system/
app_window.py there): a GTK window holding a WebKitGTK view of the loopback
HUD, with its own title, window class and icon, and no browser around it. No
address bar, no right-click menu, no developer tools, and no navigation away
from the HUD's own origin: anything else the page tries to open is refused.

Run by the *system* Python (``/usr/bin/python3``), which has GObject
introspection and WebKitGTK; the project's venv has neither. This file imports
nothing from the project so either interpreter can load it.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from typing import Optional
from urllib.parse import urlsplit

SYSTEM_PYTHON = "/usr/bin/python3"


def system_python() -> str:
    return SYSTEM_PYTHON if os.path.exists(SYSTEM_PYTHON) else (shutil.which("python3") or SYSTEM_PYTHON)


def available() -> bool:
    """Whether the system Python can open a WebKitGTK window (with an audio sink)."""
    try:
        out = subprocess.run([system_python(), "-c", "import gi; gi.require_version('Gtk', '3.0'); "
                              "gi.require_version('WebKit2', '4.1'); gi.require_version('Gst', '1.0'); "
                              "from gi.repository import Gtk, WebKit2, Gst; Gst.init(None); "
                              "import sys; sys.exit(0 if Gst.ElementFactory.find('autoaudiosink') else 3)"],
                             capture_output=True, timeout=20)
        return out.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def same_origin(url: str, home: str) -> bool:
    """Only the HUD's own scheme, host and port may load in the window."""
    a, b = urlsplit(url), urlsplit(home)
    return (a.scheme, a.hostname, a.port) == (b.scheme, b.hostname, b.port)


SESSION_KEY_ENV = "KAIA_HUD_SESSION_KEY"


def launch(url: str, title: str = "Kaia // Sentinel", wm_class: str = "KaiaSentinel",
           size: str = "1700x1080", icon: str = "", session_key: str = "") -> Optional[subprocess.Popen]:
    """Open ``url`` in its own window; None if this machine can't.

    ``session_key`` authorises the page's console commands. It travels in the
    environment, never in argv: /proc/<pid>/cmdline is readable by every user.
    """
    if not available():
        return None
    env = {**os.environ}
    env.pop(SESSION_KEY_ENV, None)
    if session_key:
        env[SESSION_KEY_ENV] = session_key
    try:
        return subprocess.Popen([system_python(), os.path.abspath(__file__), "--url", url, "--title", title,
                                 "--class", wm_class, "--size", size, "--icon", icon],
                                stdin=subprocess.DEVNULL, env=env)
    except OSError:
        return None


def _main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Open the local Kaia HUD in its own window.")
    ap.add_argument("--url", required=True)
    ap.add_argument("--title", default="Kaia // Sentinel")
    ap.add_argument("--class", dest="wm_class", default="KaiaSentinel")
    ap.add_argument("--size", default="1700x1080")
    ap.add_argument("--icon", default="")
    a = ap.parse_args()
    if urlsplit(a.url).hostname not in ("127.0.0.1", "localhost"):
        print("refusing a non-loopback HUD URL", file=sys.stderr)
        return 2

    # WebKitGTK's DMA-BUF renderer fails on this machine's NVIDIA driver under
    # Wayland ("Error 71 (Protocol error)") and the window closes at once.
    os.environ.setdefault("WEBKIT_DISABLE_DMABUF_RENDERER", "1")
    import gi
    gi.require_version("Gtk", "3.0")
    gi.require_version("WebKit2", "4.1")
    from gi.repository import GLib, Gtk, WebKit2

    GLib.set_prgname(a.wm_class)
    GLib.set_application_name(a.title)
    w, h = (int(x) for x in a.size.lower().split("x"))
    win = Gtk.Window(title=a.title)
    win.set_default_size(w, h)
    if a.icon:
        win.set_icon_name(a.icon)

    view = WebKit2.WebView()
    s = view.get_settings()
    s.set_enable_webaudio(True)
    s.set_media_playback_requires_user_gesture(False)
    s.set_enable_developer_extras(False)
    s.set_hardware_acceleration_policy(WebKit2.HardwareAccelerationPolicy.ALWAYS)
    s.set_enable_smooth_scrolling(True)
    s.set_javascript_can_open_windows_automatically(False)
    view.connect("context-menu", lambda *args: True)          # no browser menu

    def decide(_view, decision, kind):
        # Never leave the HUD: no other origin, no new windows, no downloads.
        if kind == WebKit2.PolicyDecisionType.NEW_WINDOW_ACTION:
            decision.ignore()
            return True
        if kind == WebKit2.PolicyDecisionType.NAVIGATION_ACTION:
            uri = decision.get_navigation_action().get_request().get_uri()
            if not same_origin(uri, a.url):
                decision.ignore()
                return True
        if kind == WebKit2.PolicyDecisionType.RESPONSE and not decision.is_mime_type_supported():
            decision.ignore()
            return True
        return False

    view.connect("decide-policy", decide)
    session_key = os.environ.pop(SESSION_KEY_ENV, "")
    view.load_uri(a.url + (f"#session={session_key}" if session_key and "#" not in a.url else ""))
    win.add(view)
    win.connect("destroy", Gtk.main_quit)
    win.show_all()
    Gtk.main()
    return 0


if __name__ == "__main__":
    sys.exit(_main())
