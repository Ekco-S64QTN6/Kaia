#!/usr/bin/env python3
"""KAIA // SENTINEL: the HUD as a desktop application.

Starts the loopback web bridge on a free port, opens it in its own GTK window
(core/web/app_window.py, the KaiaRX/KaiaBooth approach: WebKitGTK, no browser),
and stops the bridge when the window closes. The terminal dashboard stays
separate: ``kaia_dashboard.py`` (curses), started by scripts/activate_kaia_env.sh.

    ./kaia-hud                     # or: .venv/bin/python kaia_hud.py
    ./kaia-hud --install-desktop   # add "Kaia Sentinel" to the application menu
"""

import argparse
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "core", "web"))
import app_window  # noqa: E402  (standalone module, no project imports)

LIVE_STATE_DIR = "/var/lib/kaia"


def free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def bridge_env() -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{ROOT}:{os.path.join(ROOT, 'core')}"
    # Read the deployed gate's published state (group kaiacord) when it is there.
    if "KAIA_STATE_DIR" not in env and os.access(LIVE_STATE_DIR, os.R_OK | os.X_OK):
        env["KAIA_STATE_DIR"] = LIVE_STATE_DIR
    return env


def wait_healthy(url: str, timeout: float = 30.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url + "/api/health", timeout=2) as r:
                if r.status == 200:
                    return True
        except OSError:
            time.sleep(0.3)
    return False


def install_desktop_entry() -> str:
    apps = os.path.expanduser("~/.local/share/applications")
    os.makedirs(apps, exist_ok=True)
    path = os.path.join(apps, "kaia-sentinel.desktop")
    launcher = os.path.join(ROOT, "kaia-hud")
    with open(path, "w") as f:
        f.write("[Desktop Entry]\nType=Application\nName=Kaia Sentinel\n"
                "Comment=Kaia security HUD (loopback, read-mostly)\n"
                f"Exec={launcher}\nIcon=security-high\nTerminal=false\n"
                "Categories=System;Security;Monitor;\nStartupWMClass=KaiaSentinel\n")
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--install-desktop", action="store_true", help="add a launcher to the application menu")
    ap.add_argument("--face", help="pin one portrait (k28, k59, k60, t1, t2, t3); default rotates")
    args = ap.parse_args(argv)
    if args.install_desktop:
        print("installed", install_desktop_entry())
        return 0
    if not app_window.available():
        print("This machine cannot open a WebKitGTK window: install webkit2gtk-4.1, python-gobject "
              "and gst-plugins-good (pacman -S webkit2gtk-4.1 python-gobject gst-plugins-good).",
              file=sys.stderr)
        return 2

    port = free_loopback_port()
    url = f"http://127.0.0.1:{port}"
    python = os.path.join(ROOT, ".venv", "bin", "python")
    bridge = subprocess.Popen([python if os.path.exists(python) else sys.executable,
                               os.path.join(ROOT, "core", "web_bridge.py"), "--port", str(port)],
                              cwd=ROOT, env=bridge_env(), stdin=subprocess.DEVNULL,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        if not wait_healthy(url):
            print("the HUD bridge did not start; run core/web_bridge.py by hand to see why", file=sys.stderr)
            return 1
        window = app_window.launch(url + (f"/?face={args.face}" if args.face else "/"))
        if window is None:
            return 2
        signal.signal(signal.SIGTERM, lambda *a: window.terminate())
        return window.wait() or 0
    except KeyboardInterrupt:
        return 130
    finally:
        bridge.terminate()
        try:
            bridge.wait(timeout=5)
        except subprocess.TimeoutExpired:
            bridge.kill()


if __name__ == "__main__":
    sys.exit(main())
