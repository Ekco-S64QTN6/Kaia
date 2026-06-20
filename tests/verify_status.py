import sys
import os
os.environ.setdefault("KAIA_CAPABILITY_TOKEN_SECRET", "test_signing_secret_key_2026")
import pathlib
root_dir = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "core"))
from kaia_cli import KaiaCLI

cli = KaiaCLI()
status = cli.get_system_status()
print(cli.format_system_status_output(status))
