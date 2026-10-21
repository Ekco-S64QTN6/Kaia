import json
import os
import subprocess
import sys
from pathlib import Path


def test_system_state_mode_keeps_runtime_data_outside_read_only_checkout(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    env["KAIA_STATE_DIR"] = str(state_dir)
    env["KAIA_CAPABILITY_TOKEN_SECRET"] = "test-only-secret"
    env["PYTHONPATH"] = str(repo_root / "core")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import config, json; print(json.dumps({"
            "'state': config.STATE_DIR, 'storage': str(config.STORAGE_DIR), "
            "'security': str(config.SECURITY_STORAGE_DIR), "
            "'threat_intel': str(config.THREAT_INTEL_DIR), "
            "'system': config.STATE_DIR_IS_SYSTEM}))",
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(result.stdout.strip().splitlines()[-1])
    state = Path(data["state"])

    assert data["system"] is True
    assert Path(data["storage"]) == state
    assert Path(data["security"]) == state
    assert Path(data["threat_intel"]) == state / "threat_intel"
    assert (state / "threat_intel").is_dir()
