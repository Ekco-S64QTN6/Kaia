"""kaia-baseline.sh must write a new baseline when files changed.

Since e790749 added `set -euo pipefail`, the "changed:" display pipeline
(diff exits 1 on any difference) aborted the script after printing the diff and
before writing the baseline, so a changed tree could never be re-baselined and
the Policy Gate locked the host down on every start. This runs the script's own
diff block, under the script's own shell options, against differing baselines.
"""

import re
import subprocess
from pathlib import Path


def _diff_block() -> str:
    script = Path("scripts/kaia-baseline.sh").read_text()
    match = re.search(r'^if \[ -f "\$BASELINE" \]; then\n.*?^fi\n', script, re.S | re.M)
    assert match, "diff-display block not found in kaia-baseline.sh"
    return match.group(0)


def test_changed_baseline_does_not_abort_before_write(tmp_path):
    old = tmp_path / "baseline.sha256"
    new = tmp_path / "candidate"
    old.write_text("# header\naaaa  /opt/kaia/security/db.py\n")
    new.write_text("# header\nbbbb  /opt/kaia/security/db.py\n")

    program = (
        "set -euo pipefail\n"
        f'BASELINE="{old}"\ntmp="{new}"\nYELLOW=""\nNC=""\n'
        + _diff_block()
        + 'echo "REACHED_WRITE"\n'
    )
    result = subprocess.run(["bash", "-c", program], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert "changed: /opt/kaia/security/db.py" in result.stdout
    assert "REACHED_WRITE" in result.stdout
