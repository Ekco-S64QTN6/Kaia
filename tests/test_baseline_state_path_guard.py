"""Regression guards for trusted baseline state-directory containment."""

from pathlib import Path


def test_baseline_script_rejects_state_paths_overlapping_project_checkout():
    script = Path("scripts/kaia-baseline.sh").read_text(encoding="utf-8")

    assert 'case "$PROJECT_DIR/" in' in script
    assert 'case "$STATE_DIR/" in' in script
    assert 'KAIA_STATE_DIR must not contain the project checkout' in script
    assert 'KAIA_STATE_DIR must not be inside the project checkout' in script
    assert 'STATE_DIR="$(realpath -m -- "$KAIA_STATE_DIR")"' in script
