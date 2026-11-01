"""Every tests/test_*.py runs in CI unless it is a documented legacy exclusion.

tests/test_ebpf_ringbuf_drops.py (PR #95) was added without being listed in
scripts/run_ci_tests.sh, so CI went green without ever running it. New test
files must be listed there; legacy scripts that need a live host stay out on
purpose and are named here.
"""

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Need a running systemd host or populated runtime state; see CLAUDE.md.
LEGACY_EXCLUDED = {
    "tests/test_advanced_security.py",
    "tests/test_heuristics.py",
}


def test_every_test_file_is_in_the_ci_suite():
    runner = (REPO / "scripts" / "run_ci_tests.sh").read_text()
    missing = sorted(
        rel for rel in (p.relative_to(REPO).as_posix() for p in (REPO / "tests").glob("test_*.py"))
        if rel not in runner and rel not in LEGACY_EXCLUDED
    )
    assert not missing, f"add to scripts/run_ci_tests.sh (or LEGACY_EXCLUDED with a reason): {missing}"


def test_legacy_exclusions_still_exist():
    for rel in LEGACY_EXCLUDED:
        assert (REPO / rel).exists(), f"{rel} is gone; drop it from LEGACY_EXCLUDED"
