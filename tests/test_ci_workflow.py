"""Guard CI workflow action pins and the shared regression-suite contract."""

import re
from pathlib import Path


def test_security_workflow_uses_node24_actions_pinned_by_full_sha():
    workflow = Path(".github/workflows/security-regressions.yml").read_text()

    checkout = "3d3c42e5aac5ba805825da76410c181273ba90b1"
    setup_python = "5fda3b95a4ea91299a34e894583c3862153e4b97"

    assert f"actions/checkout@{checkout} # v7.0.1" in workflow
    assert f"actions/setup-python@{setup_python} # v7.0.0" in workflow
    assert not re.search(r"uses:\\s*actions/(?:checkout|setup-python)@v[0-9]+", workflow)
    assert 'python-version: "3.12"' in workflow
    assert "bash scripts/run_ci_tests.sh" in workflow
    assert "permissions:\\n  contents: read" in workflow
