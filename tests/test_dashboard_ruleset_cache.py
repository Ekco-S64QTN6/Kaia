"""The dashboard asks the gate for the ruleset at most once per cache period.

Every gate query is an audited request. Two collectors polling `nft_list`
every 5 and 10 s wrote ~26k ledger rows a day while a dashboard was open (#105).
"""

import os
from pathlib import Path

os.environ.setdefault("KAIA_CAPABILITY_TOKEN_SECRET", "ci-test-secret")

import kaia_dashboard as d


def _reset():
    d._ruleset_cache.update(at=None, success=False, stdout="")


def test_both_collectors_share_one_query_per_period():
    _reset()
    calls = []

    def query(kind):
        calls.append(kind)
        return True, "table inet kaia_lockdown {"

    assert d.query_ruleset(now=100.0, query=query) == (True, "table inet kaia_lockdown {")
    assert d.query_ruleset(now=105.0, query=query)[0] is True
    assert d.query_ruleset(now=109.9, query=query)[0] is True
    assert calls == ["nft_list"]
    d.query_ruleset(now=110.0, query=query)
    assert calls == ["nft_list", "nft_list"]


def test_a_failed_query_is_cached_too_and_retried_after_the_period():
    _reset()
    results = iter([(False, ""), (True, "ok")])
    query = lambda kind: next(results)
    assert d.query_ruleset(now=0.0, query=query) == (False, "")
    assert d.query_ruleset(now=5.0, query=query) == (False, "")   # no hammering a down gate
    assert d.query_ruleset(now=10.0, query=query) == (True, "ok")


def test_clock_going_backwards_refreshes():
    _reset()
    calls = []
    q = lambda kind: (calls.append(1), (True, ""))[1]
    d.query_ruleset(now=50.0, query=q)
    d.query_ruleset(now=40.0, query=q)
    assert len(calls) == 2


def test_collectors_use_the_shared_query():
    src = Path("kaia_dashboard.py").read_text()
    assert src.count('query_gate_diagnostics("nft_list")') == 0
    assert src.count("query_ruleset()") == 2
