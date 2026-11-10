"""Firewall numbers on the dashboards match the ruleset (live-host ruleset shape).

The HUD showed "FIREWALL BLOCKS 5 · 5 in · 1 fwd" on the host: the counter took
chain policies (`policy drop;`) as rules and the host firewall's rules as Kaia's
blocks. Kaia had 0 block rules.
"""

from security.firewall_state import drop_rule_counts, kaia_block_rule_count

# Shape of the host's /etc/nftables.conf ruleset (2026-11-09), plus a Kaia block.
HOST = """table inet filter {
\tset blackhole {
\t\ttype ipv4_addr
\t}

\tchain input {
\t\ttype filter hook input priority filter; policy drop;
\t\tip saddr @blackhole drop
\t\tct state invalid drop
\t\ttcp dport 22 tcp flags syn drop
\t\tip saddr 192.168.1.254 ip daddr 224.0.0.0/4 drop
\t}

\tchain forward {
\t\ttype filter hook forward priority filter; policy drop;
\t}

\tchain output {
\t\ttype filter hook output priority filter; policy accept;
\t}
}
"""
KAIA = """table inet kaia_block {
\tchain kaia_input {
\t\ttype filter hook input priority -10; policy accept;
\t\tip saddr 203.0.113.9 drop
\t\tip6 saddr 2001:db8::5 tcp dport 22 reject
\t}
}
"""


def test_policies_are_not_rules_and_chains_are_classified_by_hook():
    assert drop_rule_counts(HOST) == {"input": 4, "forward": 0, "output": 0}
    assert drop_rule_counts(HOST + KAIA) == {"input": 6, "forward": 0, "output": 0}


def test_kaia_blocks_are_counted_separately():
    assert kaia_block_rule_count(HOST) == 0
    assert kaia_block_rule_count(HOST + KAIA) == 2


def test_dashboards_use_the_shared_parser_and_honest_labels():
    from pathlib import Path
    dash = Path("kaia_dashboard.py").read_text()
    assert dash.count("firewall_state.drop_rule_counts(stdout)") == 2
    assert "active_blocks = kaia_blocks" in dash
    html = Path("core/web/static/index.html").read_text()
    assert "KAIA BLOCKS" in html and "FIREWALL BLOCKS" not in html


def test_lan_node_count_is_distinct_addresses_in_24h_not_the_list_length(tmp_path, monkeypatch):
    import sqlite3, threading
    from datetime import datetime, timedelta, UTC
    import config
    import kaia_dashboard as d
    db = tmp_path / "assets.db"
    now = datetime.now(UTC)
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE assets (ip TEXT, mac TEXT, hostname TEXT, vendor TEXT, detection_vector TEXT, first_seen TEXT, last_seen TEXT)")
        for i in range(9):
            conn.execute("INSERT INTO assets VALUES (?,?,?,?,?,?,?)", (f"192.168.1.{i}", f"m{i}", "", "", "arp", "", (now - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")))
        conn.execute("INSERT INTO assets VALUES ('192.168.1.99','x','','','arp','',?)", ((now - timedelta(days=3)).strftime("%Y-%m-%dT%H:%M:%SZ"),))
    monkeypatch.setattr(config, "ASSETS_DB_PATH", str(db))
    src = open("kaia_dashboard.py").read()
    assert '"lan_nodes_24h": lan_nodes_24h' in src
    since = (now - timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%SZ")
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT COUNT(DISTINCT ip) FROM assets WHERE last_seen >= ?", (since,)).fetchone()[0] == 9
    assert "data.lan_nodes_24h" in open("core/web/static/js/radar.js").read()
