# Documentation map

Which document is authoritative for what.

| Document | Authority |
|:--|:--|
| [README](../README.md) | The overview: what Kaia is, what is built, how to start |
| [master_plan.md](master_plan.md) | The architecture: axioms, invariants, component design, and the implementation status in §10.5 |
| [ROADMAP.md](ROADMAP.md) | What is left to build or verify |
| [OPERATIONS.md](OPERATIONS.md) | Running, deploying and recovering Kaia on a host, including capability-token custody and rotation |
| [DEPLOYMENT_CHECKLIST.md](DEPLOYMENT_CHECKLIST.md) | Privileged acceptance tests for a deployed host; its recorded host observations are dated |
| [security_review_process.md](security_review_process.md) | What a security change needs before it merges |
| [design/](design/) | Visual references for the operator HUD and the Kaia avatar |
| [`core/data/`](../core/data/) | Reference knowledge and cheat sheets that Kaia retrieves from — not the architecture. Check security-sensitive commands against the current scripts before use. |

Earlier plans, reviews and agent handoffs were retired once their open items were in the roadmap;
they remain in Git history.

## Evidence levels

When describing a feature, use these distinct labels:

1. **Implemented in source** — code is present.
2. **Automated-test covered** — a named unit or regression test exercises the behaviour.
3. **CI-passing** — the exact commit has a successful Actions run.
4. **Target-host verified** — the installed Arch Linux unit, kernel hooks, permissions, firewall
   manager and reboot behaviour were exercised directly and recorded.

Evidence at one level does not imply the next. Notification-mode fanotify is not execution-blocking
FIM, a syscall-entry event is not proof an operation succeeded, and unit tests do not prove real
nftables or systemd behaviour.

## Maintenance rule

When code or a subsystem contract changes: update the README if user-visible behaviour changes,
`master_plan.md` §10.5 and the relevant architecture section, and the roadmap. Update the
deployment checklist if a target-host command or acceptance criterion changes. Never claim host
installation, baseline trust or privileged behaviour without fresh host evidence.

## Regression suite

`python -m pip install -r requirements-test.txt`, then `bash scripts/run_ci_tests.sh`. GitHub
Actions calls the same runner, so local and CI membership stay in step; workflow actions are pinned
to full commit SHAs. It is a focused, deterministic suite, not every file under `tests/`: legacy
scripts such as `verify_security.py` are manual integration tools, and `test_advanced_security.py`
needs host and database isolation before it is safe for CI.
