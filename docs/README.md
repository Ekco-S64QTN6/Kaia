# Documentation Map and Source of Truth

**Last reconciled:** 2026-10-22  
**Repository:** `Ekco-S64QTN6/Kaia`

This page defines which documents are authoritative and how to interpret implementation claims.

## Current documents

- **[README.md](../README.md)** — user-facing overview, prerequisites, deployment shape, current feature summary and operator cautions.
- **[master_plan.md](master_plan.md)** — authoritative target architecture, invariants, component design, roadmap and the current repository implementation/verification matrix in §10.5. Target requirements must not be confused with implemented code.
- **[NEXT_STEPS.md](NEXT_STEPS.md)** — deployment and privileged acceptance tests. Its historical host observations are dated; verify actual unit state, paths, capabilities and trusted baseline before operating.
- **[03_Gemini35_Execution_Report.md](03_Gemini35_Execution_Report.md)** — historical report from 2026-07-09. Its host-state assertions and test counts are not current evidence.
- **[Kaia UI handoff](../Kaia_UI_Upgrade/claude_handoff.md)** — dated UI integration handoff, reconciled against the 2026-10-22 code review.
- **[UI design specification](../Kaia_UI_Upgrade/2026-10-07-kaia-hardened-security-ui-upgrade-design-spec.md)** — original 2026-10-07 design proposal; its schedule and initial-state descriptions are historical, while its status note points to the current implementation record.

## Historical source material

- **[`docs/archived/`](archived/)** contains prior plans, code reviews, issue lists and reference snapshots. These files are retained for provenance; statements inside them describe the repository as it was when written. Do not use them to infer current implementation or host state without checking the current source.
- **[`core/data/`](../core/data/)** contains reference knowledge and operator cheat sheets, not the architecture source of truth. Validate security-sensitive commands against the current scripts and service units before use.

## Evidence levels

When describing a feature, use these distinct labels:

1. **Implemented in source** — code is present.
2. **Automated-test covered** — a named unit/regression test exercises the behavior.
3. **CI-passing** — the exact commit has a successful Actions run; report the run and test count.
4. **Target-host verified** — the installed Arch Linux unit, kernel hooks, permissions, firewall manager and reboot behavior were directly exercised and recorded.

Evidence at one level does not imply the next. In particular, notification-mode fanotify is not execution-blocking FIM, a syscall-entry event is not proof an operation succeeded, and unit tests do not prove real nftables/systemd behavior.

## Maintenance rule

When code or a subsystem contract changes, update the README summary if user-visible behavior changes, `master_plan.md` §10.5 and the relevant architecture section, and this checklist if a target-host command or acceptance criterion changes. Mark dated reports as historical instead of silently rewriting what they originally observed. Never claim host installation, baseline trust or privileged behavior without fresh host evidence.
