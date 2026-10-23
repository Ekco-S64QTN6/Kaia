# Security change review and verification process

This process is intended to match Kaia's runtime principle: untrusted or automated output is not itself proof of correctness. A well-written audit narrative is not a substitute for evidence.

## Scope requiring independent review

Treat a change as security-critical if it touches any of the following, even when the edit appears small:

- `security/**`, especially `security/policy_gate.py`, `security/host_executor.py`, schemas, telemetry, FIM, and tamper detection.
- `core/config.py`, the web bridge's authorization or IPC boundary, capability-token/signing logic, or the audit ledger.
- Firewall containment/recovery scripts, trusted-baseline handling, installer scripts, and `scripts/kaia-lockdown.sh` or `scripts/kaia-unlock.sh`.
- systemd service units, sandbox/resource limits, privileged file permissions, dependency pins, GitHub Actions workflows, or CI test selection.

**Expected gate:** security-critical changes should receive at least one approval from a reviewer who is not the PR author. Reviewers should inspect the diff and the evidence, not approve based only on a generated summary. Review threads should be resolved, and a new security-relevant push should be reviewed before merge.

A PR template cannot enforce approvals on its own. Repository maintainers must configure the `main` ruleset/branch protection to require an approving review and, where appropriate, code-owner review and resolution of review threads. Verify the active settings directly in GitHub; do not assume they are enforced because this document exists.

## Required evidence for a security fix

1. **Describe the finding and threat model.** Identify the trust boundary, the attacker's capability, the vulnerable behavior, and the expected security invariant.
2. **Write a targeted regression.** Prefer a test that demonstrates the vulnerable behavior. Where practical, show that it fails on the parent commit and passes on the fix. If a pre-fix failure is impractical, explain why.
3. **Record actual verification.** Link the CI run for the exact final head SHA. Name any other commands and summarize their result. A successful compile is not a passing test suite.
4. **State what was not verified.** Distinguish automated tests from manual code inspection and from live-host integration. Do not imply that a test ran when it did not.
5. **Explain residual risk.** Note compatibility changes, bounded/omitted output, existing on-disk data that is not migrated, and any required operator follow-up.
6. **Review the final commit.** The reviewer should inspect the exact pushed diff and re-check after later changes. Resolve applicable review threads before merge.

## AI-assisted development provenance

For AI-assisted work, the PR should record the tool/model and version when known, a link to the issue/task or a concise prompt summary, and the human who verified the patch. Do not paste secrets, credentials, private data, or sensitive prompts into a PR. Record what was independently checked; do not attribute a finding to a model unless that provenance is actually known.

Separate the **audit finding** from the **verification claim**. “This code appears to be vulnerable” is a finding; “the regression reproduces it and passes after the fix” is a verification claim requiring evidence.

## Host-dependent security controls

The deterministic regression suite does not prove that all privileged or kernel-dependent controls work on a target installation. Unless directly tested, record these as unverified:

- eBPF/BCC event delivery and kernel-version compatibility.
- fanotify/FIM permission responses and event-descriptor lifecycle.
- live nftables rule installation, lockdown, snapshot validation, and restoration.
- systemd user/system services, cgroup resource limits, Bubblewrap, and process-group termination.
- installed state/config ownership, permissions, symlink resistance, and root-to-service-user access.

For a host-integration result, record the distribution, kernel and systemd versions, the exact command or harness, expected result, observed result, and relevant logs with credentials and sensitive host data redacted. If no supported target host was available, say so explicitly.

## Provenance and traceability

Link follow-up PRs to the finding or prior fix that prompted them. When a later change corrects an incomplete earlier fix, state which residual failure remained and how the new regression covers it. Avoid describing multiple iterative patches as independent findings when they address the same underlying issue.
