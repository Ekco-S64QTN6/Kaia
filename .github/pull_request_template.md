## Change category and risk

- [ ] Security-critical: changes authorization, capability tokens, audit logs, host execution, firewall/lockdown, filesystem integrity, secrets, privileged services, or CI supply-chain controls.
- [ ] Runtime behavior / bug fix
- [ ] Tests only
- [ ] Documentation / process only

## Finding and threat model

**Observed behavior / evidence:**

**Trust boundary and attacker capability:**

**Impact if left unfixed:**

**Why this patch addresses the failure mode:**

## AI-assistance provenance

- **Author/operator:**
- **AI tool and model/version (if used):**
- **Task, issue, or prompt reference:** (Link or concise summary; do not paste credentials or sensitive prompts.)
- **Human verifier:** (Who inspected the actual diff and test evidence?)
- **Independent reviewer:** (For security-critical changes, this must be someone other than the PR author.)

## Regression evidence

- **Regression test that fails before the fix:** (Test name and parent-commit run, or explain why a pre-fix failure is not applicable.)
- **Passing test evidence:** (Exact CI run or command and result.)
- **Other tests run:**
- **Tests not run / host-dependent checks:**

## Verification checklist

- [ ] The change was reviewed against the threat model, not just the PR description.
- [ ] The regression exercises the real failure mode rather than only checking source-code text.
- [ ] Security-relevant failure paths fail closed.
- [ ] Output, input, storage, and time bounds are addressed where applicable.
- [ ] CI/test claims link to the actual run; unrun tests are explicitly called out.
- [ ] Host-dependent behavior is marked as unverified unless exercised on a target host.
- [ ] Residual risk and compatibility consequences are documented.
- [ ] Review threads are resolved and the final pushed commit was reviewed.
