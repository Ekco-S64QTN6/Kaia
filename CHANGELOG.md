# Changelog

## Unreleased

### Network discovery

- Parse NBNS NB and NBSTAT answer records, including scoped/compressed owner names and bounded active/unique node-name extraction; reject truncated or malformed records without partial persistence.

### Security

- Enforce timezone-aware, signed capability-token issue and expiry timestamps; reject future-issued, expired, non-positive and over-one-hour tokens, reject malformed signatures, non-string scope/timestamp fields, and excessively nested JSON.
- Preserve the existing capability-token JSON fields and HMAC format. Enforce durable single-use token consumption with an atomic SQLite fingerprint registry; fail closed when the registry cannot be updated. Raw tokens and signatures are not persisted.
- Extend synthetic Policy Gate IPC coverage to fragmented headers/payloads and truncated-frame EOF, asserting incomplete requests close without reaching policy evaluation.
