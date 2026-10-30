# Changelog

## Unreleased

### Security

- Enforce timezone-aware, signed capability-token issue and expiry timestamps; reject future-issued, expired, non-positive and over-one-hour tokens, and reject malformed signatures or non-string scope fields.
- Preserve the existing capability-token JSON fields and HMAC format. Durable replay protection is still outstanding: valid bearer tokens remain reusable until expiry.
