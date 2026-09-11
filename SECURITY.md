# Security Policy

## Supported Versions

`methodos` is currently in 0.x release. Per the SemVer spec, breaking
changes may occur between minor releases until 1.0. Security fixes are
back-ported to the latest minor release only.

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

## Reporting a Vulnerability

Please **do not** open a public GitHub issue for security-sensitive
reports. Instead, use one of the following private channels:

1. **GitHub private vulnerability reporting** (preferred):
   https://github.com/<owner>/pgraph/security/advisories/new
2. **Email**: `security@<project-domain>` (see repo metadata for the
   current address)

A maintainer will acknowledge receipt within 3 business days and aim to
provide an initial triage within 7 days. We'll coordinate disclosure
timing with you; please give us a reasonable window (typically 90 days)
before any public write-up.

## What to Include

Help us triage efficiently by including:

- A clear description of the vulnerability and its impact
- Reproduction steps or a minimal proof-of-concept
- Affected versions (commit SHA or tag)
- Any known mitigations or workarounds you've identified

## Hardening Posture (0.1.0)

The hosted service ships with these defaults that operators should review
before exposing it to untrusted networks:

- **Auth off by default** — set `PGRAPH_API_KEY` to a high-entropy value
  (>= 32 random bytes, base64- or hex-encoded) to require bearer tokens
  on all `/v1/*` routes.
- **Rate limiter on by default** — defaults: 60 tokens per bucket,
  1 token/sec refill. Set `PGRAPH_RATE_LIMIT_DISABLED=1` to disable,
  or scale `PGRAPH_RATE_LIMIT_CAPACITY` for higher throughput.
- **No secrets in logs** — structured JSON logs scrub reserved
  `LogRecord` fields; do not pass secrets via `extra={...}`.
- **TLS termination** is the operator's responsibility; the service
  speaks plain HTTP and the API key travels in headers.

## Threat Model (Summary)

Out of scope (caller's responsibility):

- Network-layer DoS — front the service with a WAF or load balancer.
- Secrets in environment variables — use a secret manager.
- Cluster-wide rate limiting — the bundled limiter is per-process.

In scope (we address):

- Constant-time API-key comparison (`hmac.compare_digest`).
- Sanitized 500 responses (no traceback in body).
- Input validation at the FastAPI boundary (Pydantic `extra="forbid"`,
  string length caps, numeric ranges).
- Safe logging (no `print`, JSON formatter escapes non-serializable
  values via `repr`).

## Acknowledgments

We gratefully acknowledge reporters who help improve the security of
`methodos`. With your consent, we'll credit you in the release notes for
the fix.
