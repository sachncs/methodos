# Security policy

## Supported versions

Only the latest `master` revision and the latest published release receive
security fixes. Deployments should consume an immutable signed container digest
or a pinned Python artifact rather than a mutable `latest` tag.

## Reporting a vulnerability

Do not open a public issue for an exploitable vulnerability. Use GitHub's private
security advisory flow for this repository. Include the affected revision,
impact, reproduction steps, and any suggested mitigation. Remove credentials,
personal data, and production identifiers from the report.

We will acknowledge a valid report, assess severity and affected versions, and
coordinate a fix and disclosure timeline with the reporter.

## Deployment expectations

Production and staging must use authentication, durable persistence, explicit
roles, TLS at the perimeter, secret storage outside source control, and the
backup, restore, and signature-verification procedures in `docs/operations.md`.
