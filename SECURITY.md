# Security policy

## Supported versions

Security fixes go to the latest release. This project is in beta (`0.x`), so
older releases are not patched.

## Reporting a vulnerability

Please do not open a public issue. Report it privately through GitHub's
[private vulnerability reporting](https://github.com/sachncs/scmle/security/advisories/new)
or email sch.cs.96@gmail.com with the affected version, a description and, if
possible, a reproduction. Expect an acknowledgement within a week.

The threat model, controls and operator guidance are in
[`docs/SECURITY.md`](docs/SECURITY.md). The main point: model bundles are
pickle-based, so only load bundles you produced or whose signature you have
verified, and run the HTTP adapter behind an authenticating gateway.
