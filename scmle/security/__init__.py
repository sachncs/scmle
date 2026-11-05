"""Authentication, authorization, limits and audit for the prediction service.

`tokens` verifies bearer tokens (JWT, HS256 and RS256) against a rotating key
set; `access` turns a verified token into an authorization decision (scopes,
tenant, rate limit) with errors that are safe to show a caller; `audit`
appends activation, access and administration rows to the hash-chained
deployment ledger.
"""
