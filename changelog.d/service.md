### Added

- `benchmarks/evidence/service.py` measures C7's three service clauses
  against the real `scmle serve` command in a child process over real HTTP
  (client side in `benchmarks/evidence/traffic.py`). Objectives: 1-record and
  100-record load profiles against the `docs/SERVICE_SLO.md` objectives, each
  until zero failures bound the failure rate below 0.001 at alpha (2995
  requests), the p95 decided by its distribution-free binomial upper bound;
  startup and resident memory checked too. Hung request: a `Stall` step hangs
  inside the model's predict for one request while load and slow-body clients
  run; every other request must succeed, then every worker is hung and
  `/livez` must fail so the platform restarts the replica, and SIGTERM must
  still end the process. SIGTERM: 29 rounds under load behind a router that
  stops routing on readiness, with `SCMLE_LINGER_SECONDS` at least the time
  the router takes to stop, plus one round with the product's default
  environment (no linger); zero resets, a router still routing at the
  signal and requests answered after it are required in every one of them.
  Three rounds of clients that ignore readiness are reported, not deciding.
  `--soak SECONDS` runs the
  soak for any duration (`python -m benchmarks.evidence.service --soak 86400
  --output soak.json`). Run it with `--write | --check`.
- `scmle serve` keeps serving out of rotation for `SCMLE_LINGER_SECONDS`
  (default 0: close at once) after SIGTERM (`/readyz` answers 503
  `draining`) before it closes the socket. Set it to the probe failure
  threshold times (period + timeout) to get the zero-reset drain, so callers
  the platform still routes there are answered instead of reset. A second
  stop signal ends the wait.

### Fixed

- A request whose scoring hangs no longer wedges a replica. The server kept
  refusing every request with 503 while one timed-out call was still running,
  and a call that never returned kept it unready and alive forever. Now the
  replica keeps serving on its other workers. It is unready only when no
  call can run (every worker holds a timed-out call, or a series model's lock
  is held), and `/livez` fails once that has lasted `STRIKES` times the scoring
  timeout.
- A hung scoring call no longer keeps a drained server process from exiting
  after SIGTERM: scoring runs on daemon threads (`workers.Crew`).
- A client that sends its body slowly no longer holds a scoring slot. The body
  is read before admission, and a body not received within the request I/O
  timeout gets a 408.
- A caller that reset its connection while sending the body no longer
  escapes the server's handling: it is logged and the connection dropped.
- Connections the kernel had queued when the server stopped were reset. The
  listener now answers the backlog before it closes. Its listen backlog is
  `socket.SOMAXCONN` instead of 5, so bursts of callers were refused or reset
  before.
