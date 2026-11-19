### Added

- `benchmarks/evidence/protocols.py` measures C7's "lease and journal
  protocols model-checked" clause. It holds executable models of three
  protocols: the workflow run lease with fencing tokens, the custody journal
  and the erasure journal. A small explicit-state model checker explores each
  model breadth first over every interleaving, with a crash allowed before any
  step. The lease is checked with distinct owner names and with the shared
  default owner name. The journals are checked under process kills and under
  power loss, where a line can be cut. The checker reports the shortest
  counterexample of each failing invariant. `Link` ties each model action to
  the function it models by text that function must contain. A gated object
  store drives the real implementation through every counterexample and 20
  sample paths per model, one store or file call per step. Each journal's
  client resumes as its real caller does: `Erasure.execute` skips every step
  the journal shows, so a step recorded twice breaks the `single` invariant;
  the orchestrator's gate review decides every gate again on resume, so
  repeated custody verdicts are by design and are outside the model.
  After each step it
  compares the real state with the model's. Run it with
  `python -m benchmarks.evidence.protocols --show | --write | --check`.
  The clause fails on these counterexamples, which the real code reproduces:
  - Fence reuse in the run lease. `Leases.relinquish` deletes the lease, so the
    next `acquire` takes fence 1 again. After a takeover at fence 2, the run's
    records carry fence 2, and every later valid holder is refused
    (`EvictionError`) for as long as the run lasts. With the default owner name
    (`Executor(owner="executor")`), one executor's `relinquish` can delete
    another's fresh lease, which has the same content and so the same store
    version. Two holders can then write the run record interleaved.
  - The custody journal's reader
    (`scmle.assurance.custody.journal.Journal.read`) cannot read a journal
    whose last line was cut by a power loss. It raises a raw
    `json.JSONDecodeError`, so the run's custody record, and with it the run,
    cannot be resumed. The erasure journal survives both crash kinds.
