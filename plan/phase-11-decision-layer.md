# Phase 11: Decision layer (F11)

Names follow [`00-canon.md`](00-canon.md). Feature definition:
[`../product/features/F11-decision-layer.md`](../product/features/F11-decision-layer.md).

Built: the charter's decision block, `DecisionBudget`, thresholds with a
calibration gate, net benefit with a simultaneous band, off-policy
evaluation, fairness constraints, the anytime-valid engine, rollout design and
the promote, hold and rollback rule, the threshold and net benefit wiring in a
charter run, and certificates with pre-registered floors (`--check f11`).
`CHANGELOG.md` and `git log` hold the record. Below is what is left.

## Work still to do

**W1 Lifecycle hook** (one agent). Call `Rollout` and `RollbackMonitor` from
`LifecycleManager` (`scmle/governance/lifecycle.py`), make the bootstrap
promotion and rollback rules descriptive, keep independent approval, write the
assignments to the decision log. Done when: B9 runs through `LifecycleManager`
and existing lifecycle tests are unchanged where no charter rollout is set.

**W2 remainder** (one agent). The files and the one-time policy value and
fairness tests are written by a charter run. Left: experiment assignments in
the F3 ledger (no experiment runs inside a charter run) and `claims verify`
checking the four files.

**W3 Fairness integration** (one agent). Make `evaluation/fairness.py` and
`mitigation.py` clients of `decision.constraints`; stop their constants gating;
solve the `reweighing` mechanism; add a cluster-aware satisfaction test.

**W4 Referents** (one person outside the author). Obtain Criteo (after the
licence check) and the Open Bandit Dataset, add manifests with hash and licence
to `corpora.json`, run R `dcurves` or `rmda` and Open Bandit Pipeline on the
fixtures, compare. Reproduce A6, B1 and B7 independently. Phase 11 is not closed
until the reproduction exists.

**W5 Statistics** (one agent). Re-register and re-measure the net benefit band
(coverage miss at `n = 2000`, low power); read the cited confidence-sequence
papers and the e-LOND treatment of overlapping tests before enabling the
rejection boost; a rule for differential missingness; policy trees of depth
two; check the three incompatible fairness pairs against the sources.

## Exit gate still to pass

```bash
uv run python -m benchmarks.qualification --check f11
```

passes today with `A3` descriptive. The phase closes when W4's outside
reproduction is on file, W1 and W2 are done, and the A3 gate either passes a
re-registered floor or the band stays descriptive on purpose.

## Risks that remain

- Costs are inputs, not facts; the report shows the cost-ratio range over which
  the decision holds.
- The anytime-valid test is wider than a fixed-horizon test and its
  construction is the author's; its measured cost is in the certificates.
- Randomization is assumed; logs without recorded probabilities cannot give a
  `confirmed` decision claim.
- Fairness is contested; the instrument enforces only what the charter declares
  and the legal admissibility of group thresholds is an attestation.
- Dataset licences (Criteo non-commercial, Hillstrom unstated) may force the
  referents to be replaced; if so the plan says so openly.

## Out of scope

Adaptive allocation and bandit control, Bayesian decision rules without
frequentist error guarantees, interference-aware designs, reinforcement
learning evaluation, dynamic treatment regimes, learning of fairness criteria,
estimating causal effects from observational logs (F10), data shift (F6), the
HTML report.
