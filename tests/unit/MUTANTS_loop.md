# Mutation results: the closed loop (`scmle/monitoring/`, `scmle/governance/`)

mutmut 3.8.0 on Python 3.14 (free-threaded build), run on 2026-11-05 in copies
of the working tree (`also_copy` as in `setup.cfg`, `timeout_multiplier = 2`,
`--max-children 4`; the machine was shared and heavily loaded). Score is
`killed / (killed + survived + timeout)`, as in
`scripts/check_mutation_score.sh`. "Not killed" is survived plus timeout.

## Monitoring (`sequential_tests.py`, `labels.py`)

Tests: `test_sequential_tests.py`, `test_label_mechanism.py` and the added
`test_monitoring_mutants.py` (golden numbers from the committed code on fixed
sequences, exact error messages, boundary values, state round trips).

Before the added tests:

| Module | Killed | Survived | Timeout | Score |
| --- | --- | --- | --- | --- |
| `monitoring/labels.py` | 224 | 25 | 0 | 90.0% |
| `monitoring/sequential_tests.py` | 587 | 30 | 0 | 95.1% |
| Total | 811 | 55 | 0 | 93.6% |

After:

| Module | Killed | Survived | Timeout | Score |
| --- | --- | --- | --- | --- |
| `monitoring/labels.py` | 236 | 13 | 0 | 94.8% |
| `monitoring/sequential_tests.py` | 612 | 5 | 0 | 99.2% |
| Total | 848 | 18 | 0 | 97.9% |

The 18 survivors were read one by one. Not killed and equivalent:

| Mutant | Change | Why it cannot change behaviour |
| --- | --- | --- |
| `MixtureTest.update` `strict=` (3 mutants) | `zip(log_weights, inner, strict=True)` with `None`, no argument or `False` | The two lists always have the same length (a term and its weight are appended and dropped together). |
| `MixtureTest.update` `axis=None` | `stacked.max(axis=None)` for `max(axis=0)` | Only a shift of the log-sum-exp, which does not change the value except by rounding (checked to 1e-12 on array nulls); the shift is there for underflow. |
| `PairedEProcess.__init__` | `scale <= 0.0` to `scale < 0.0` | With `scale = 0` the second test `-scale < margin < scale` is already false, so the same error is raised. |
| `estimate_batch` | `np.where(seen, losses, 1.0)` for `0.0` | The result is multiplied by `seen` on the next operation, so the value for unobserved cases is never used. |
| `estimate_batch` | `min(mean / scale, 2.0)` for `1.0` | With at least one unobserved case (otherwise the batch is `complete`) and propensities at least `epsilon`, `mean / scale <= (n-1)/n`; the clip is a guard against rounding. |
| `stream_mechanism` | `>` to `>=` | Replacing the weakest mechanism by an equal one changes nothing. |

Left unclassified: 10 `dtype=` or `dtype` argument mutants of `np.asarray`
for inputs the API documents as floats or booleans; the cases where the
conversion matters (integer lists, `Fraction`) are pinned, the others were
not shown equivalent one by one.

## Governance (`deployment_ledger.py`, `escalation.py`, `never_list.py`, `loop.py`)

Tests: `test_deployment_ledger.py`, `test_escalation.py`, `test_loop.py`,
`test_loop_streams.py`, `test_loop_wiring.py` and the added
`test_loop_golden.py` and `test_escalation_golden.py` (fixtures
`tests/fixtures/loop_golden.json` and `escalation_golden.json`). The golden
loop test replays ten fixed scenarios (stable, abrupt rise with retrain,
uncertified alarm, promotion in order, rollback, halt and recalibration,
routing with novelty and coverage, unverified labels, replies and
amendments, feedback test) and compares every ledger row (rounded, without
hashes), the saved state and the artifacts. No run without the golden tests
was kept (it was stopped when they were added), so there is no before figure.

| Module | Killed | Survived | Timeout | Score |
| --- | --- | --- | --- | --- |
| `governance/deployment_ledger.py` | 360 | 32 | 9 | 89.8% |
| `governance/escalation.py` | 407 | 75 | 0 | 84.4% |
| `governance/loop.py` | 1576 | 264 | 18 | 84.8% |
| `governance/never_list.py` | 100 | 17 | 1 | 84.7% |
| Total | 2443 | 388 | 28 | 85.4% |

| Module | Not killed | message/string | numpy/zip keyword | replaced by None | comparison boundary | arithmetic operator | other |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `governance/deployment_ledger.py` | 41 | 14 | 0 | 13 | 1 | 0 | 13 |
| `governance/escalation.py` | 75 | 19 | 20 | 0 | 21 | 0 | 15 |
| `governance/loop.py` | 282 | 40 | 25 | 58 | 13 | 8 | 138 |
| `governance/never_list.py` | 18 | 3 | 0 | 5 | 2 | 0 | 8 |

The category columns are a mechanical classification of the mutant's diff
(see `MUTANTS_decision.md`); it is a pointer for where to look, not a triage.
Of the new wiring in `Loop` (`record_incident`, `check_feedback`,
`guard_registry`) and in the registry only three mutants survived, all in
`check_feedback`: the text of its error (now pinned by an exact-message test
added after this run, so not re-measured), and two that drop or null the
`proposed_default` argument of `escalate`, which are equivalent because
`escalate` substitutes `policy.safe_default` when the argument is falsy. The
other survivors of `loop.py` (138 in the `other` column) were not read.
Below the 80 percent target: none of the four modules.

## Scope

All of these modules and their tests are in `setup.cfg`. `tests/unit/MUTANTS_decision.md`,
`MUTANTS_identification.md`, `MUTANTS_evaluation.md` and
`MUTANTS_framing_ingest.md` hold the other new modules.
