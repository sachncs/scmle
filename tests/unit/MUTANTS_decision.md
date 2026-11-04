# Mutation results: the decision layer (`scmle/decision/`)

mutmut 3.8.0 on Python 3.14 (free-threaded build), run on 2026-11-05 in copies
of the working tree (`also_copy` as in `setup.cfg`, `timeout_multiplier = 2`,
`--max-children 4`; the machine was shared and heavily loaded). Each batch used
only the tests named for its modules in `setup.cfg`. "Not killed" is survived
plus timeout; `segfault` mutants (a crashed test process) are in neither column
and so are not counted as killed here. Score is `killed / (killed + survived +
timeout)`, as in `scripts/check_mutation_score.sh`. The ingestion and
identification copies include other agents' uncommitted edits of those
modules at the time.

Tests: `test_decision_anytime.py`, `test_decision_constraints.py`,
`test_decision_ope.py`, `test_decision_rollout.py`,
`test_decision_threshold_netbenefit.py`, and the added
`test_decision_golden.py` (fixture `tests/fixtures/decision_golden.json`).

Pins. The behaviour of the public functions of these modules on fixed inputs
is recorded in a JSON fixture and compared by the golden tests named below
(`tests/unit/golden.py` rounds floats to 9 digits). That is a characterization
pin: it notices a change of behaviour, it does not show the recorded behaviour
is right. Survivors that remain are mostly parameters and branches the golden
inputs do not reach, error-message text, and defensive conversions.

## Before the golden tests (existing tests only)

| Module | Killed | Survived | Timeout | Score |
| --- | --- | --- | --- | --- |
| `decision/anytime.py` | 376 | 128 | 0 | 74.6% |
| `decision/constraints.py` | 514 | 115 | 0 | 81.7% |
| `decision/design.py` | 151 | 60 | 0 | 71.6% |
| `decision/netbenefit.py` | 329 | 11 | 0 | 96.8% |
| `decision/ope.py` | 372 | 120 | 0 | 75.6% |
| `decision/rollout.py` | 313 | 105 | 0 | 74.9% |
| `decision/threshold.py` | 393 | 135 | 0 | 74.4% |
| Total | 2448 | 674 | 0 | 78.4% |

## With the golden tests

| Module | Killed | Survived | Timeout | Score |
| --- | --- | --- | --- | --- |
| `decision/anytime.py` | 396 | 103 | 14 | 77.2% |
| `decision/constraints.py` | 545 | 84 | 0 | 86.6% |
| `decision/design.py` | 163 | 48 | 0 | 77.3% |
| `decision/netbenefit.py` | 373 | 11 | 0 | 97.1% |
| `decision/ope.py` | 403 | 89 | 0 | 81.9% |
| `decision/rollout.py` | 330 | 78 | 10 | 78.9% |
| `decision/threshold.py` | 427 | 101 | 0 | 80.9% |
| Total | 2637 | 514 | 24 | 83.1% |

(75 mutants segfaulted before and 31 after; they are not counted.)

The category columns are a mechanical classification of the mutant's diff
(`message/string`: text of an error message; `numpy/zip keyword`: a `dtype=`,
`strict=`, `axis=`, `ddof=` or `copy=` argument; `replaced by None`: an
argument or value replaced by `None`; `comparison boundary`: `<` against `<=`
and similar; `arithmetic operator`; `other`). It is a heuristic for finding
where to look, **not a triage**: a mutant in a category is not thereby
equivalent, and no survivor listed in the tables was proved equivalent unless
a paragraph says so.

| Module | Not killed | message/string | numpy/zip keyword | replaced by None | comparison boundary | arithmetic operator | other |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `decision/anytime.py` | 117 | 40 | 20 | 10 | 13 | 9 | 25 |
| `decision/constraints.py` | 84 | 17 | 5 | 14 | 10 | 6 | 32 |
| `decision/design.py` | 48 | 16 | 5 | 4 | 8 | 2 | 13 |
| `decision/netbenefit.py` | 11 | 6 | 0 | 2 | 0 | 0 | 3 |
| `decision/ope.py` | 89 | 19 | 14 | 15 | 7 | 3 | 31 |
| `decision/rollout.py` | 88 | 7 | 22 | 9 | 4 | 16 | 30 |
| `decision/threshold.py` | 101 | 27 | 11 | 9 | 8 | 2 | 44 |

Not done: no survivor of this layer was read and proved equivalent, and the
message-text survivors (about a fifth) are not pinned: the tests assert
`pytest.raises(ValueError)` without the text.
