# Mutation results: causal identification (`scmle/data/identification/`)

mutmut 3.8.0 on Python 3.14 (free-threaded build), run on 2026-11-05 in copies
of the working tree (`also_copy` as in `setup.cfg`, `timeout_multiplier = 2`,
`--max-children 4`; the machine was shared and heavily loaded). Each batch used
only the tests named for its modules in `setup.cfg`. "Not killed" is survived
plus timeout; `segfault` mutants (a crashed test process) are in neither column
and so are not counted as killed here. Score is `killed / (killed + survived +
timeout)`, as in `scripts/check_mutation_score.sh`. The ingestion and
identification copies include other agents' uncommitted edits of those
modules at the time.

Tests for the estimators: `test_identification_aipw.py`, `_battery.py`,
`_did.py`, `_iv.py`, `_rd.py`, `_select.py`, `_sensitivity.py`, `_synth.py`,
`_tier.py` and the added `test_identification_golden.py` (fixture
`tests/fixtures/identification_golden.json`). `run.py` was run separately
against `test_identification_run.py`, `_select.py` and `_tier.py`.
`certified.py` produced no mutants (no covered line). The two runs below
already include the golden tests; no run of the estimators without them was
kept.

Pins. The behaviour of the public functions of these modules on fixed inputs
is recorded in a JSON fixture and compared by the golden tests named below
(`tests/unit/golden.py` rounds floats to 9 digits). That is a characterization
pin: it notices a change of behaviour, it does not show the recorded behaviour
is right. Survivors that remain are mostly parameters and branches the golden
inputs do not reach, error-message text, and defensive conversions.

## Estimators

| Module | Killed | Survived | Timeout | Score |
| --- | --- | --- | --- | --- |
| `data/identification/aipw.py` | 492 | 58 | 0 | 89.5% |
| `data/identification/battery.py` | 122 | 25 | 0 | 83.0% |
| `data/identification/did.py` | 294 | 40 | 0 | 88.0% |
| `data/identification/iv.py` | 273 | 15 | 0 | 94.8% |
| `data/identification/rd.py` | 371 | 34 | 0 | 91.6% |
| `data/identification/select.py` | 137 | 59 | 0 | 69.9% |
| `data/identification/sensitivity.py` | 379 | 53 | 0 | 87.7% |
| `data/identification/synth.py` | 179 | 27 | 0 | 86.9% |
| `data/identification/tier.py` | 185 | 12 | 0 | 93.9% |
| Total | 2432 | 323 | 0 | 88.3% |

## `run.py`

| Module | Killed | Survived | Timeout | Score |
| --- | --- | --- | --- | --- |
| `data/identification/run.py` | 839 | 470 | 0 | 64.1% |
| Total | 839 | 470 | 0 | 64.1% |

`run.py` is the orchestration and artifact-writing layer; its 470 survivors
(report fields, file names, messages) are untouched by the golden tests, and
another agent was editing the file during this run. It is the weakest module
of this batch and is below the 80 percent target.

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
| `data/identification/aipw.py` | 58 | 4 | 7 | 8 | 11 | 6 | 22 |
| `data/identification/battery.py` | 25 | 7 | 3 | 0 | 4 | 2 | 9 |
| `data/identification/did.py` | 40 | 7 | 4 | 2 | 6 | 7 | 14 |
| `data/identification/iv.py` | 15 | 4 | 0 | 1 | 4 | 0 | 6 |
| `data/identification/rd.py` | 34 | 1 | 3 | 1 | 13 | 3 | 13 |
| `data/identification/run.py` | 470 | 118 | 7 | 107 | 12 | 2 | 224 |
| `data/identification/select.py` | 59 | 25 | 0 | 5 | 0 | 0 | 29 |
| `data/identification/sensitivity.py` | 53 | 13 | 3 | 2 | 18 | 2 | 15 |
| `data/identification/synth.py` | 27 | 4 | 5 | 1 | 4 | 1 | 12 |
| `data/identification/tier.py` | 12 | 7 | 0 | 0 | 0 | 0 | 5 |

Not done: no survivor was read and proved equivalent; the estimators'
below-target modules are `select.py` (69.9 percent) and `run.py`.
