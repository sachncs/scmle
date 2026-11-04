# Mutation results: uncertainty and scores (`scmle/evaluation/`)

mutmut 3.8.0 on Python 3.14 (free-threaded build), run on 2026-11-05 in copies
of the working tree (`also_copy` as in `setup.cfg`, `timeout_multiplier = 2`,
`--max-children 4`; the machine was shared and heavily loaded). Each batch used
only the tests named for its modules in `setup.cfg`. "Not killed" is survived
plus timeout; `segfault` mutants (a crashed test process) are in neither column
and so are not counted as killed here. Score is `killed / (killed + survived +
timeout)`, as in `scripts/check_mutation_score.sh`. The ingestion and
identification copies include other agents' uncommitted edits of those
modules at the time.

Modules: `conformal.py`, `coverage.py`, `ipcw.py`, `proper_scores.py`,
`reconcile.py`. Tests: `test_conformal.py`, `test_uncertainty_methods.py`
and the added `test_evaluation_golden.py` (fixture
`tests/fixtures/evaluation_golden.json`). The run already includes the golden
tests; no run without them was kept.

Pins. The behaviour of the public functions of these modules on fixed inputs
is recorded in a JSON fixture and compared by the golden tests named below
(`tests/unit/golden.py` rounds floats to 9 digits). That is a characterization
pin: it notices a change of behaviour, it does not show the recorded behaviour
is right. Survivors that remain are mostly parameters and branches the golden
inputs do not reach, error-message text, and defensive conversions.

| Module | Killed | Survived | Timeout | Score |
| --- | --- | --- | --- | --- |
| `evaluation/conformal.py` | 318 | 87 | 0 | 78.5% |
| `evaluation/coverage.py` | 273 | 54 | 0 | 83.5% |
| `evaluation/ipcw.py` | 164 | 25 | 0 | 86.8% |
| `evaluation/proper_scores.py` | 130 | 34 | 0 | 79.3% |
| `evaluation/reconcile.py` | 119 | 20 | 0 | 85.6% |
| Total | 1004 | 220 | 0 | 82.0% |

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
| `evaluation/conformal.py` | 87 | 14 | 33 | 2 | 17 | 5 | 16 |
| `evaluation/coverage.py` | 54 | 5 | 16 | 3 | 6 | 2 | 22 |
| `evaluation/ipcw.py` | 25 | 5 | 11 | 1 | 3 | 1 | 4 |
| `evaluation/proper_scores.py` | 34 | 6 | 22 | 0 | 4 | 0 | 2 |
| `evaluation/reconcile.py` | 20 | 0 | 16 | 0 | 1 | 0 | 3 |

Most of `numpy/zip keyword` here are `dtype=float` conversions of inputs the
tests already pass as floats (a mutant dropping the conversion does not change
the result) and are probably equivalent for the inputs the API accepts, but
this was not checked per mutant. Below the 80 percent target: `conformal.py`
and `proper_scores.py`.
