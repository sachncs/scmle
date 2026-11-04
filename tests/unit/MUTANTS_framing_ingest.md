# Mutation results: problem framing and raw-data ingestion

mutmut 3.8.0 on Python 3.14 (free-threaded build), run on 2026-11-05 in copies
of the working tree (`also_copy` as in `setup.cfg`, `timeout_multiplier = 2`,
`--max-children 4`; the machine was shared and heavily loaded). Each batch used
only the tests named for its modules in `setup.cfg`. "Not killed" is survived
plus timeout; `segfault` mutants (a crashed test process) are in neither column
and so are not counted as killed here. Score is `killed / (killed + survived +
timeout)`, as in `scripts/check_mutation_score.sh`. The ingestion and
identification copies include other agents' uncommitted edits of those
modules at the time.

Framing tests: `test_framing_cost_model.py`, `_baseline.py`, `_feasibility.py`,
`_provenance_questions.py`, `_draft.py`. Ingestion tests:
`test_ingest_profile.py`, `_grain.py`, `_joins.py`, `_synthesize.py`,
`_reconcile.py`, `_entities.py`, `test_connector_consistency.py`.
`ingest/plan.py` produced no mutants (no covered line). These modules were
in the scope already (except `framing/draft.py`, `ingest/entities.py`,
`ingest/plan.py`) but had no recorded run. **No pinning tests were added for
them in this pass**; the numbers are what the existing tests give.

## Framing

| Module | Killed | Survived | Timeout | Score |
| --- | --- | --- | --- | --- |
| `data/framing/baseline.py` | 417 | 151 | 0 | 73.4% |
| `data/framing/costs.py` | 237 | 70 | 0 | 77.2% |
| `data/framing/draft.py` | 535 | 214 | 0 | 71.4% |
| `data/framing/feasibility.py` | 159 | 62 | 0 | 71.9% |
| `data/framing/provenance.py` | 5 | 0 | 0 | 100.0% |
| `data/framing/questions.py` | 74 | 16 | 4 | 78.7% |
| `data/framing/refusal.py` | 201 | 104 | 0 | 65.9% |
| Total | 1628 | 617 | 4 | 72.4% |

| Module | Not killed | message/string | numpy/zip keyword | replaced by None | comparison boundary | arithmetic operator | other |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `data/framing/baseline.py` | 151 | 38 | 10 | 28 | 10 | 5 | 60 |
| `data/framing/costs.py` | 70 | 25 | 8 | 10 | 7 | 0 | 20 |
| `data/framing/draft.py` | 214 | 66 | 0 | 40 | 6 | 0 | 102 |
| `data/framing/feasibility.py` | 62 | 11 | 1 | 11 | 5 | 10 | 24 |
| `data/framing/questions.py` | 20 | 2 | 0 | 3 | 2 | 3 | 10 |
| `data/framing/refusal.py` | 104 | 26 | 12 | 10 | 7 | 3 | 46 |

## Ingestion

| Module | Killed | Survived | Timeout | Score |
| --- | --- | --- | --- | --- |
| `data/ingest/asof_view.py` | 134 | 41 | 0 | 76.6% |
| `data/ingest/budget.py` | 58 | 13 | 0 | 81.7% |
| `data/ingest/entities.py` | 634 | 253 | 1 | 71.4% |
| `data/ingest/grain.py` | 309 | 118 | 0 | 72.4% |
| `data/ingest/joins.py` | 352 | 98 | 3 | 77.7% |
| `data/ingest/profile.py` | 239 | 81 | 5 | 73.5% |
| `data/ingest/reconcile.py` | 84 | 4 | 0 | 95.5% |
| `data/ingest/synthesize.py` | 499 | 95 | 1 | 83.9% |
| Total | 2309 | 703 | 10 | 76.4% |

| Module | Not killed | message/string | numpy/zip keyword | replaced by None | comparison boundary | arithmetic operator | other |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `data/ingest/asof_view.py` | 41 | 17 | 2 | 11 | 0 | 1 | 10 |
| `data/ingest/budget.py` | 13 | 2 | 0 | 0 | 5 | 0 | 6 |
| `data/ingest/entities.py` | 254 | 13 | 21 | 53 | 13 | 35 | 119 |
| `data/ingest/grain.py` | 118 | 14 | 0 | 37 | 11 | 9 | 47 |
| `data/ingest/joins.py` | 101 | 11 | 6 | 27 | 2 | 8 | 47 |
| `data/ingest/profile.py` | 86 | 16 | 0 | 27 | 0 | 4 | 39 |
| `data/ingest/reconcile.py` | 4 | 1 | 0 | 0 | 0 | 0 | 3 |
| `data/ingest/synthesize.py` | 96 | 33 | 0 | 19 | 2 | 5 | 37 |

The category columns are a mechanical classification of the mutant's diff
(`message/string`: text of an error message; `numpy/zip keyword`: a `dtype=`,
`strict=`, `axis=`, `ddof=` or `copy=` argument; `replaced by None`: an
argument or value replaced by `None`; `comparison boundary`: `<` against `<=`
and similar; `arithmetic operator`; `other`). It is a heuristic for finding
where to look, **not a triage**: a mutant in a category is not thereby
equivalent, and no survivor listed in the tables was proved equivalent unless
a paragraph says so.

Below the 80 percent target: every framing module except `provenance.py`, and
`entities.py`, `grain.py`, `profile.py`, `asof_view.py` and `joins.py` in
ingestion. No survivor was read.
