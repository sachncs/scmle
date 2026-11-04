# Mutation results: the modules that were already in the mutation scope

mutmut 3.8.0 on Python 3.14 (free-threaded build), run on 2026-11-04 in copies
of the working tree, started between commits `4307a7d` and `15b6175`, plus other agents' uncommitted edits.
Each batch used only the tests of its modules (the list in `setup.cfg`)
because the full selection takes hours on this machine. The custody, charter,
vault and attestation modules are in `tests/unit/MUTANTS_custody.md`.

Score is `killed / (killed + survived + timeout)`, as in
`scripts/check_mutation_score.sh`. "Crash" counts mutants for which mutmut
reported a segfault or an aborted test process; a crash is not a failed
assertion, so it is not counted as killed here.

Except for the five ledger mutants named below, no survivor in this file was
triaged one by one or pinned by a test: the numbers are what the existing
tests give. They are a record, not a claim that the survivors are
equivalent. The release target in the phase exit gate is 80; modules below it
are listed.

## Ledger (`scmle/data/ledger/`)

Tests: `test_online_fdr.py`, `test_online_fdr_properties.py`,
`test_multiplicity.py`, `test_ledger_controls.py`, `test_ledger_bivariate.py`,
`test_ledger_mutants.py`, `test_ledger_mutants_io.py`, `test_ledger.py`,
`test_ledger_confirm.py`, `test_ledger_governed.py`, `test_error_budget.py`,
`test_error_budget_boundaries.py`.

| Module | Killed | Survived | Timeout | Score |
| --- | --- | --- | --- | --- |
| `online_fdr.py` | 483 | 0 | 0 | 100% |
| `budget.py` | 309 | 1 | 0 | 99.7% (the equivalent `Fraction(0)` start, see `MUTANTS_budget.md`) |
| `controls.py` | 446 | 9 | 0 | 98.0% |
| `multiplicity.py` | 185 | 3 | 1 | 97.9% |
| `bivariate.py` | 236 | 6 | 0 | 97.5% |
| `explore.py` | 121 | 10 | 0 | 92.4% |
| `ledger.py` | 679 | 70 | 0 | 90.7% |
| `governed.py` | 191 | 35 | 0 | 84.5% |
| `confirm.py` | 472 | 93 | 0 | 83.5% |
| Total | 3122 | 227 | 1 | 93.2% |

Five survivors were pinned by `test_ledger_mutants_edges.py` and checked by
applying each mutant (`quantile_index` floor, two `percentile_interval`
indices, `nunique(dropna=...)`, the regression-target message), which brings
the total to 3127 killed and 222 survived. The other survivors of
`controls.py`, `multiplicity.py` and `bivariate.py` that were read are
equivalent: an `np.array(..., copy=True)` copy flag on a freshly built array,
`kind="STABLE"` (numpy accepts it), initial values `[0.0] * m` and
`[False] * m` that every position overwrites, an `index` bound that cannot be
reached, and `rho >= 0` where `|rho| >= 1` already holds. Those in `ledger.py`,
`confirm.py` and `governed.py` were not read; the files are being edited in
another lane.

## Admissibility (`scmle/data/admissibility/`, without `attestations.py`)

Tests: `test_admissibility_{provenance,asof,rules,dag,independence,leakage,reconciliation}.py`
and `test_leakage_screen.py`. Four `test_leakage_screen.py` tests that check a
shipped certificate against the code hash cannot pass under mutation (mutmut
rewrites the source, so the hash differs) and were deselected for this batch:
`test_the_shipped_certificate_is_current_for_the_screen_code`,
`test_only_the_certified_target_and_settings_may_gate`,
`test_a_mechanism_without_a_passing_type_stays_descriptive` and
`test_with_the_shipped_certificate_a_suspect_blocks`. The `screen.py` numbers
therefore measure the other tests only.

| Module | Killed | Survived | Timeout | Crash | Score |
| --- | --- | --- | --- | --- | --- |
| `asof.py` | 272 | 38 | 0 | 0 | 87.7% |
| `dag.py` | 320 | 41 | 3 | 0 | 87.9% |
| `provenance.py` | 296 | 67 | 0 | 0 | 81.5% |
| `leakage.py` | 962 | 148 | 0 | 102 | 86.7% of 1110 |
| `independence.py` | 347 | 90 | 0 | 0 | 79.4% |
| `rules.py` | 276 | 85 | 0 | 0 | 76.5% |
| `reconciliation.py` | 145 | 60 | 0 | 0 | 70.7% |
| `screen.py` | 650 | 8 | 0 | 732 | 98.8% of 658 |
| Total | 3268 | 537 | 3 | 834 | 85.8% of 3808 |

Below the 80 target: `independence.py` (79.4), `rules.py` (76.5) and
`reconciliation.py` (70.7). No module was dropped from the scope. The crashes
in `screen.py` and `leakage.py` are mutants that make the mutated process
die; they are left out of the score rather than counted as kills.

## Robustness (`scmle/data/robustness/`)

Tests: `test_robustness_{multiverse,stability,shift,null,coverage_forward,qualify}.py`.
`test_the_shipped_certificates_are_found_from_a_checkout` was deselected (it
looks for the certificate files from a checkout, which the mutants folder is
not). `test_robustness_prefilter.py` and `test_robustness.py` are not in the
selection (they were not in the scope before this pass; `prefilter.py` is
not mutated).

The run used `timeout_multiplier = 2` after a first run with the default made
the batch take hours. On this overloaded machine a slow test that would have
finished is reported as a timeout, so the timeout column mixes true infinite
loops with slow runs and the score below is a lower bound.

| Module | Killed | Survived | Timeout | Score |
| --- | --- | --- | --- | --- |
| `coverage.py` | 95 | 5 | 11 | 85.6% |
| `multiverse.py` | 146 | 11 | 18 | 83.4% |
| `stability.py` | 618 | 36 | 136 | 78.2% |
| `forward.py` | 174 | 51 | 1 | 77.0% |
| `qualify.py` | 116 | 0 | 47 | 71.2% |
| `report.py` | 131 | 69 | 0 | 65.5% |
| `shift.py` | 503 | 0 | 268 | 65.2% |
| Total | 1783 | 172 | 481 | 73.2% |

Below the 80 target: `stability.py`, `forward.py`, `qualify.py`, `report.py`
and `shift.py`. For `qualify.py` and `shift.py` every non-killed mutant is a
timeout, none survived; a run without load or with a longer multiplier would
move them. For `report.py` and `forward.py` the survivors are real gaps in the
assertions on the report text and fields; they were not triaged. No module
was dropped from the scope.

## Detectors

Modules: `scmle/data/detectors/{certificates,missingness,labels,sentinels,duplicates,outliers,registry,pipeline}.py`
(`registry.py` produced no mutants). Tests: `test_detector_{missingness,labels,sentinels,structure}.py`
and `test_duplicate_groups.py`. Same `timeout_multiplier = 2` and load caveat
as for robustness: the timeout column mixes infinite loops with slow runs,
and the score is a lower bound.

| Module | Killed | Survived | Timeout | Score |
| --- | --- | --- | --- | --- |
| `sentinels.py` | 266 | 3 | 49 | 83.7% |
| `duplicates.py` | 183 | 3 | 33 | 83.6% |
| `missingness.py` | 692 | 58 | 103 | 81.1% |
| `outliers.py` | 156 | 17 | 28 | 77.6% |
| `certificates.py` | 166 | 43 | 61 | 61.5% |
| `labels.py` | 253 | 0 | 223 | 53.2% |
| `pipeline.py` | 48 | 0 | 65 | 42.5% |
| Total | 1764 | 124 | 562 | 72.0% |

Below the 80 target: `outliers.py`, `certificates.py`, `labels.py` and
`pipeline.py`. For `labels.py` and `pipeline.py` every non-killed mutant is a
timeout (the label-noise detectors and the pipeline run permutation or
cross-validated fits that are slow under this load); none survived. For
`certificates.py` and `outliers.py` the survivors are untriaged gaps. No
module was dropped from the scope.
