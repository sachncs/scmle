# Mutation survivors: custody, charter, vault and attestations

mutmut 3.8.0 on Python 3.14 (free-threaded build), run on 2026-11-04 in a copy
of the working tree (the second custody and charter runs started at `4307a7d`
and `15b6175`; other agents' uncommitted edits were in files outside these
modules). Modules and tests are listed in `setup.cfg`
(`source_paths`, `pytest_add_cli_args_test_selection`). Each batch was run with
a narrowed test selection (only the tests of these modules) because the full
selection takes hours on this machine; the selection is the one in
`setup.cfg` for these modules.

A mutant counts as killed when a test fails, as a timeout when the mutated
code loops (an infinite loop is detected, but mutmut counts it apart). The
gate in `scripts/check_mutation_score.sh` scores `killed / (killed + survived +
timeout)`.

## Custody modules and the vault

Modules: `scmle/data/custody/{carve,manifest,merkle,pipeline,sealed}.py`,
`scmle/governance/vault.py`. Tests: `tests/unit/test_custody_*.py`,
`tests/property/test_custody_properties.py`, `tests/unit/test_vault.py`,
`tests/leakage/test_custody_access.py`.

| Step | Mutants | Killed | Survived | Timeout | Score |
| --- | --- | --- | --- | --- | --- |
| First run, tests as they were before this pass | 1500 | 1234 | 261 | 5 | 82.3% |
| After `test_custody_mutants.py` and `test_custody_mutants_data.py` | 1516 | 1444 | 67 | 5 | 95.3% |

Of the 1444, 1424 were killed by the second mutmut run. 16 more mutants
(error-message wording and dropped positional arguments, which mutmut left
"not checked" because their lines were first reached by the new tests) and 4
`reset_index(drop=...)` mutants were killed by applying each one with
`mutmut apply` and running the custody tests against it; the baseline passed
in that copy.

Per module after the second run (survived + timeout, before those 20 manual
kills): carve 18 + 1, manifest 7, merkle 10 + 4, pipeline 24, sealed 10,
vault 2.

### The 72 mutants left (67 survived, 5 timeout), none pinned

Counted by pattern on the generated diffs; a few mutants match two patterns, so the counts are approximate. None is a behaviour a test could
distinguish on the machine used, except where the reason says otherwise.

| Count | Mutation | Why it is left |
| --- | --- | --- |
| 4 | `encoding="utf-8"` becomes `None` or a missing argument | Equivalent on a UTF-8 locale. On Windows the locale encoding differs, so a test with non-ASCII content could kill it there; none was written |
| 2 | `"utf-8"` becomes `"UTF-8"` | Same codec |
| 8 | `strict=True` becomes `False`, `None` or is dropped in `zip` calls of `merkle.row_hashes` | The zipped sequences have equal length by construction (one value per column per row), so `strict` never raises |
| 4 | `kind="stable"` of `argsort` becomes `None`, `"STABLE"` or is dropped | `"STABLE"` is accepted by numpy; `None` selects an unstable sort that only differs for tied times in arrays too large to rely on insertion sort. Not killed: a test would need tied times in a large array and a seed-dependent outcome |
| 11 | `dtype=int`/`float` argument becomes `None` or is dropped in `asarray`, `to_numpy`, `ones` | Inputs are already integer or float, so the default dtype gives the same values |
| 4 | `in_c[...] = False` or `embargo[:] = False` becomes `= None` | Assigning `None` to a boolean array stores `False` |
| 3 | default of `entry.get("mac", ...)` in `OpenLog.verify` | Any default fails the comparison with a real MAC |
| 3 | `np.full(size, "all")` becomes `None`, `"ALL"`, `"XXallXX"` | The strata only need to be equal for all rows |
| 2 | `n_minority` is not passed to `feasibility.assess`, or `minority_rows(None, ...)` | `assess` uses `n_minority` only when `max_half_width` is given, and `pipeline.assess` never passes it. Becomes testable when that argument is wired |
| 2 | `SealedSet.open`: `claims_hash is not None or ledger_hash is not None`, and `... or True` | A partial or absent pair offers a digest that cannot match a committed set, and an uncommitted set ignores the offer, so outcomes are the same |
| 1 | `rows == 0 and not in_c.any()` in `embargo_mask` | With `rows == 0` the convolution window has width 1 and flags only `C` rows, which are then removed: the mask is empty either way |
| 1 | `<=` becomes `<` in `acf_embargo` | Differs only if an autocorrelation equals the band to the last bit |
| 1 | `int.from_bytes(digest[:4], "big")` loses `"big"` | Big-endian is the default since Python 3.11, the minimum supported version |
| 2 | `-1` becomes `-2` as the "no position" or "no duplicate" marker in `restrict_folds` and `unit_labels` | Only negative-ness is tested |
| 1 | `order = None` in `carve_confirmation` | Overwritten before use on the only path that reads it |
| 2 | `process.wait(timeout=...)` becomes `None` or 11 | A timeout only matters if the vault hangs |
| 5 | Loop conditions in `merkle_root` (`>= 1`, `/ 2`, `% 3`) and `unit_labels` (`==`) | Infinite loops: counted as timeouts |

## Charter modules and attestations

Modules: `scmle/data/charter/{contracts,feasibility,rules,sealing}.py`,
`scmle/data/admissibility/attestations.py`. Tests:
`tests/unit/test_charter_*.py` (including `test_charter_mutants.py`,
`test_charter_feasibility_mutants.py`,
`test_charter_rules_sealing_mutants.py`) and
`tests/unit/test_admissibility_attestations.py`.

| Step | Mutants | Killed | Survived | Timeout | Score |
| --- | --- | --- | --- | --- | --- |
| First run | 1131 | 852 | 231 | 48 | 75.3% |
| Second run, after the three `*_mutants` files | 1146 | 1108 | 35 | 3 | 96.7% |
| After 31 more were killed by `mutmut apply` and the tests | 1146 | 1139 | 4 | 3 | 99.4% |

The 7 left are 5 `encoding=` mutants (`None`, `"UTF-8"`; same
Windows caveat as above), 1 `)` that only moves a closing bracket in
`write_charter`, and 1 missing `encoding` argument in
`attestations.load`; all are equivalent on a UTF-8 locale. The mutants
that wrapped a message in extra characters survived the older tests because
`pytest.raises(match=...)` accepts a substring; the new tests compare whole
messages.

## Not covered by this file

The ledger, admissibility, detectors and robustness modules are in
`tests/unit/MUTANTS_scoped.md`.
