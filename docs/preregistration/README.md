# Pre-registrations of the scorecard claims

Plan section 6: "The analysis for each row is written and committed before
its data is scored." This folder holds that analysis for each claim C1 to
C14 of `benchmarks/scorecard.py`. Each file states the population and data,
the split the row is decided on, the statistic, the decision rule with its
`acceptance.toml` variables and `alpha`, the planned sample size and its
derivation, what counts as a failure, who grades it, and where the current
evidence module differs from the registered analysis.

Location. The plan names `plan/preregistration/`, but `plan/` is ignored by
git in this repository (by `.git/info/exclude`, not by `.gitignore`), so a
file there could never be committed and would not count as registered. The
registrations live here instead.

Registered on 2026-11-17, against `acceptance.toml` as of commit 53790feb.

| Claim | File | Decided on | Evidence module | Today |
|---|---|---|---|---|
| C1 No silent wrong outcome | [c1.md](c1.md) | validation | `benchmarks/evidence/faults.py` | measured, development only |
| C2 Leakage caught, clean data not blocked | [c2.md](c2.md) | acceptance | `benchmarks/evidence/leakage.py` | measured, development only |
| C3 Statistics correct and uncertainty honest | [c3.md](c3.md) | validation | `benchmarks/evidence/coverage.py` | coverage part only |
| C4 At least a competent practitioner's quality | [c4.md](c4.md) | acceptance | `benchmarks/evidence/quality.py` | tabular, development tasks |
| C5 Data read and joined correctly | [c5.md](c5.md) | validation | `benchmarks/evidence/reading.py` | simulated worlds and builder's corpus |
| C6 Reproducible and traceable | [c6.md](c6.md) | validation | `benchmarks/evidence/reproducibility.py`, `benchmarks/evidence/platforms.py` | one platform |
| C7 Operates without loss or stall | [c7.md](c7.md) | validation | `benchmarks/evidence/budget.py` | budget and kill-resume only |
| C8 Controls cannot be bypassed | [c8.md](c8.md) | stage D exercise | `benchmarks/evidence/bypass.py` | automated search only, not the exercise |
| C9 Secure, private, trustworthy supply chain | [c9.md](c9.md) | validation | none | not measured |
| C10 Can keep changing safely | [c10.md](c10.md) | validation | `scripts/structure.py`, `benchmarks/evidence/mutation.py` | size budgets and a mutation sample |
| C11 A non-specialist uses it correctly | [c11.md](c11.md) | stage D sessions | none | not measured |
| C12 Docs tell the truth | [c12.md](c12.md) | validation | `benchmarks/evidence/guarantees.py`, `benchmarks/evidence/snippets.py` | linkage and snippets |
| C13 Every stage on steroids | [c13.md](c13.md) | validation; study in stage D | none | not measured |
| C14 Refuses only when it should | [c14.md](c14.md) | acceptance | `benchmarks/evidence/refusal.py` | rate on development tables |

## Rules shared by every row

- **Results scored before this registration decide nothing.** Every stored
  `benchmarks/evidence/*.json` was computed before these files existed,
  mostly on data the builder chose and looked at. They are development
  evidence. A row is decided only by a run on its registered split made
  after the commit that adds its registration.
- **Splits.** Every evidence set is split three ways before it is scored
  (plan section 6). For simulated evidence the split is a disjoint seed
  stream: the streams the modules use today are development; validation
  uses a stream named `<module>.validation` that has never been drawn;
  simulated evidence has no acceptance split. For real corpora the split is
  by source table (never by row, so a table and a leak planted on it fall
  on the same side) and is fixed by a seeded hash, as
  `benchmarks/sota/split.json` does. No real corpus is split three ways
  yet; each file says what has to be sealed before its row can be decided.
- **Statistical rows** are decided by the one-sided Clopper-Pearson bound
  of `benchmarks.acceptance` (`Ceiling` or `Floor`) at `alpha`. `minimum`
  is the smallest sample that can pass at all; `planned` is
  `rule.plan(truth, alpha, power)`, the smallest sample whose exact
  binomial chance of passing reaches `power` = 0.80 at the stated true
  rate. Sample-size tables in these files list
  `variable | rule | true rate | alpha | minimum | planned`, and
  `tests/benchmarks/test_preregistration.py` recomputes every row of them
  from `benchmarks.acceptance`, so a changed threshold that would make a
  registration wrong fails CI.
- **Multiplicity.** The plan applies Holm across a stage's rows. Stage D
  has four Clopper-Pearson acceptance rows (C2 recall, C2 false blocks,
  C14 rate, C14 precision), so their planned samples are sized at the
  first Holm step, `alpha / 4` = 0.0125; at the unadjusted plan their
  power under Holm falls to about 0.3. Validation rows are sized at
  `alpha`. Holm itself is not implemented in `benchmarks/acceptance.py`
  or `benchmarks/scorecard.py` today: every row is decided at the
  unadjusted `alpha`. That is the first difference every module shares.
- **Graders.** Claude builds and tunes the product; the owner is the
  practitioner. Neither grades a row it tunes. A row is graded by a script
  committed and hashed (the module's `digest`) before the data are scored,
  run by CI; the validator reviews that script before opening the
  acceptance split, and opens it once. Labels, rubrics and expected values
  that decide a row are written by someone who did not build the component
  being graded. Today the builder wrote every evidence module and every
  corpus, so no row yet meets this rule; each file says what replaces it.
- **One shot on acceptance.** A row decided on the acceptance split is not
  re-run there after a fix; a fix needs a new acceptance split of the
  planned size (plan section 6).
- **Raw results** are committed next to the summary, as each module's
  `<module>.json` already is.
