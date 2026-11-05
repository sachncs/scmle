# Leakage: what the screen can and cannot tell you

This page explains one idea that decides how much to trust a score: leakage.
It is written in plain words, but the statements are exact, and every number
comes from a file in this repository that you can regenerate.

New words are in the [glossary](GLOSSARY.md). If you only want to know what to
do when a run says `PROVISIONAL`, read [When things go wrong](WHEN_THINGS_GO_WRONG.md).

## What leakage is

A **leak** is a feature whose value you would not have at the moment you use
the model, usually because it is recorded after the outcome or because of it.
The model gets part of the answer and scores far too well. On the Titanic
table, `boat` (the lifeboat) and `body` (whether the body was recovered) are
leaks for the question "who survived, predicted at boarding".

A **strong legitimate predictor** is a feature that is just very informative
and is known in time. On the breast-cancer table, the measurements of cell
size (`worst radius`, `worst area`) separate malignant from benign samples
almost perfectly, and they are measured from the image at the time of the
sample. They are not leaks.

## Why the data alone cannot tell them apart

Take two stories about one column `C` and the answer `Y`:

1. `C` is measured before `Y` is known, and it is very informative about `Y`.
2. `C` is recorded after `Y` is known, and is largely a copy of `Y`.

Both stories can produce exactly the same joint distribution of `(C, Y)`. Any
procedure that reads only the table (any function of the rows) behaves
identically under the two stories. So no test on the data alone can separate
them. This is a property of the problem (it is called **identifiability**),
not a weakness of one method. The missing fact is *when the value is known*,
and that is a fact about the world, not about the table.

Consequence: any data-only screen has **false flags** (legitimate columns it
calls suspicious) or misses leaks, and usually both. The question is only how
well the error rates are measured and what is promised.

## What the screen tests

The screen is `scmle.data.admissibility.screen`. For every candidate column:

1. **Skill.** Fit a small smoother of the answer on that one column, using
   cross-fitting (each row is predicted by a model that did not see it), and
   compute the held-out ROC AUC. Compare it with the same procedure on many
   shuffled copies of the answer to get a permutation p-value.
2. **Missingness.** Test whether *being recorded* depends on the answer (a
   value present only for some outcomes), and whether the recorded values are
   nearly all one class ("record purity").
3. **Multiplicity.** All the p-values form one family, adjusted with the
   Benjamini-Yekutieli procedure at level 0.05. This controls the **false
   discovery rate**: among the columns declared significant, the expected share
   that are truly null is at most 0.05 times the share of null columns, under
   any dependence between columns.
4. **Ceiling.** A column becomes a **suspect** only if it is significant *and*
   the lower end of a 95 percent bootstrap interval for its skill is above a
   ceiling fixed in advance (AUC 0.95; for missingness, Cramer's V 0.9).
   Significance alone is not enough, because honest strong predictors are
   significant too.

The ceilings are fixed before measuring and are not tuned to any table.

## What the guarantee is, and what it is not

**It is** a statement about a world with no excess dependence. If no column
has anything to do with the answer, the chance that the screen flags *any*
column is at most 5 percent. The repository measures this on 1000 such worlds
made from real breast-cancer columns with the answer shuffled: the screen
flagged a column in **0 of 1000** of them (95 percent Wilson interval for the
false-alarm rate: 0 to 0.38 percent). It also measures detection of planted
leaks (below).

**It is not** a statement about legitimate strong predictors. They are not null.
They do depend on the answer, and a column whose honest skill is above the
ceiling will be flagged. The false discovery rate bounds false flags *among
null columns*; it says nothing about a column that is really informative for
honest reasons. The certificate says this itself: strong legitimate predictors
are "not on the clean side: the test is expected to call those suspects".

**It is not** a proof of leakage. A flag is a suspicion that has to be settled
by provenance: when is the value known? Only a person can answer.

## What was measured

Everything below is in
[`benchmarks/leakage_screen_measurements.json`](../benchmarks/leakage_screen_measurements.json),
produced by `benchmarks/leakage_screen_measurements.py`, with the shipped
certified settings (skill ceiling 0.95, level 0.05, 199 permutations, seed 0).
"True leak" is decided by provenance, not by the data: every column of the
scikit-learn tables is a legitimate measurement; on Titanic only `boat` and
`body` are recorded after the outcome.

### Detection of planted leaks (certified)

Five leak types were planted among real breast-cancer columns, 200 tables per
type (the certificate, `benchmarks/qualification/screen_measurements.json`):

| Planted leak | Found |
| --- | --- |
| a copy of the answer | 200 of 200 |
| a copy with 2 percent of labels flipped | 199 of 200 |
| a threshold rule with noise | 200 of 200 |
| a value recorded only when the answer is not one class | 200 of 200 |
| a value recorded for a quarter of one class only | 200 of 200 |

The 95 percent Wilson interval for the worst type (199 of 200) is 97.2 to 99.9
percent. These are synthetic plants, so detection is optimistic; real leaks
can be weaker.

A second, smaller check in this repository's measurements file plants each
type in 10 tables of 9 real columns plus the plant: **50 of 50** planted leaks
were flagged.

### False flags on legitimate columns (not certified, measured)

| Table | Classes | Columns | Flagged | Would stop a run under `block` |
| --- | --- | --- | --- | --- |
| breast cancer | 2 | 30 | `worst radius`, `worst area` (2 of 30, 6.7 percent) | yes |
| wine | 3 | 13 | none (0 of 13) | no |
| iris | 3 | 4 | `petal length (cm)`, `petal width (cm)` (2 of 4) | no |
| digits | 10 | 64 | none (0 of 64) | no |
| Titanic, as shipped | 2 | 9 | `boat`, `body`: both true leaks | yes |
| Titanic without `boat` and `body` | 2 | 7 | none (0 of 7) | no |

Read this carefully:

* On breast cancer two of 30 honest measurements are flagged: a **false flag
  rate of 6.7 percent** on this table. In the 10-table planting experiment, 7
  of the 90 honest columns screened (7.8 percent) were flagged. Those 10 tables
  are the same for every leak type, so these are not independent repeats.
  Which of the size measurements is flagged depends on the rows the screen sees.
  A normal run on the development rows flagged `worst perimeter` instead, and
  a charter run on its exploration rows flagged `worst area`.
* On Titanic as shipped the screen found both real leaks (2 of 2) and flagged
  nothing legitimate (0 of 7). Without the two leaks, it flagged nothing.
* For wine, iris and digits the target has more than two classes. The screen is
  only certified for a two-class target, so for these it is **descriptive**: it
  may report suspects (iris did) but it can never stop a run, whatever the
  policy.
* This is six tables, and only breast cancer combines a two-class target with
  very strong honest predictors. Do not read 6.7 percent as a general rate. It
  says: where honest predictors are very strong, expect some false flags.

### Reproduce

```bash
python -m benchmarks.leakage_screen_measurements --check
python -m benchmarks.qualification.screen_certify --check
```

Run both from a source checkout (they read `tests/fixtures/titanic_extract.csv`
and the certificate files). The first takes several minutes. `--write` rewrites
the file. The second re-measures the 1000 null worlds and the planted leaks.

## What the product does about it

The right design is a **qualification instrument**: it attaches evidence and a
tier to every claim and never hands you a silent verdict. For leakage:

* **It trains and reports loudly.** The default `leakage_policy` is `warn`. The
  run logs a line naming the flagged columns, labels the score
  `test (PROVISIONAL: N flagged column(s) not attested: ...)` on its last line
  and in the first line of `report.html`, and lists the suspicions under
  "Findings" at the top of the report.
* **It does not refuse by default.** A default refusal would stop honest
  analyses (it would stop breast cancer). A default silence was also wrong: an
  earlier version printed nothing at the end of a run that scored 0.988 on
  Titanic because of two leaks. Loud, labelled, and settled by a person is the
  middle path.
* **It lets you stop.** `--leakage-policy block` (opt-in) refuses a run that has
  a flagged column whose test is certified. Use it when you cannot attest where
  each value comes from, and in regulated settings where an unexplained column
  must not reach a model.
* **In a charter run it withholds.** A certified suspect is left out of the
  features until a person attests it. With a valid attestation the column is
  kept and recorded at tier `attested`, never `confirmed`.

What a tier does and does not say here: the tier of a **claim** (`confirmed`,
`attested`, `unverified`) is about the one-time test on the sealed rows. In
the breast-cancer charter runs below, the claims "worst radius is associated
with the answer" were `confirmed` even though the column was flagged and not
attested. That is true as a statement about association, and says nothing about
provenance. Provenance is recorded separately, per feature, in
`admissibility_table.json` (`unverified`, `attested`). Making a claim's tier
depend on the provenance of a column it uses would be a new rule. It has not
been designed or built, and this page does not claim it.

## How to settle a flag

Ask: **when is this value known?**

### Titanic: the value is not known in time

`boat` and `body` are only known after the sinking. Declare them:

```bash
scmle run titanic.csv --target survived --unavailable-at-prediction boat body \
  --time-budget 120 --workers 2 --output-dir runs
```

The flag is gone, the label is gone, and the exam ROC AUC falls from 0.988 to
the honest 0.885 (see [Start here](START_HERE.md), Steps 4 and 6).

### Breast cancer: the value is known in time

Suppose a pathologist confirms that `worst radius`, `worst perimeter` and
`worst area` are computed from the same image at the time the sample is taken,
so they are available at prediction. Settling the flag means recording that,
with a name, a statement and an expiry. Attestations are used in charter runs.

The file (`att2.json`; `scope` must be the `charter_hash` of your signed
charter, so an attestation cannot be reused for another analysis):

```json
{
  "format_version": 1,
  "attestations": [
    {
      "column": "worst area",
      "name_or_id": "A. Pathologist",
      "role": "data owner",
      "statement": "Measured from the same image as every other column, when the sample is imaged; available at prediction.",
      "evidence": "imaging protocol v2, section 4",
      "date": "2026-11-05",
      "expires": "2027-11-05",
      "scope": "<the charter_hash of your signed charter>",
      "lead_seconds": 0
    }
  ]
}
```

All nine keys are required for each entry (one entry per column). `expires`
must not be before `date`. The run is told about the file with
`--leakage-attestations att2.json`:

```bash
scmle run cancer.csv --target target --charter charter2.json \
  --leakage-attestations att2.json --time-budget 120 --workers 2 --output-dir runs_att
```

What happened in the real runs (charter with a 10 percent confirmation set; all
three columns attested in the file):

Without the file, the run quarantined the flagged `worst area`:

```text
"quarantined": {"worst area": "availability at serving is unknown; as-of rule is unverified; leakage suspicion with no as-of evidence"}
```

With the file, nothing was quarantined, `worst area` was kept at tier
`attested` (and `admit_flagged`), and the run noted for the other two columns:

```text
attestation for 'worst radius' unused: the screen did not flag it
attestation for 'worst perimeter' unused: the screen did not flag it
```

Two things to notice. The charter run's screen looks at the exploration rows
only, which are fewer than the development rows, so it flags fewer columns
(the lower bound on skill falls below the 0.95 ceiling for some of them). And
the run's last line still said
`PROVISIONAL: 2 flagged column(s) not attested: worst perimeter, worst radius`,
because the development-row screen flagged them and the attestation was unused.
That label is conservative; it is a known rough edge (see
[Beginner findings](BEGINNER_FINDINGS.md)).

An attestation is as good as the person. SCMLE records who said it, for which
analysis, until when. It does not check that they were right.

## When to use block

Choose `--leakage-policy block` when:

* you cannot get a person to say when each value is known;
* the work is regulated, and an unexplained strong column must not reach a
  model;
* you would rather stop and ask than read a label.

The price is false stops on honest analyses, and for tables with more than two
classes or regression the screen is not certified, so `block` cannot stop them.

## Summary

| Claim | True? |
| --- | --- |
| The screen finds planted leaks | yes: 199 to 200 of 200 per type, certified |
| The screen almost never flags a world with no dependence | yes: 0 of 1000 |
| A flag means leakage | no: a flag is a suspicion |
| The screen flags legitimate strong predictors | yes: 2 of 30 on breast cancer |
| No flag means no leak | no: leaks needing several columns together are not seen |
| A person can settle a flag | yes, by provenance: declare it unavailable or attest it |
