# Start here: your first model, start to finish

This page takes you from nothing to a model that is built, checked, shipped,
watched and rolled back. You need no data-science background. You need to be
able to open a terminal and type commands.

Every command below was run by the author on a real dataset, and every
"What you should see" block is the real output (trimmed where it says so).
Your numbers may differ in the last digits. Time-limited searches depend on
how fast your computer is.

Where to type: a **terminal** is the window where you type commands. On a Mac
it is the Terminal app, on Linux any terminal. On Windows use WSL (Windows
Subsystem for Linux) or Git Bash, because the commands below are written for
that kind of shell. Type each command, press Enter, and wait for the prompt to
come back before the next one. Lines starting with `#` are comments; skip them.

How long it takes: about 30 minutes, of which about 8 minutes is waiting for
the computer.

If a word is new, look it up in the [glossary](GLOSSARY.md). If a step goes
wrong, look in [When things go wrong](WHEN_THINGS_GO_WRONG.md). If you are
unsure how to choose something, read [Decisions](DECISIONS.md).

## The story in five lines

1. You have a table. One column is the answer you want to predict.
2. You show SCMLE the table and name that column. It tries many models.
3. It keeps some rows hidden until the very end, like an exam. The exam score
   is the number to trust.
4. You save the winning model as a signed folder and let other programs ask it
   for predictions.
5. You watch it. If it gets worse, you go back to the old one.

## What we will predict

We use the Titanic passenger table that ships in this repository. Each row is
one passenger. We want to predict **`survived`**: `1` if the passenger lived,
`0` if not. This column is called the **target**. The other columns (class,
sex, age, fare and so on) are the **features**: the facts the model may use to
guess the target.

## Step 1. Install

You need Python 3.11 or newer. Check:

```bash
python3 --version
```

What you should see: `Python 3.11` or higher (the author used 3.14).
What could go wrong: an older version. Install a newer Python first.

Get the code and create a private Python environment (a **virtual
environment** is a folder that keeps this project's libraries apart from the
rest of your computer):

```bash runnable
git clone https://github.com/sachncs/scmle
cd scmle
python3 -m venv .venv
source .venv/bin/activate
pip install .
```

If you use Windows PowerShell instead of WSL, run `.venv\Scripts\activate`
in place of the `source` line, and expect to adapt other lines.
The install downloads a few hundred megabytes. It may take a few minutes.

Check that the command exists:

```bash runnable
scmle --help
```

What you should see (first lines):

```text
usage: scmle [-h] [--log-level {DEBUG,INFO,WARNING,ERROR}]
             {plan,run,resume,catalog,predict,batch,serve,export-onnx,package,ingest,charter,claims,frame,loop,deploy,trail} ...
```

What could go wrong: `scmle: command not found` means the environment is not
active. Run the `source` line again. Every new terminal needs it.

Make a working folder and copy the data into it. The data file lives in the
repository's test folder:

```bash
mkdir -p ~/first-model
cp tests/fixtures/titanic_extract.csv ~/first-model/titanic.csv
cd ~/first-model
```

All the commands below run inside `~/first-model`.

## Step 2. Look at the data first

Never train on a table you have not looked at. Print its size and first rows:

```bash runnable
python -c "import pandas as pd; d = pd.read_csv('titanic.csv'); print(d.shape); print(d.head(3).to_string())"
```

What you should see:

```text
(1309, 10)
   survived  pclass     sex      age  sibsp  parch      fare embarked boat  body
0         1       1  female  29.0000      0      0  211.3375        S    2   NaN
1         1       1    male   0.9167      1      2  151.5500        S   11   NaN
2         0       1  female   2.0000      1      2  151.5500        S  NaN   NaN
```

Look at it: 1309 rows, 10 columns. `NaN` means "no value recorded". Now ask
SCMLE for a fact sheet about the table:

```bash runnable
scmle catalog titanic.csv --label-column survived --output catalog.json
```

What you should see: a block of JSON. Find these parts:

```text
"label": {"distribution": {"0": 809, "1": 500}, ...}
"duplicates": {"duplicate_fraction": 0.1787..., "duplicate_rows": 234, ...}
"missingness": {"by_field": {"age": 263, "boat": 823, "body": 1188, ...}}
```

What to look at:

* 809 died and 500 lived. That is not extreme, so a plain score is fine.
* 234 rows are exact copies of another row. Copies can make a model look
  better than it is, because a copy can sit in both the practice and the exam
  part. Remember this. SCMLE will warn you about it again.
* `boat` is missing for 823 people and `body` for 1188. These two columns look
  odd. Hold that thought.

What could go wrong: nothing here changes your data. The command only reads.

## Step 3. State the question before you run anything

Write these answers down. They are explained in [Decisions](DECISIONS.md).

| Question | Our answer | Why |
| --- | --- | --- |
| What do I predict? | `survived` | It is the outcome we care about |
| What is a good score? | ROC AUC | It measures how well the model puts survivors above non-survivors. 0.5 is a coin flip, 1.0 is perfect |
| When will I use the model? | Before the voyage, from the ticket | This decides which columns I may use |
| Which columns would I not know at that moment? | `boat` and `body` | Which lifeboat you got on, or whether your body was found, is known only after the ship sinks |

The last row is the most important one. A column that records something that
happens after the outcome gives the model the answer. That is called
**leakage**. A model with leakage scores wonderfully and then fails on real
use. You will see SCMLE flag it in Step 4.

## Step 4. Your first run finishes, and it warns

Run the simplest command, with only a short time limit and an output folder:

```bash runnable
scmle run titanic.csv --target survived --time-budget 120 --workers 2 --output-dir runs_first
echo "exit code: $?"
```

This takes two to four minutes. Near the start you will see one line that is
easy to miss, and the last lines look like this:

```text
leakage suspect(s) boat, body stay in the features; the test score is provisional until a person settles them (declare the column unavailable at prediction, or attest it)
...
run_20261105_001: selected bagged_logistic_regression (cv roc_auc=0.9917); test (PROVISIONAL: 2 flagged column(s) not attested: boat, body): roc_auc=0.9880, pr_auc=0.9692, log_loss=0.1188, f1=0.9574, balanced_accuracy=0.9628, accuracy=0.9657
artifacts: runs_first/run_20261105_001
```

and `exit code: 0`.

What to look at:

* The run **finished**. SCMLE did not stop you. It trained a model and
  reported the exam score.
* But the score is labelled `PROVISIONAL: 2 flagged column(s) not attested:
  boat, body`. "Flagged" means a statistical screen found those two columns
  suspiciously good at predicting the answer. "Not attested" means no person
  has said when their values are known.
* The exam ROC AUC is **0.988**. That is far too good for "who survived the
  Titanic, from the ticket". Do not trust it.

Why not trust it: `boat` (which lifeboat you got on) and `body` (whether your
body was found) are only known after the ship sinks. A model that uses them has
been given the answer. That is **leakage**. It looks brilliant today and fails
the first day it is used for real.

Why SCMLE warns instead of refusing. From the data alone, a leaked column and a
genuinely strong, legitimate column look identical. The screen has no way to
know when a value is recorded. If it refused every flagged column, it would also
refuse good analyses (it flags legitimate measurements on the breast-cancer
table). So it trains, says so loudly, labels the score provisional, and leaves
the decision to a person who knows when each value is known. The full
reasoning and the measured error rates are in [Leakage](LEAKAGE.md).

You settle a flag in one of two ways:

1. **The value is not known at prediction time**: declare it so. This is what
   we do next (Step 6). The score falls to the honest 0.885.
2. **The value is known at prediction time and a person vouches for it**: write
   an attestation (see [Leakage](LEAKAGE.md)).

If you would rather have a hard stop than a warning, add `--leakage-policy
block`. This is useful in regulated work. Try it:

```bash runnable
scmle run titanic.csv --target survived --leakage-policy block --time-budget 120 --workers 2 --output-dir runs_blocked
echo "exit code: $?"
```

What you should see (last two lines):

```text
[run_20261105_001] state.BLOCKED: unresolved leakage risk (certified screen, Benjamini-Yekutieli level 0.05): body, boat, boat, boat
blocked: unresolved leakage risk (certified screen, Benjamini-Yekutieli level 0.05): body, boat, boat, boat
```

and `exit code: 2`. `BLOCKED` is a safe stop, not a crash: no model was built.
`boat` appears three times because three different checks flagged it.

## Step 5. Plan (nothing is trained yet): see the evidence

```bash runnable
scmle plan titanic.csv --target survived
```

This takes about 7 seconds and trains nothing. It prints about 70 lines: each model SCMLE could
try and whether it will, then warnings. Here is the start and the end:

```text
Plan plan_3e14748f79: binary_classification, metric=roc_auc, validation=stratified, budget=1800s
Candidates: dummy, hist_gradient_boosting, decision_tree, logistic_regression, bagged_decision_tree, bagged_logistic_regression, knn, bagged_knn, extra_trees, random_forest, neural_network, svm
  ...
  WARNING: feature 'body' is a leakage suspect: share of one target class among the recorded values 1.000 ... This is a suspicion, not proof: a strong legitimate predictor looks the same without provenance; review before trusting
  WARNING: feature 'boat' is a leakage suspect: cross-fitted skill 0.973 ... This is a suspicion, not proof ...
```

What to look at:

* First line: `binary_classification` (two possible answers), `metric=roc_auc`
  (it picked a score for you), `validation=stratified` (the practice rounds
  keep the survivor share the same in each part), `budget=1800s` (it would
  search up to 30 minutes).
* `Candidates` lists the **models** it will try. `dummy` is a model that
  ignores the features. It is the floor every real model must beat.
* `[reject] ... missing dependency: lightgbm` lines are normal. Those models
  need optional extras you did not install.
* The two `WARNING` lines say `body` and `boat` look like leaks. SCMLE is
  telling you what you already suspected.

What could go wrong: if it picks the wrong metric or task, tell it with
`--metric` or `--task`. If the target is a column with many numbers, check
that it says `regression` and not `classification`.

## Step 6. Run for real

Tell SCMLE those two columns are only known after the outcome. This settles
the flag:

```bash runnable
scmle run titanic.csv --target survived \
  --unavailable-at-prediction boat body \
  --time-budget 120 --workers 2 --output-dir runs
```

This takes two to four minutes, depending on your computer. Lines scroll by
as it works. The ones that
matter:

```text
[run_20261105_001] state.PROFILING: profile dataset
[run_20261105_001] split.created: stratified split
[run_20261105_001] state.TRAINING: 7 candidates
[run_20261105_001] trial.ok: decision_tree
...
[run_20261105_001] state.SELECTING: best mean CV score; exact ties -> simpler_model; dummy baseline excluded when other models ran
[run_20261105_001] selection.frozen: stacking/default
[run_20261105_001] state.TESTING: final one-time test evaluation of stacking
[run_20261105_001] test.opened: locked test set released once, after selection was frozen
[run_20261105_001] state.COMPLETED: run complete
run_20261105_001: selected stacking (cv roc_auc=0.8482); test: roc_auc=0.8855, pr_auc=0.8539, log_loss=0.4081, f1=0.7738, balanced_accuracy=0.8137, accuracy=0.8241
artifacts: runs/run_20261105_001
```

What to look at, in the order the run does them:

1. `split.created`: SCMLE set some rows aside before looking at anything.
2. `trial.ok` lines: one practice model each. `trial.pruned` means it gave up
   early on a hopeless one. That is fine.
3. `selection.frozen`: the winner is chosen using only the practice rows.
4. `test.opened`: only now are the hidden rows used, and only once.
5. The last line: `cv roc_auc=0.8482` is the practice score and
   `test: roc_auc=0.8855` is the exam score. Trust the exam score.

What could go wrong:

* `state.BLOCKED` again: you forgot `--unavailable-at-prediction boat body`.
* `state.FAILED`: read the line after it. Run `scmle resume runs/run_...` to
  continue after an interruption (finished trials are kept).
* `all trials failed`, or a trial `exceeded its 6.0s hard time ceiling and was
  killed`: the time budget is too small for your computer. The first run on a
  new installation is the slowest. Raise `--time-budget` (try 300).
* It stops early: the time budget ran out. Raise `--time-budget`. A short
  budget searches fewer models and can leave some report parts empty (see
  [Reading results](READING_RESULTS.md)).

## Step 7. Read the result

SCMLE wrote a folder `runs/run_20261105_001`. The file `report.html` is the
full report. Open it in a browser. It starts with a long list of plan
decisions. Scroll to the "Leaderboard" heading. The same numbers are easier to
read with this short script. Save it as `read_results.py`:

```python runnable
import json
import pathlib

run = sorted(pathlib.Path("runs").glob("run_*"))[-1]
ev = json.loads((run / "evaluation.json").read_text())
print("run folder       :", run)
print("chosen model     :", ev["selected_model"])
print("metric           :", ev["metric"])
print(
    "practice score   :",
    round(ev["selected_cv_score"], 3),
    "+/-",
    round(ev["selected_cv_std"], 3),
)
low, high = ev["cv_confidence"]["interval"]
print("practice 95% band:", round(low, 3), "to", round(high, 3))
print("exam score       :", round(ev["test_metrics"]["roc_auc"], 3))
unc = json.loads((run / "diagnostics" / "metric_uncertainty.json").read_text())
print("exam 95% band    :", [round(x, 3) for x in unc["ci_95"]])
print("dummy baseline   :", ev["baseline_cv_score"])
print("fit health       :", ev["fit_health"]["verdict"])
print(
    "rows             :",
    ev["n_dev"],
    "to learn from,",
    ev["n_test"],
    "kept for the exam",
)
cm = json.loads(
    (run / "diagnostics" / "classification_report.json").read_text()
)
print("confusion matrix :", cm["confusion_matrix"])
imp = json.loads(
    (run / "diagnostics" / "permutation_importance.json").read_text()
)
if imp["truncated"]:
    print("importance       : not finished in time; raise --time-budget")
for row in imp["features"][:4]:
    print("importance       :", row["feature"], round(row["importance"], 3))
for item in json.loads(
    (run / "diagnostics" / "recommendations.json").read_text()
):
    print("recommendation   :", item["message"])
```

Run it:

```bash runnable
python read_results.py
```

What you should see:

```text
run folder       : runs/run_20261105_001
chosen model     : stacking
metric           : roc_auc
practice score   : 0.848 +/- 0.023
practice 95% band: 0.8 to 0.896
exam score       : 0.885
exam 95% band    : [0.841, 0.927]
dummy baseline   : 0.5
fit health       : healthy
rows             : 1093 to learn from, 216 kept for the exam
confusion matrix : [[113, 18], [20, 65]]
importance       : sex 0.242
importance       : pclass 0.074
importance       : age 0.039
importance       : fare 0.017
recommendation   : 14% of rows are exact duplicates; duplicates can appear in both training and validation folds and inflate scores. Deduplicate, or set data.group_column if rows belong to the same entity
```

Your numbers will not match these exactly, and that is normal. The search
stops when the time budget ends, so a slower or busier computer tries fewer
models. When the author repeated this page on a second, busier machine with the
same command, the winner was `bagged_logistic_regression` (practice 0.835, exam
0.869) instead of `stacking` (practice 0.848, exam 0.885), and the importance
rows were missing because the time ran out. What must hold on your machine: the
exam score is clearly above 0.5, it is close to the practice score, and
`fit health` says `healthy`. If the importance rows are missing, rerun with
`--time-budget 300`.

How to read it, line by line:

* **practice score 0.848** and **exam score 0.885** are close. Good. A big gap
  (practice far above exam) would mean the model memorised.
* **exam 95% band 0.841 to 0.927**: with only 216 exam rows, the true score
  could plausibly be anywhere in that band. Do not report "0.885" as if it were
  exact.
* **dummy baseline 0.5**: a model that ignores the data scores 0.5. Ours is
  far above, so it learned something.
* **fit health healthy**: what it learned in practice carried over.
* **confusion matrix**: rows are the truth (died, lived), columns are the guess.
  113 deaths guessed right, 18 wrongly guessed as lived; 65 survivors guessed
  right, 20 wrongly guessed as died.
* **importance**: shuffling `sex` hurts the score most. That fits the history
  (women were let into boats first). If the top feature were a surprise, you
  would investigate.
* **recommendation**: a warning that the table has copies. It is the same
  warning from Step 2. It means 0.885 may be a little optimistic. This is
  honest: the model's number is not perfect, and the tool says so.

If this were a leaked run, the exam score would be about 0.99 and the top
features would be `boat` or `body`. See [Reading results](READING_RESULTS.md).

What could go wrong: the list of `importance` rows is empty, or
`FileNotFoundError`. A short time budget can leave the importance table empty
(the file then says `"truncated": true`). Rerun with a larger `--time-budget`.

## Step 8. Save the model and predict

The model is inside `runs/run_.../model`. It is stored as a "bundle" (a folder
of files). Bundles can run code when opened, so SCMLE refuses to open one that
is not **signed**. A signature is a stamp made with a secret key that shows
the folder has not been changed. Pick a secret and keep it. Here we use a demo
one:

```bash
export SCMLE_SIGNING_KEY=demo-key-change-me
```

The run did not sign its bundle. Sign it by saving a copy. Save this as
`save_model.py`. The `if __name__` lines at the bottom are required (see the
note after):

```python runnable
import pathlib

import scmle


def main():
    run = sorted(pathlib.Path("runs").glob("run_*"))[-1]
    bundle = scmle.ModelBundle.load(run / "model", trusted=True)
    bundle.save("artifacts/titanic", signing_key="demo-key-change-me")
    print("saved artifacts/titanic")


if __name__ == "__main__":
    main()
```

```bash runnable
python save_model.py
ls artifacts/titanic
```

What you should see:

```text
saved artifacts/titanic
conformal.json
coverage_report.json
model_manifest.json
novelty.joblib
pipeline.joblib
reference.json
schema.json
signature.json
```

Notice the new file `signature.json`. (On a slower computer the list can lack
`conformal.json` and `coverage_report.json`; that is the time budget again.)

`trusted=True` means "I made this folder myself, open it." Never use it on a
folder from someone else.

Now make a small file of new passengers. They have the same columns as before,
minus `survived`, `boat` and `body`:

```bash runnable
python -c "
import pandas as pd
d = pd.read_csv('titanic.csv')
cols = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
pd.concat([d.iloc[100:105], d.iloc[800:805]])[cols].to_csv('new_passengers.csv', index=False)
"
scmle predict artifacts/titanic new_passengers.csv --output scored.csv
head -4 scored.csv
```

What you should see:

```text
wrote 10 predictions to scored.csv
prediction,proba_0,proba_1
0,0.5570198341867083,0.44298016581329164
0,0.657531739140053,0.34246826085994697
1,0.07917512598754506,0.9208248740124549
```

`prediction` is the guess (0 or 1). `proba_1` is the model's chance that the
passenger lived. A prediction of 1 with `proba_1` 0.92 is confident. A 0
with `proba_1` 0.44 is a near toss-up. Look at the probabilities, not only the
0 or 1.

Now see two refusals on purpose:

```bash runnable
SCMLE_SIGNING_KEY=wrong-key scmle predict artifacts/titanic new_passengers.csv
echo "exit code: $?"
printf 'class,sex\n1,m\n' > bad.csv
scmle predict artifacts/titanic bad.csv
echo "exit code: $?"
```

What you should see:

```text
artifact error: bundle signature is invalid: the bundle was modified or signed with a different key
exit code: 1
schema error: input is missing required columns: ['pclass', 'age', 'sibsp', 'parch', 'fare', 'embarked']
exit code: 1
```

The first says: wrong key, or someone changed the folder. The second says: the
columns you gave are not the ones the model was trained on. SCMLE never guesses
at column names. You can map names yourself with
`--rename INCOMING=EXPECTED`.

What could go wrong: forgetting the `export` line gives
`refusing to load an unauthenticated bundle`. Run the `export` again in this
terminal.

A note on the `if __name__ == "__main__":` lines. SCMLE trains models in helper
processes. On macOS and Windows, a script without that guard makes the helper
re-run your whole script, and you get a long error about "bootstrapping phase".
Always keep the guard in scripts that call `run()`.

## Step 9. Package, then serve

To let other programs ask for predictions, run the model as a small web
service. Start it (leave this terminal open; open a second one for the next
commands, activate the environment there too, and `export` the key again):

```bash runnable
scmle serve artifacts/titanic --port 8765
```

What you should see:

```text
serving stacking on http://127.0.0.1:8765
```

In the second terminal:

```bash runnable
curl -s localhost:8765/livez
curl -s localhost:8765/readyz
```

What you should see:

```text
{"status": "alive", "request_id": "srv-ae7f7cff553540149e0b6af82c622a08"}
{"ready": true, "phase": "ready", "completed": {"verifying": 0.0, "loading": 0.0, "warming": 0.2252}, ...}
```

`livez` means "the program is running". `readyz` means "the model is checked
and loaded and can answer". They are different on purpose. If `readyz` says
`ready: false`, read its `reason`; see
[When things go wrong](WHEN_THINGS_GO_WRONG.md).

Ask for a prediction:

```bash runnable
curl -s -X POST localhost:8765/predict -H 'Content-Type: application/json' \
  -d '{"records": [{"pclass": 1, "sex": "male", "age": 49.0, "sibsp": 1, "parch": 0, "fare": 56.9292, "embarked": "C"}]}'
```

What you should see: JSON with `"predictions": [0]` and `"probabilities"`
holding two numbers, plus `"model_version": "v1"` and an `artifact_digest`.
The digest is a fingerprint of the exact model that answered. Write it down in
any report you make from these predictions.

Stop the service with Ctrl-C in the first terminal.

Two safety rules to see once:

```bash runnable
scmle serve artifacts/titanic --host 0.0.0.0 --port 8766
```

What you should see:

```text
config error: refusing to listen on '0.0.0.0' without authentication: set SCMLE_AUTH_KEYS_FILE (or SCMLE_AUTH_HS256_SECRET) with SCMLE_AUTH_ISSUER and SCMLE_AUTH_AUDIENCE, or bind a loopback address
```

The reference server has no login and no encryption. It only listens on your
own machine unless you set up tokens (see
[`TENANT_ISOLATION.md`](TENANT_ISOLATION.md)). Never put it on the internet as
it is.

To ship to a real server you build a container. `scmle package` writes the
folder for that:

```bash runnable
scmle package artifacts/titanic --output service
ls -A service
```

What you should see (the listing may be laid out in columns):

```text
wrote service folder service
.dockerignore
Dockerfile
README.md
build.sh
environment.lock
model
provenance.json
requirements.lock
requirements.txt
sbom.cdx.json
serve.py
vuln-policy.json
```

What these are:

* `Dockerfile` and `build.sh` build the container image. You need Docker and
  network access to run them. The author did **not** build an image for this
  page.
* `sbom.cdx.json` is a **software bill of materials**: a list of every
  library inside, so a security tool can check for known flaws.
* `provenance.json` is a signed statement of what the package was made from.
  The service checks it when it starts.

The cloud steps after this (release gate, live checks, promotion to AWS or
Google Cloud) are in [`CLOUD_RELEASE.md`](CLOUD_RELEASE.md). They have not
been run against a real cloud account. See
[What SCMLE cannot do](WHAT_SCMLE_CANNOT_DO.md).

## Step 10. Monitor

A model is correct only about the world it was trained on. When the world
changes, it gets worse, quietly. **Monitoring** compares new data and new
outcomes with what the model saw in training.

For this walk-through we pretend a batch of rows is new. Real monitoring uses
rows that arrived after training. Our pretend rows were seen during training, so
the performance number below is too good. What matters is how the alarm
behaves. Save as `monitor_model.py`:

```python runnable
import pandas as pd

import scmle
from scmle.monitoring import monitor


def main():
    data = pd.read_csv("titanic.csv").drop(columns=["boat", "body"])
    bundle = scmle.ModelBundle.load(
        "artifacts/titanic", signing_key="demo-key-change-me"
    )
    watcher = monitor.ModelMonitor(bundle)

    batch = data.sample(300, random_state=0)
    report = watcher.assess(batch.drop(columns=["survived"]), batch["survived"])
    print(
        "normal batch :",
        report.recommendation.action,
        report.recommendation.reasons,
    )

    changed = batch.copy()
    changed["fare"] = changed["fare"] * 8
    changed["pclass"] = 3
    changed["age"] = changed["age"] + 30
    report = watcher.assess(
        changed.drop(columns=["survived"]), changed["survived"]
    )
    print("changed batch:", report.recommendation.action)
    for reason in report.recommendation.reasons:
        print("   -", reason)


if __name__ == "__main__":
    main()
```

```bash runnable
python monitor_model.py
```

What you should see (the changed batch is built by hand to look different):

```text
normal batch : none []
changed batch: retrain
   - 43% of features drifted strongly: 'fare' (PSI 6.32), 'age' (PSI 4.77), 'pclass' (PSI 3.66)
   - model output drifted strongly (PSI 0.72)
```

The feature numbers repeat on any computer. The output number (0.72 here, 0.91
on the author's second machine) depends on which model won.

How to read it:

* The recommendation is one of three words: `none` (all is well),
  `investigate` (look, something moved), `retrain` (the model is clearly
  stale).
* **PSI** is a number for "how different is this column from training".
  Under about 0.1 is calm. Above about 0.25 is a strong shift.
* **Drift** means the incoming data stopped looking like the training data.
* A drift alarm is not always a real problem. In the author's first try, a
  "fresh" batch was the last 509 rows of the file, which is sorted by
  passenger class, so it was nearly all third class and the monitor said
  `investigate`. The alarm was right: that batch really was different. The cure
  was a better sample, not a new model.

What could go wrong: `labeled_fraction` low. Performance is judged only on rows
whose true outcome has arrived. If most outcomes are still unknown, trust the
drift numbers and be careful with the performance numbers.

## Step 11. Register, promote and roll back

A **registry** is a shelf of model versions with labels. The label
`champion` marks the version in use. To move the label you need an
**approval**: a signed note by a person who is not the one who built the model.
Promotion without it is refused, on purpose.

To practice a rollback, first make a deliberately bad second model: the same
data with the answers shuffled. Save as `bad_model.py`:

```python runnable
import pandas as pd

import scmle


def main():
    data = pd.read_csv("titanic.csv")
    data["survived"] = (
        data["survived"].sample(frac=1, random_state=1).to_numpy()
    )
    project = scmle.MLProject(
        data,
        target="survived",
        unavailable_at_prediction=["boat", "body"],
        time_budget=120,
        workers=2,
        output_dir="runs_bad",
    )
    result = project.run()
    print(result.summary())
    result.save("artifacts/titanic_v2", signing_key="demo-key-change-me")


if __name__ == "__main__":
    main()
```

```bash runnable
python bad_model.py
```

What you should see (one to three minutes; the numbers will differ on your
computer, but the exam score will sit near 0.5):

```text
run_20261105_001: selected hist_gradient_boosting (cv roc_auc=0.5242); test: roc_auc=0.4974, pr_auc=0.3856, log_loss=0.7885, f1=0.3194, balanced_accuracy=0.4828, accuracy=0.5288
```

A score of 0.50 is a coin flip. This is what "the model learned nothing"
looks like. Now register both and try to promote. Save as `ship_model.py`:

```python runnable
from scmle.governance import registry

KEY = "demo-key-change-me"


def approval_for(record, why):
    return registry.PromotionApproval(
        model="titanic",
        version=record.version,
        content_id=record.content_id,
        reviewer="you@example.com",
        decision=registry.PromotionDecision.APPROVED,
        reason=why,
        evidence=("runs/report.html",),
        approved_at="2026-11-05T00:00:00+00:00",
    ).sign(KEY)


def main():
    store = registry.ModelRegistryStore("registry")
    v1 = store.register("titanic", "artifacts/titanic", signing_key=KEY)
    v2 = store.register("titanic", "artifacts/titanic_v2", signing_key=KEY)
    print("versions:", [record.version for record in store.versions("titanic")])
    try:
        store.promote("titanic", 1)
    except Exception as error:
        print("without approval:", error)
    store.promote(
        "titanic",
        1,
        approval=approval_for(v1, "first release"),
        signing_key=KEY,
    )
    print("after promoting v1:", store.aliases("titanic"))
    store.promote(
        "titanic",
        2,
        approval=approval_for(v2, "second release"),
        signing_key=KEY,
    )
    print("after promoting v2:", store.aliases("titanic"))
    previous = store.resolve("titanic", "previous")
    note = approval_for(previous, "roll back: v2 is worse")
    store.promote("titanic", previous.version, approval=note, signing_key=KEY)
    print("after rolling back:", store.aliases("titanic"))


if __name__ == "__main__":
    main()
```

```bash runnable
python ship_model.py
```

What you should see:

```text
versions: [1, 2]
without approval: champion promotion requires independent approval
after promoting v1: {'champion': 1}
after promoting v2: {'champion': 2, 'previous': 1}
after rolling back: {'champion': 1, 'previous': 2}
```

How to read it: `champion` is the version being served. Promoting v2 kept v1
under the label `previous`, so going back is one step. After the rollback,
version 1 is champion again.

What you should know that this demo skips: a real approval is written by a
second person, with real evidence files, using a key you keep secret. The demo
signs its own approval, which proves the mechanism, not the independence.
The registry also has `LifecycleManager.check_rollback`, which compares the two
versions on fresh labeled rows and rolls back only if the new one is clearly
worse. See [`TUTORIAL.md`](TUTORIAL.md) section 7.

## Step 12. Where to go next

* A beginner trap list: [Decisions](DECISIONS.md).
* Understand every number you saw: [Reading results](READING_RESULTS.md).
* A refusal you do not understand: [When things go wrong](WHEN_THINGS_GO_WRONG.md).
* The "charter" and the claims it produces (the careful, pre-registered mode
  of working): see "A second mode: the charter" in
  [Reading results](READING_RESULTS.md).
* A "did the discount cause the sales" question: a causal claim stays
  `unverified` until the charter names the **attestors** (the people who vouch
  for each assumption the design needs) and their signed statements verify;
  see "causal claim" and "attestation (attestor)" in
  [the glossary](GLOSSARY.md).
* What this tool does not do: [What SCMLE cannot do](WHAT_SCMLE_CANNOT_DO.md).
* Why a flag is a suspicion and not a verdict, with measured error rates:
  [Leakage](LEAKAGE.md).
* All the options: [`CLI.md`](CLI.md) and [`TUTORIAL.md`](TUTORIAL.md).
