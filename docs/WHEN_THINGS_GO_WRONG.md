# When things go wrong

SCMLE refuses, blocks and warns on purpose. A refusal is the tool doing its
job. This page lists what you can hit, in the order you are likely to meet it.
For each: what you see, what it means, why it exists, and what to do.

How sure am I about each entry? Each message is marked:

* **Seen**: the author triggered it and copied the real output.
* **From the code**: the author found the message in the source and did not
  trigger it. The wording is exact, but nobody has watched it happen for this
  page.

Words you do not know are in the [glossary](GLOSSARY.md).

## First, the exit code

| Exit code | Meaning |
| --- | --- |
| 0 | it worked |
| 1 | something failed: a bad option, bad data, a bad file, a refused bundle, a failed check |
| 2 | **blocked**: SCMLE stopped on purpose before doing harm; also `deploy gate` refusing a release and `frame` refusing a draft |

Check with `echo $?` right after the command. Messages begin with
`blocked:` or with a category such as `config error:`, `data error:`,
`schema error:`, `artifact error:`.

## Contents

* [Setup and scripts](#setup-and-scripts)
* [A run is blocked](#a-run-is-blocked)
* [A run fails or is interrupted](#a-run-fails-or-is-interrupted)
* [Results that need care](#results-that-need-care)
* [Unverified: what to do](#unverified-what-to-do)
* [Charter and claims](#charter-and-claims)
* [Bundles and predictions](#bundles-and-predictions)
* [The service](#the-service)
* [Registry, promotion and rollback](#registry-promotion-and-rollback)
* [Monitoring alarms](#monitoring-alarms)
* [The closed loop and escalations](#the-closed-loop-and-escalations)
* [Release gate and live checks](#release-gate-and-live-checks)

## Setup and scripts

### `scmle: command not found`

**What it means.** The Python environment with SCMLE in it is not active.
**What to do.** Run `source .venv/bin/activate` in the folder where you
installed it. Every new terminal needs it.

### "An attempt has been made to start a new process before the current process has finished its bootstrapping phase" (Seen)

**What it means.** Your Python script calls `run()` but is missing the guard
`if __name__ == "__main__":`. SCMLE trains in helper processes. On macOS and
Windows each helper starts by importing your script, which runs the training
again.
**What to do.** Put your code in a `main()` function and end the file with:

```python
if __name__ == "__main__":
    main()
```

The run may still finish but with a long, alarming trace. The command-line
tool does not have this problem.

## A run is blocked

A blocked run exits with code 2, writes `state.BLOCKED` in its event log, and
leaves the sealed test rows unopened. Nothing is broken. You can read why:

```bash
tail -n 1 runs/run_*/events.jsonl
```

### Leakage block (Seen)

```text
blocked: unresolved leakage risk (certified screen, Benjamini-Yekutieli level 0.05): body, boat, boat, boat
```

**What it means.** You used `--leakage-policy block` (or `leakage_policy=
"block"`), and a statistical screen found columns that predict the target
suspiciously well, or whose "is it recorded?" pattern gives the answer away.
The names are the suspects. A name repeats when more than one test flagged it.
**Why it exists.** A leaked column makes a model look wonderful and then fail.
The default setting only warns, which is easy to miss; `block` makes it a stop.
Only checks with a measured error rate (a certificate) are allowed to block.
**What to do.** Ask for each suspect: "Would I know this value at the moment I
use the model?"

* **No**: exclude it with `--unavailable-at-prediction COL ...` (or
  `--exclude COL ...`) and run again.
* **Yes, and I can say who confirms it**: in a charter run you can keep it by
  giving an attestation file (see [Unverified: what to do](#unverified-what-to-do)).
* **I do not know**: ask someone who does. Do not guess.

The screen gives a *suspicion*, not proof. A truly strong feature looks the
same.

### Other blocks (From the code)

Each is exit code 2 and appears as `blocked: <message>`.

| Message (shortened) | What it means | What to do |
| --- | --- | --- |
| `target '<t>' has N missing values; set data.unlabeled to 'ignore' or 'pseudo_label' ...` | Some rows have no answer | Remove them, or pass `--unlabeled ignore` (set aside) or `pseudo_label` (classification only) |
| `target ... has N distinct value(s); there is nothing to learn` | The target is constant | Pick another target, or get more varied data |
| `target ... looks like an identifier or free text, not a class label; choose another target` | Almost every row has a unique value | Choose the right column |
| `... classification vs regression is ambiguous; set task= explicitly` | A small whole number could be a class or a count | Pass `--task` |
| `binary_classification requested but target has N classes` | `--task` does not match the data | Fix `--task` |
| `a class has fewer than 2 examples: cannot stratify` / `insufficient minority-class examples after test split` | A class is too rare to split | Get more rows of the rare class, or change the validation |
| `need at least N groups for group validation` | Too few groups for the number of folds | Lower `--folds` or add groups |
| `group column '<c>' has missing values` | Rows with no group | Fill the group column |
| `no group-disjoint split keeps every class in every fold; the classes are concentrated in too few groups` | A class lives in few groups | More groups, or fewer folds |
| `time column '<c>' has unparseable/missing values: ordering ambiguous` | Dates that cannot be read | Fix the dates or change `--time-column` |
| `too few rows for temporal cross-validation` / `only N rolling window(s) fit ...` | Not enough history | More data, or smaller `--horizon-rows`, `--window-rows`, `--embargo-rows` |
| `no usable feature columns remain after exclusions` | You excluded everything | Reduce `--exclude` |
| `monotonic constraints need numeric feature columns ...` / `... not multiclass` / `no available model can enforce the monotonic constraints; install the 'boosting' extra ...` | Your `--monotonic` rule cannot be met | Fix the rule, or `pip install "scmle[boosting]"` |
| `none of the top N candidates satisfies the serving constraints; relax the limits or allow more candidates` | Your model-size or latency limit is impossible | Relax `max_model_mb` or `max_p95_latency_ms`, or raise `--top-models` |
| `PLANNING not approved by <reviewer>: <reason>` | A human approval checkpoint said no (Python `checkpoints=`) | Address the reason; run again |
| `unresolved blocking curation findings` | The data checks found a blocking issue | Read the finding ids in the run's events and fix the data |
| `admissibility quarantined every feature` (charter runs) | Every feature was kept out | Provide provenance or attestations |
| `the charter unit '<u>' does not match the grain of the target table` | Your table has a different unit than the charter says | Fix the charter `unit` or the table |

## A run fails or is interrupted

### `state.FAILED` or `cancelled error: interrupted`

**What it means.** A run died (a crash, a Ctrl-C, a lost process, or an
unexpected error shown as `internal: ...`).
**Why it matters.** Finished trials are saved. You do not have to start over.
**What to do.** `scmle resume runs/run_YYYYMMDD_NNN`. It reuses finished trials.

Resume itself can refuse (From the code):

* `cannot resume: the data changed since the checkpointed trials were
  computed`: you changed the file. Start a fresh run.
* `source schema changed incompatibly while resuming`: same cure.
* `artifact error: run directory not found: <path>` (Seen): wrong path.

### `config error: ...`

Exit code 1. A setting is invalid. Common ones (From the code):

* `unknown config sections [...]; allowed: [...]`: a typo in the `--config`
  JSON.
* `invalid project specification: ...`: a value out of range. The text names
  the field.
* `success criteria name unreported metrics [...]`: a metric name that is not
  reported for this task.
* `the run specification sets X=A but the charter fixes B; change the
  specification or re-sign the charter`: a run option contradicts the charter.
* `data error: target column 'x' not found; columns: [...]` (Seen): spelling.
* `data error: group column 'x' not found; columns: [...]`: same, for the group,
  time or weight column.

### `all trials failed` / `TrialLimitError: trial exceeded its 6.0s hard time ceiling and was killed` (Seen)

**What it means.** Each trial may run only until the end of the time budget plus
a 10 percent grace. When the earlier steps (reading, splitting and checking
the data) used up the budget, only the grace is left: 6 seconds for a 60 second
budget. Every trial passed that and was stopped. The author met this on a busy
computer, on a first run, with `--time-budget 60`.
**Why it exists.** A limit that is not enforced is not a limit; a runaway model
must not hold the computer.
**What to do.** Raise `--time-budget` (120 to 300 is comfortable for a small
table). Look at `runs/run_*/trials/trial_001.json` to see the error. A second
run on the same machine was faster because the libraries were already loaded.

### The run finished but part of the report is empty

A short `--time-budget` can leave the importance table empty, or print
`conformal.skipped: ... time budget exhausted`. The file says `"truncated":
true`. Raise `--time-budget` and run again.

## Results that need care

### Release evaluation says `policy_not_declared` (From the code)

**What it means.** You did not declare what a good enough model is, so the
release check is `blocked: true`.
**Why it exists.** The tool will not invent a passing mark from the scores it
just saw.
**What to do.** Declare success criteria in the `problem` section of your
`--config` file, such as `{"problem": {"success_criteria": {"roc_auc": 0.8}}}`
(set the number before you look at results).

### A very high score (Seen)

An exam ROC AUC of 0.988 on a real table is a warning, not a prize. See
[Reading results](READING_RESULTS.md). Nothing "fails"; the tool completes.

### A drift or shift report says `unverifiable` (Seen)

**What it means.** SCMLE had no sample of the population you will deploy to, so
it cannot say whether your data resembles it. It will never say `pass` without
evidence.
**What to do.** Supply a deployment sample in the charter's
`deployment_population` reference, or accept that this check was not done.

### Stability says `not_evaluated` (Seen)

**What it means.** The charter listed no alternative reasonable choices (for
example how to fill gaps), so there is nothing to test the robustness against.
**What to do.** Declare more `multiverse` choices in the charter, if you want
that evidence.

## Unverified: what to do

`unverified` is a tier, not an error. It means: "I am relying on a default, and
no person has checked it." It never blocks a run by itself. Whether it
matters is up to you. Here is each place you will meet it.

### 1. A feature is `unverified` (Seen)

In `admissibility_table.json`, every feature of the Titanic charter run looked
like this:

```json
{"name": "sex", "admission": "admit_flagged", "tier": "unverified",
 "reasons": ["availability at serving is unknown", "as-of rule is unverified"]}
```

**What it means.** Nobody told SCMLE when this column is filled in. It is
allowed (`admit_flagged`), with a note.
**Why it exists.** SCMLE cannot know from the numbers alone whether a column
exists at prediction time. It says so rather than assume.
**What to do.**

* For a first model: nothing. Check with your own eyes that each column is
  something you will have. Keep that as a written decision.
* To record that a person vouches for a strong predictor that the screen
  flagged (the `leakage suspect` case), write an attestation file and pass it
  with `--leakage-attestations FILE` in a charter run. The file is JSON:

```json
{
  "format_version": 1,
  "attestations": [
    {
      "column": "days_since_signup",
      "name_or_id": "A. Owner",
      "role": "data owner",
      "statement": "Recorded at signup, always before the prediction moment.",
      "evidence": "ticket DATA-7",
      "date": "2026-11-05",
      "expires": "2027-11-05",
      "scope": "<the charter_hash from charter.json>",
      "lead_seconds": 0
    }
  ]
}
```

All nine keys are required. `scope` must be your charter's hash, so one
attestation cannot be reused for another analysis. `expires` must not be before
`date`. One attestation per column. A valid one keeps the column, marks it
`attested` (never `confirmed`), and writes a diagnostic row in the ledger. A
malformed file gives `config error: invalid leakage attestations: ...`
(Seen as a traceback in an earlier version; now a clean error). If the scope or
dates do not fit, the run notes
`attestation for '<col>' ignored: ...` and the column stays quarantined.

The shape comes from the code and the repository's own test of it; the author
did not run a full charter run with an attestation for this page.

### 2. A claim is `unverified` (From the code)

A claim in a charter run stays `unverified` when:

* `no hypothesis survived exploration; C was not opened`: there was nothing to
  confirm. Not a failure. Try other ideas on `X` or accept that the data
  shows nothing.
* `no negative control was recorded; no claim is promoted`: the check that
  shuffles the target did not run. Re-run with a charter that has the
  permutation count set.
* `findings do not clearly exceed the false-discovery floor: N discoveries,
  95th percentile of B permuted_target re-runs is P`: the real search found no
  more than random labels did. The "discoveries" could be noise. Treat them as
  hunches.
* The confirmation set was too small (`feasibility` = `downgrade` or
  `infeasible`): scope becomes `exploratory_unconfirmed`. Get more rows or
  lower your promise (`min_effect`).

### 3. A cost is `unverified`

Costs that no person supplied (for example, derived from the data) leave a
decision claim `unverified`. **What to do.** Have a person state the costs and
put them in the charter's cost model.

### 4. A drafted field is `unverified`

Fields drafted by a language model with `scmle frame` are `unverified` until a
person signs the charter. **What to do.** Read the draft, fix it, and run
`scmle charter sign`. A replayed draft with no calibrated `tau` is refused.

### 5. A causal claim is `unverified` or `not_identified` (From the code)

* `no eligible identification strategy`: nothing in your charter lets the
  effect be estimated. The claim is `not_identified`, with `action:
  refuse_claim` and no estimate. See `identification_report.json` for what is
  missing.
* `no named attestor for: ...`: each assumption needs a named person.
* `a design diagnostic failed` / `a falsification test rejected`: the
  data contradict the design.

A held-out set alone never confirms a causal effect.

### 6. An escalation was not answered

If nobody replies in time, the safe default is applied and the outcome is
recorded as `unverified`. See [The closed loop](#the-closed-loop-and-escalations).

## Charter and claims

### `config error: invalid analysis charter: population must not be empty` (Seen)

**What it means.** You ran `scmle charter check` on the fresh template. It is
meant to fail until you fill it in.
**What to do.** Fill in `question`, `estimand.population`, `estimand.target`,
`estimand.prediction_moment`, `estimand.summary_measure`, at least one
`estimand.intercurrent_events` item as `{"event": "...", "strategy": "..."}`,
`unit` and `deployment_population`.

### `config error: invalid analysis charter: unexpected field(s): ['description']` (Seen)

An event needs the keys `event` and `strategy`, not anything else. The error
names the extra field.

### `config error: charter.json exists; pass --force` (Seen)

`charter init` will not overwrite your work. Use a new name or `--force`.

### `artifact error: charter hash mismatch in <file>: the file was edited` (Seen)

**What it means.** Someone changed the charter file after it was hashed or
signed, even by one letter.
**Why it exists.** The whole point is that the plan cannot be quietly changed
after seeing results.
**What to do.** If the edit was a mistake, restore the original. If it was
deliberate, make a new charter and sign it again. A loosened plan should look
like a new plan.

### Other charter refusals (From the code)

* `charter signature does not verify for <file>`: wrong key or edited file.
* `no signing key: set SCMLE_SIGNING_KEY or pass --key-file` (Seen).
* `an amendment may only tighten the budget`: you may lower the error
  allowance, never raise it.

### `feasibility: downgrade (273 rows < 333 needed for power 0.8 at effect 0.2; detectable effect is 0.221)` (Seen)

**What it means.** Your sealed confirmation set is too small to find the effect
size you promised.
**Why it exists.** A test with too little data would usually find nothing even
when something is there, so the claim could not be called confirmed.
**What to do.** Add rows, or lower your ambition (`min_effect` 0.25 made the
same data feasible: `feasible (273 rows >= 214 needed for power 0.8)`). The run
still completes when downgraded; the scope is `exploratory_unconfirmed` and no
claim is called confirmed. `infeasible` (under 30 confirmation rows) carves no
confirmation set at all.

### `scmle claims verify` shows `FAIL` (From the code, except the missing case)

**What it means.** The run's files no longer agree: the ledger chain, the
claims file, the charter, the custody log, or the negative control. Any
`FAIL` makes the command exit 1.
**Why it exists.** It is the audit. It proves the artifacts agree with each
other, not that the statistics are right, and not that nobody rewrote every
file consistently.
**What to do.** Read the line: it names the step and detail. A missing ledger
looks like (Seen):

```text
FAIL    ledger chain verifies and decisions replay: ledger nodir/hypothesis_ledger.jsonl does not exist
0 of 1 steps passed; verification FAILED
```

Treat any `FAIL` on a run you intend to rely on as a reason not to rely on it
until explained. `SKIPPED` is not a failure; the step did not apply.

## Bundles and predictions

### `artifact error: refusing to load an unauthenticated bundle ...` (Seen)

**What it means.** The model folder has no valid signature, and a bundle is a
program that runs when opened.
**What to do.** Set `SCMLE_SIGNING_KEY` to the key the bundle was signed with. If
the bundle is unsigned and you made it yourself, sign it (see
[Start here](START_HERE.md) Step 8). `--trusted` skips the check; use it only
for a bundle you made on this machine.

### `artifact error: bundle signature is invalid: the bundle was modified or signed with a different key` (Seen)

**What it means.** Wrong key, or someone changed the folder.
**What to do.** Check the key. If the key is right, treat the bundle as
tampered with and rebuild it from the run. An invalid signature is refused
even with `--trusted`.

### `artifact error: trusted loading is refused when SCMLE_ENV=production` (Seen)

**What it means.** In production mode `--trusted` is never allowed.
**What to do.** Sign the bundle and supply the key.

### `checksum mismatch: model file was modified or corrupted` (From the code)

A file in the bundle no longer matches its recorded fingerprint. Restore the
bundle from a good copy.

### `bundle was built with different dependencies: scikit-learn X (installed Y) ...` (From the code)

Models can behave differently under a different library version, so SCMLE
refuses. Install the versions the bundle was built with, or retrain.

### `schema error: input is missing required columns: [...]` (Seen)

**What it means.** The file you gave has different columns than the model was
trained on.
**What to do.** Supply the columns named. If yours are called something else,
map them: `--rename INCOMING=EXPECTED`, `--case-insensitive`,
`--strip-whitespace`. Mapping is always explicit; SCMLE does not guess.

### `non-numeric values in numeric columns` (From the code)

A numeric column holds text. Clean the file.

## The service

### `config error: refusing to listen on '0.0.0.0' without authentication ...` (Seen)

**What it means.** You asked the reference server to accept connections from
other machines but configured no login.
**Why it exists.** The server has no encryption and no accounts of its own.
**What to do.** Bind to loopback (the default) for local work. For anything
else, set `SCMLE_AUTH_KEYS_FILE` (or `SCMLE_AUTH_HS256_SECRET`) with
`SCMLE_AUTH_ISSUER` and `SCMLE_AUTH_AUDIENCE`, and put it behind a gateway that
provides TLS ([`TENANT_ISOLATION.md`](TENANT_ISOLATION.md)).

### `/readyz` answers 503 or `"ready": false`

**What it means.** The service is alive but not ready. `/livez` still says 200.
The body shows a `phase` (`starting`, `verifying`, `loading`, `warming`,
`ready`, `failed`) and a `reason`.
**Why it exists.** A service that failed verification must stay out of use but
stay alive so you can read why.
**What to do.** Read `reason`. Typical causes: an unsigned or modified bundle,
a wrong key, `SCMLE_ENV=production` with the published demo key
(`the signing key is the published demo fixture; configure a managed key`).
Fix the cause and restart. Remember `/livez` is "the process runs" and
`/readyz` is "the model is checked and loaded".

### Request errors (From the code, one was Seen)

| Status | Meaning | What to do |
| --- | --- | --- |
| 400 | malformed JSON or unknown request field | fix the request |
| 401 | no valid token (`authentication required`) | send `Authorization: Bearer <token>` |
| 403 | the token lacks the scope or tenant | ask for `model:predict` |
| 409 `schema_change` | you sent a `schema_version` or `expected_schema_digest` that does not match the model | refresh your schema from `/schema` |
| 413 | body over 16 MiB or over 10,000 records | send fewer |
| 415 | not `Content-Type: application/json` | set the header |
| 422 `schema` | missing or non-numeric features (Seen: `records are missing required features: [...]`) | send every feature; send `null` for unknown, never omit |
| 422 `data_policy` | nulls or unseen categories, when the policy is `reject` | clean the input or relax the policy |
| 429 | too many requests in flight or too many from you | wait `Retry-After` seconds |
| 503 | not ready | see above |
| 504 | scoring took over 30 seconds | send fewer records |

## Registry, promotion and rollback

### `champion promotion requires independent approval` (Seen)

**What it means.** You tried to make a version the live one without a signed
approval.
**Why it exists.** One person must not push an unreviewed model into use.
**What to do.** Have a reviewer create and sign a `PromotionApproval` bound to
that version's exact content (see [Start here](START_HERE.md) Step 11). Other
refusals of the same family (From the code): `promotion approval does not match
bundle identity`, `promotion was rejected by the reviewer`, `champion promotion
requires independent review`, `champion promotion requires a valid signature`.

### `AttestationRequired`: `move_champion_or_roll_back (never-list item 1) needs a valid human attestation; escalation esc-N is pending` (From the code)

**What it means.** The registry was given a guard that insists a named person
decide before the live model moves.
**What to do.** Answer the escalation (see
[the closed loop](#the-closed-loop-and-escalations)).

### A challenger was staged but not promoted

The champion and challenger are compared on the same fresh labeled rows. (The
closed loop uses e-values for this: an evidence score that stays honest however
often you look.)

The challenger is promoted only if the lower end of its 95 percent advantage
interval is above zero; otherwise the champion is kept
(`challenger was not clearly better; champion kept`). Even a winner is only
staged: `staged challenger as vN; independent approval required before
champion promotion`. Nothing is wrong; follow the approval step.
`need at least 30 labeled rows to compare models` means too few fresh rows.

### Rollback did not happen (From the code)

`RollbackOutcome.reason` tells you why:

* `no previous version`: there is nothing to go back to.
* `fewer than N labeled rows` (default 100): not enough fresh outcomes.
* `the new champion holds up`: it is not clearly worse (the whole interval must
  be below minus the tolerance, 0.01 by default).
* `rollback requires independent approval`: provide a signed approval for the
  previous version.

A manual rollback is the same promotion in the other direction (Start here
Step 11) and needs the same approval.

## Monitoring alarms

The monitor returns `none`, `investigate` or `retrain` and a list of reasons
(From the code; defaults: moderate PSI 0.1, high PSI 0.25, a share of
features at high drift 0.3, unseen categories 20 percent, 30 labeled rows
needed, performance tolerance 10 percent).

| Recommendation | Triggered by | What to do |
| --- | --- | --- |
| `investigate` | one or a few features moved, or the output moved moderately, or 20 percent or more unseen categories | look at the named features; is the data feed broken, or did the world change? |
| `retrain` | the main metric is more than 10 percent worse on labeled rows; or 30 percent or more of features drifted strongly; or the output drifted strongly and there are no labels | retrain on recent data; compare the challenger with the champion on fresh labeled rows; promote with approval |

Seen: a batch with fare ×8, class 3 for everyone, and age +30 gave `retrain`
with reasons `43% of features drifted strongly: 'fare' (PSI 6.32), 'age' (PSI
4.77), 'pclass' (PSI 3.66)` and `model output drifted strongly (PSI 0.72)`.

Before you act on an alarm, check the batch itself. In the author's first try,
an `investigate` alarm came from a sample sorted by passenger class: the data
really was different, because the sample was unrepresentative. The cure was a
better sample, not a retrain.

`insufficient_labels` (From the code): fewer than 30 matured labeled rows.
No performance judgement is made. Wait for outcomes, and trust only the drift
numbers meanwhile.

## The closed loop and escalations

The closed loop is a set of commands (`scmle loop ...`) that watch a deployed
model batch by batch. When it is unsure, it stops and asks a person. This is an
**escalation**. These are the real commands and outputs, using a made-up
stream of cases (the author wrote the files by hand; see "What the loop needs").

```bash
scmle loop init deploy_c --policy policy.json --charter-hash <charter-hash>
scmle loop observe deploy_c --batch b0.csv
scmle loop mature deploy_c --labels l0.csv
scmle loop step deploy_c
scmle loop pending deploy_c
```

### What the loop needs

* `policy.json`: at minimum `{"reference_risk": 0.1}` (the loss rate you expect
  when all is well). The author used also `"min_matured": 5, "outcome_window":
  0` for a quick test. Everything else has a default.
* A batch file: columns `case_id`, `prediction`, `decision`. Optional columns:
  `set_size`, `score`, `novel`, `under_covered`, `propensity`, `audit`,
  `bypass`.
* A labels file: columns `case_id` and `loss` (a number from 0 up to the loss
  bound). A file without `case_id` gives a raw `KeyError` (From the code).

### Seen: an alarm and an escalation

After three batches whose losses were all 1.0 against an expected 0.1:

```text
step 2:
{"status": "ok", "mechanism": "complete", "events": [["alarm", "mean loss above reference plus tolerance"], ["escalation", ""]], "escalation": "esc-5"}
```

```text
[
  {
    "id": "esc-5",
    "question": {
      "action": "l1_alarm_investigate",
      "detail": {"batch": 2, "reason": "alarm is not certified"},
      "scope": "stream"
    },
    "question_hash": "d1044b617af39a7b0dced0edfa0476b94465c92333b2d60872a2e89f5f5672ab",
    "evidence": [],
    "proposed_default": "defer"
  }
]
```

**What it means.** The loop saw the loss climb above the reference plus a
tolerance (an **alarm**). Because the alarm check has no passing certificate,
it does not retrain by itself. It opens a question for a person:
`l1_alarm_investigate`. If nobody answers in time, `proposed_default`
(`defer`) is applied and the outcome is recorded as `unverified`.
**Why it exists.** Some moves are not for software to take alone.
**What to do.** Look at the cause. Then reply, as a named person with your own
key (not the system's key).

### How to reply

A reply is a signed JSON file. There is no command to make it; create it in
Python:

```python
import json
from scmle.governance import deployment_ledger as dl

reply = dl.Attestation.create(
    "alices-own-secret",          # your key, never the system's
    escalation_id="esc-5",
    question_hash="d1044b61...",   # copy from `scmle loop pending`
    decision="approve",            # approve | override | reject | cannot_decide
    statement="I looked at the batch: the labels were mis-keyed, not the model.",
    evidence="ticket-1234",
    role="model owner",
    name_or_id="alice",
    date="2026-11-05",
    scope="stream",
    expires="2026-12-05",
)
json.dump(reply.to_dict(), open("reply.json", "w"), indent=2)
```

Put your key in a file and send the reply:

```bash
printf 'alices-own-secret' > alice.key
scmle loop reply deploy_c esc-5 --attestation reply.json --key-file alice.key
```

Seen:

```text
{"tier": "attested", "decision": "approve", "safe_default_applies": false}
```

The answer is recorded as an `attested` row. A second reply to the same
escalation is refused (Seen):

```text
config error: reply refused: 'esc-5' is not a pending escalation
```

Other reply refusals (From the code): `the reply answers a different
escalation`, `the reply is bound to a different question`, `the reply names a
different scope`, `the reply is signed with a key the system holds` (use your
own key), `no verification key for the reply's key id`, `the reply's signature
does not verify`, `the reply is expired or dated in the future`. Only
`approve` and `override` allow the action. `reject` and `cannot_decide` are
recorded and the safe default applies.

### Other loop events (From the code)

| Event | Meaning | What to do |
| --- | --- | --- |
| `insufficient_labels` status | fewer than `min_matured` outcomes arrived | keep feeding labels |
| `alarm_on_unverified_label_mechanism` halt | the loop cannot tell how labels were chosen (no logged propensity, no audit slice), so it will not retrain or promote | arrange labels that are complete, have logged propensities, or include a random audit slice |
| `selective_risk_above_target` halt (escalation `halt_selective_risk`) | among the cases the system answers itself, errors exceed the target | a person reviews; calibrate again (Python `Loop.calibrate`) |
| `ledger_verification_failed` / `state_does_not_match_ledger` halt | the files do not agree; possible edit | stop; run `scmle loop verify`; restore from a trusted copy |
| `coverage_below_nominal`, `out_of_support_rate_above_bound`, `two_consecutive_rollbacks`, `challenger_evidence_insufficient`, `feedback_loop_detected` | escalations with those names | investigate the named condition |
| any never-list action | the 11 actions that always need a person (moving the champion, reopening a sealed set, changing the charter, relaxing a threshold, ...) | answer with an attestation |

While halted, every case is escalated, and an `incident` row is opened.
`scmle loop verify deploy_c` checks the chain (Seen: `{"ok": true, "rows": 8,
"head": "1b740ab7..."}`). It cannot show that the newest rows were removed;
keep a copy of the head somewhere else.

The loop's escalation rule has not been compared with a human domain owner's
judgement, and two of its detection-delay gates missed their floor. See
[What SCMLE cannot do](WHAT_SCMLE_CANNOT_DO.md).

## Release gate and live checks

These run when you release to a cloud. They have been exercised on recorded
and mock data only, not on a real cloud account.

### `scmle deploy gate admit|certify manifest.json --key-file KEY` refuses (exit 2)

**What it means.** The evidence does not allow this release. All reasons are
printed together, and no record is written.
**What to do.** Fix each reason. From the code:

* `the model artifact is unsigned`, `the model has no approval`, `the model
  approval is not valid: ...`: sign the bundle; get an independent approval.
* `the image has no scan result`, `the scan result is for a different image`,
  `the image scan did not pass`: scan this exact image and fix the findings. An
  unreadable or empty scan counts as failed.
* `the source commit is not a full commit hash`: give the 40-character hash.
* `the approver is the person who requested the release`: someone else must
  approve.
* `production needs a staging deployment record` (and variants: the staging
  record's signature is not valid, is for another commit, model or image):
  deploy and certify staging first, for the same build.
* For `certify`: `live check <name> is missing` / `failed` / `was run for a
  different build`, `the platform's running state was not read`, `the platform
  runs <d1>, not <d2>`, `the platform does not report the service ready`.

If the manifest is incomplete you get (Seen):

```text
artifact error: manifest lacks environment, provider, source_commit, registry, model, version, container_digest, config_digest, requested_by, approver, approved_at
```

### `scmle deploy live` exits 1

One of four checks failed: health, schema, inference or security. The failure
strings (From the code) include `/readyz answered N and is not ready`, `the
answering model is not the released model`, `the service is not reached over
HTTPS`, `/predict with no token answered N` (it should refuse), `the refusal
leaks a traceback`, and `a read-only token reached /predict`. Fix the service,
not the test.

### `scmle deploy reconcile` exits 1

The registry, the ledger and the cloud disagree. The finding codes are
`no_registry_record`, `revision_mismatch`, `digest_mismatch`, `traffic_split`,
`state_mismatch`, `not_ready`, `alias_not_deployed`, `ledger_missing`,
`ledger_disagrees`. It repairs nothing; a person decides which side is right.

### `scmle deploy verify-record` exits 1

The signature of the deployment record is not valid, or the file cannot be
read. Treat the record as untrustworthy.
