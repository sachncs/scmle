# Decisions to make before you run anything

A model is the answer to a question you chose. If you choose badly, the tool
will still give you a number, and the number will look fine. This page lists
the choices only a person can make, how to make each one, and what happens if
you get it wrong.

Fill in the sheet at the end before you press run. Ten minutes here can save a
week.

New words are explained in the [glossary](GLOSSARY.md).

## 1. What exactly am I predicting? (the target)

**The question.** Which single column holds the answer, and does it really
mean what I think it means?

**How to choose.**

* Say it as a sentence: "Predict whether _this_ happens to _this kind of
  thing_ by _this moment_."
* Look at how the column is recorded. Who writes it, and when? A column called
  `churned` that is set when the account is closed is different from "will
  churn in the next 30 days".
* If there are many possible answers, check the task SCMLE picks in
  `scmle plan` (`binary_classification`, `multiclass`, `regression` and so on).
  If it picks wrong, say so with `--task`.

**If you choose wrong.** You get a precise answer to the wrong question. A
model that predicts "customer has closed their account" is useless for finding
customers about to leave. SCMLE cannot see this; only you can.

**Command.** `--target survived`.

## 2. How will I judge a prediction? (the metric)

**The question.** Which number tells me the model is good enough for my use?

**How to choose.**

| Your situation | A good first metric | Why |
| --- | --- | --- |
| Two answers, balanced, you rank cases | `roc_auc` | Ignores the cut-off; good for ranking |
| Two answers, the "yes" is rare (under about 10 percent) | `pr_auc` (PR AUC) | ROC AUC looks fine even when you miss most rare cases |
| You act on the probability itself | `log_loss` | Punishes confident mistakes and bad probabilities |
| A number, big mistakes hurt more | `rmse` | Squares the errors |
| A number, all mistakes count the same | `mae` | The typical miss |
| Many categories | `f1_macro` or `log_loss` | Counts every category equally |

SCMLE lists the others it knows in `scmle.evaluation.metrics.CATALOG`. You can
also add your own with `scmle.register_metric`, for example the money lost.

**If you choose wrong.** The search picks the model that is best at the wrong
thing. Accuracy on a rare event picks a model that always says "no". The
tool's pick is `roc_auc` for two answers by default; that is only a default.

**Command.** `--metric pr_auc`.

## 3. Which columns may the model use? (what you may not use)

**The question.** At the moment I will use the model, which of these facts do I
really have?

**How to choose.** Go down the column list and for each ask: "When is this
value filled in?" If it can be filled in only after the outcome, or because of
the outcome, remove it.

Typical traps:

* A column recorded after the event (a cancellation date, a lifeboat number, a
  payment-chased flag).
* A column that is a copy or near-copy of the target in disguise (a status
  code that contains the answer).
* An identifier (row number, customer id). It lets the model memorise rows.
  Mark it with `--id-columns`.
* A column you are not allowed to use for legal or ethical reasons. Remove it,
  and also name it in `--sensitive-columns` so SCMLE reports how the model
  treats each group.

Tools:

* `--unavailable-at-prediction boat body` says "known only after the outcome".
  This is the right flag for leaks.
* `--exclude colA colB` drops columns for any other reason.
* `--leakage-policy block` turns SCMLE's suspicions into a stop. Use it.

**If you choose wrong.** A leak makes the model look brilliant. On the Titanic
table, the score was 0.988 with `boat` and `body`, and 0.885 without. A model
shipped with a leak fails the first day it is used, because the leaked column
is empty or different in real use. See
[Reading results](READING_RESULTS.md).

## 4. Are my rows independent? (grouping)

**The question.** Do several rows belong to the same person, shop, machine or
household?

**How to choose.** If yes, name the identifier column as the group. SCMLE then
keeps every row of a group on the same side of every split.

**If you choose wrong.** If one customer has rows in both the practice part and
the exam part, the model has seen that customer's habits before the exam. The
exam score is too high, and the model will do worse on new customers than the
report says.

**Command.** `--group-column customer_id`. If the same entity appears many
times and you do not group, SCMLE may print a recommendation about duplicates
(see the last line in Step 7 of [Start here](START_HERE.md)).

## 5. Does time matter? (time order)

**The question.** Will I use this model to predict the future from the past?

**How to choose.** If your rows have dates and you will always predict later
rows, then yes. Name the date column. SCMLE trains on the earlier rows and
tests on the later ones. If the order of rows is meaningless, no.

* `--time-column event_time` trains on the past and tests on the future.
* `--validation temporal` or `rolling` for stricter schemes; `--embargo-rows N`
  leaves a gap between training and test so that neighbouring rows do not leak
  into each other.
* `--feature-delay COL=ROWS` for a column that is only known some rows after
  the fact.

**If you choose wrong.** Randomly shuffled rows let the model learn from the
future. The exam looks easy, and then the real future is harder. For
time-series data this is the most common reason for a model that works in the
lab and not in use.

## 6. What do the two kinds of mistakes cost? (the cost of errors)

**The question.** If I say "yes" and I am wrong, what does that cost? If I say
"no" and I am wrong, what does that cost?

**How to choose.** Put a number on each, in any unit you like, as long as they
are in the same unit. A missed fraud may cost 8 times a false alarm. A missed
disease may cost 50 times an unneeded test. If you truly cannot tell, say so,
and see "If you do not know the costs" below.

Write it down once:

```json
{"decision": {"threshold_objective": "cost", "false_positive_cost": 1, "false_negative_cost": 8}}
```

Save that as `cost.json` and pass it with `--config cost.json`. SCMLE then picks
the probability cut-off that minimises the total cost on practice predictions,
instead of 0.5.

**If you choose wrong.** The default cut of 0.5 treats both mistakes as equal.
If a miss is 8 times worse, 0.5 is too timid. If you invent costs, the cut will
look scientific and be arbitrary. Report the costs next to the cut.

**If you do not know the costs.** Write that down as an open question. The
careful (charter) mode treats costs that no person supplied as `unverified`,
and lists them as questions that could change the verdict. That is better than
a made-up number.

## 7. How sure do I need to be? (error budget and the exam)

**The question.** How many mistakes in my *findings* (not predictions) can I
tolerate, and how many rows can I keep unseen?

**How to choose.** For a first model, the defaults are fine: SCMLE keeps about
a fifth of the rows for the exam (`--test-fraction`, default 0.20; our
Titanic run kept 216 of 1309). With a charter, you
choose the confirmation fraction (default 0.25), the effect size worth finding
(default 0.2) and the error allowance (`alpha_confirm` of 1/20). Run
`scmle charter check charter.json --rows N` to see whether the number of rows
you have can support your promises.

**If you choose wrong.** With too few rows the exam score has a wide band and
can be luck. A charter that promises more than your rows support is
"downgraded": the run still finishes but its scope becomes
`exploratory_unconfirmed`, and no claim is called confirmed. Example, with the
default effect 0.2 and the 1093 Titanic rows left after the exam rows:
`feasibility: downgrade (273 rows < 333 needed for power 0.8 at effect 0.2; detectable effect is 0.221)`.

## 8. How much time may the search use?

**The question.** How long may the computer search?

**How to choose.** Start with `--time-budget 120` to see the whole pipeline
work. For a result you will rely on, give it 10 minutes or more. Set
`--workers` to the number of spare CPU cores. A longer budget lets SCMLE tune
more hyperparameters (the settings of each model).

**If you choose wrong.** A budget under 300 seconds searches only the cheap
models, and the report may skip parts. In our run, a 60-second budget left the
feature-importance table empty. A tight budget also makes the result depend on
how fast your computer is.

## 9. Who is allowed to approve it? (people and keys)

**The question.** Who will review the model before it goes live, and where is
the secret signing key kept?

**How to choose.** The person who approves a promotion must not be the one who
built the model. Keep the signing key in a secret manager, not in a file in
the repository. Use `SCMLE_SIGNING_KEY_FILE` to point at a mounted file.

**If you choose wrong.** If you sign your own approvals with a key you leave in
the code, the signature proves nothing. SCMLE checks the mechanism, not the
honesty of the people.

## 10. What will make me stop?

Write this before you see results, so that you cannot talk yourself out of it
later.

* The lowest exam score I would accept (a number, not "good").
* The largest gap between practice and exam score I would accept.
* The drift or performance drop at which I roll back.
* The names of the people to call.

SCMLE can hold your thresholds as success criteria
(`ProblemSpec.success_criteria`). If you declare none, the release evaluation
says `policy_not_declared` and blocks: the tool will not invent thresholds from
the metrics it just saw.

## The decision sheet

Copy this, fill it in, keep it next to the run.

```text
Question in one sentence:
Target column:
What each row is:
Metric and why:
Columns I may NOT use, and why:
Identifier columns:
Group column (rows that belong together):
Date column (and do I predict the future?):
Cost of a false alarm:        Cost of a miss:
Costs came from (a person's name, or "unknown"):
Lowest exam score I accept:
Largest practice-exam gap I accept:
Time budget:
Reviewer (not me):
Where the signing key lives:
```
