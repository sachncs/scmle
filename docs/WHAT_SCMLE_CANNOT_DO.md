# What SCMLE cannot do

SCMLE is careful, and it is honest about its limits. This page lists them in
plain words, so that you do not trust it for something it does not promise.
Every limit here is already stated somewhere in the repository; the sources are
linked.

## It does not replace your judgement about the problem

SCMLE cannot know:

* Whether you are predicting the right thing.
* Whether a column will exist when you use the model. It can *suspect* a leak
  from the numbers, never *know*. `unverified` means exactly that: nobody
  checked.
* Whether the costs of your mistakes are what you wrote down.
* Whether it is acceptable, fair or legal to use a model for this purpose.
* Whether the world will stay like your training data.

The tool's job is to make your choices explicit and to stop you when the data
disagrees with them. The choices are yours. See [Decisions](DECISIONS.md).

[`product/README.md`](../product/README.md) says it directly: it is not a
replacement for a data scientist today. Its measured evidence is its own
benchmarks, which are not a head-to-head with people.

## Statistics: what is guaranteed and what is not

* **A held-out score is an estimate with a band, not a promise.** With 216
  exam rows, our ROC AUC of 0.885 had a 95 percent band of 0.841 to 0.927. The
  band assumes new data looks like the data you tested on.
* **Choosing the best of many models is slightly optimistic.** The more models
  tried, the more the winner's practice score flatters it. The exam score is
  the one to trust ([`README.md`](../README.md) says the same).
* **Duplicates can inflate scores.** SCMLE warns and keeps copies together in a
  charter run, but it does not remove them for you.
* **A leakage flag is a suspicion.** A real strong predictor looks the same to
  the screen. Only a person who knows when the column is filled in can settle
  it.
* **No leak is proven absent.** A leakage screen that finds nothing means
  nothing was flagged, not that none exists.
* **Calibrated uncertainty (conformal intervals and sets) holds if new data
  resembles the calibration data.** It is not a guarantee against a changed
  world.

## Claims: confirmed, attested, unverified, and what each is worth

* **`confirmed` is narrow.** It says a finding passed one test on sealed rows
  with a stated error allowance for the whole list. It says nothing about
  causes, about other data, or about whether the model is good.
* **The guarantee depends on the charter being honest.** A charter written
  after looking at the data defeats its own purpose. The signature proves
  possession of the key, not the identity of the person
  ([`CLI.md`](CLI.md)).
* **`attested` is as good as the person.** SCMLE records who vouched. It does
  not check they were right.
* **`unverified` is the default for anything a person has not supplied.** This
  includes every cost, every "when is this column known", and every field a
  language model drafted.
* **Causal claims need a named attestor for every assumption.** Without one the
  tier is `unverified`. A held-out set alone never confirms a causal effect,
  and `confirmed` does not say the assumptions are true. If no identification
  strategy fits your charter, the claim is `not_identified` and SCMLE refuses
  to produce an estimate.
* **Custody is tamper-evident, not adversary-proof.** The sealed rows sit in a
  separate helper process, so ordinary code cannot read them. But a person with
  the same account on the same machine could read the original file, and the
  DataFrame you passed in still holds every row. For a hostile setting you need
  separation: another operating-system user, a service, or write-once storage
  ([`SECURITY.md`](SECURITY.md)).
* **Newest ledger rows can be cut off unnoticed.** The hash chain shows edits,
  deletions in the middle and reordering, not the removal of the last rows.
  Keep a copy of the head ([`ARTIFACTS.md`](ARTIFACTS.md)).
* **`scmle claims verify` proves the files agree with each other.** It does
  not recompute the statistics, and a party who can rewrite every file
  consistently can pass it.

## Certified versus descriptive

A check may *stop* a run or a release only if its error rate has been measured
against planted problems and is on file as a certificate. Everything else is
**descriptive**: shown, never decisive. In practice:

* Some outputs say `descriptive` or `unverifiable` (for example the shift
  report with no deployment sample). They inform; they do not certify.
* Stability results are warnings unless a passing certificate for your
  thresholds exists. Without a charter that asks for it, nothing is withdrawn.
* The certificate files live in `benchmarks/qualification/` in a source
  checkout. A package installed from pip does not carry them, so detectors there
  are descriptive until you have the files
  ([`ARTIFACTS.md`](ARTIFACTS.md)).

## The closed loop and escalation

* The loop's detection-delay gates (H2 and H10) **missed their floor** in
  measurement ([`ARTIFACTS.md`](ARTIFACTS.md)).
* Its rule for when to ask a person has **not been compared with a domain
  owner's own audit**, and has not been replayed on recorded incident streams.
  Do not treat "the loop did not escalate" as "a human would not have".
* A feedback loop (where the model's own decisions shape the labels you see)
  cannot be detected without a randomly chosen slice of cases that bypass the
  model.
* Retraining, challengers and calibration are Python calls, not commands. The
  command line cannot, for example, create a signed reply or release a halt.

## Serving and security

* The built-in HTTP server is a **minimal reference server**: no encryption, no
  user accounts of its own. It refuses to listen beyond your machine unless
  token checking is configured, and you must terminate TLS elsewhere
  ([`SERVICE_SLO.md`](SERVICE_SLO.md)).
* Signatures are shared-secret (HMAC). They prove "someone with this key made
  this and it has not changed". They are **not** public-key signatures, a
  transparency log, or proof of who built a model or that the build machine
  was honest ([`ARTIFACTS.md`](ARTIFACTS.md)).
* Model bundles are pickle files. Opening one runs code. That is why signing
  is mandatory by default.
* A memory limit is checked every tenth of a second, so a very fast spike can
  overshoot once before the trial is killed ([`README.md`](../README.md)).
* Latency and availability objectives are **targets measured on a developer
  machine**, not promises under load ([`SERVICE_SLO.md`](SERVICE_SLO.md)).

## The cloud has not been proven

* AWS and Google Cloud release, deployment and recovery have **never been run
  against a real account**. The workflows, Terraform and the release gate are
  tested against recorded answers, scripted platforms and mocks.
  The answers were shaped from the platforms' documentation, not captured
  from them ([`CLOUD_RELEASE.md`](CLOUD_RELEASE.md)).
* The local demo ran on a developer machine, but **Docker was not run**: no
  image was built and no container started
  ([`LOCAL.md`](LOCAL.md)). The container steps in
  [Start here](START_HERE.md) (`build.sh`) are untested end to end.
* Staged traffic (canary) is library code that is **not wired into the
  production workflow** yet ([`CLOUD_RELEASE.md`](CLOUD_RELEASE.md)).
* No production-qualified claim on AWS or GCP may be made until the evidence
  for both providers exists and passes the project's own check
  ([`RELEASING.md`](RELEASING.md)). A passing local run is not a step towards
  that sentence.
* Cost, multi-region availability and latency equivalence between providers are
  not claimed.

## Scope of what it models

* Tabular data is the home ground. [`API_STABILITY.md`](API_STABILITY.md) says
  which parts of the interface are stable, and the package is in beta: the
  interface may change between `0.x` releases.
* TabPFN needs its own licence and token. PyTorch and Keras families need an
  extra install. No language-model client ships; recorded drafts can be
  replayed, and a replayed draft with no calibrated threshold is refused.
* Series features use the recent rows stored with the model. A record that
  predates an entity's stored history gets missing values.
* Windows: peak-memory reporting counts only the main process, and the
  free-threaded build is not tested there.

## Things a first-time user often assumes (and should not)

| You may think | The truth |
| --- | --- |
| A high score means a good model | It may mean a leak. Check the top features |
| `confirmed` means it works | It means one narrow test passed on sealed rows |
| `unverified` means broken | It means nobody checked a default |
| A green drift report means safe | It means no drift was detected, not that none exists |
| The cloud steps are tested | They are tested on mocks only |
| SCMLE chose the right metric | It chose a default. Check it |
| Signing means trusted | Signing proves the key, not the person |
| The tool will tell me my target is wrong | It cannot see what your target means |
