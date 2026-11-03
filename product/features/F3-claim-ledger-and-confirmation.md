# F3: Claim ledger and confirmation

Vocabulary and principles: [`../../plan/00-canon.md`](../../plan/00-canon.md).
Built in [Phase 3](../../plan/phase-3-ledger-and-confirmation.md); needs F2.

## Purpose

Make every comparison the system runs during exploration a logged
hypothesis, spend a fixed alpha-wealth on them, and turn the survivors into
claims by testing each once on the untouched confirmation set `C`.

## Decision it serves

Which exploration findings become `confirmed` claims that may shape the
modelling specification, and which are reported as `unconfirmed` or
`fragile`. A finding that is not confirmed never changes feature admission
or preprocessing.

## Behaviour

Proposed interface:

```python
ledger = Ledger.open(run_dir, charter)  # append-only, hash-chained
h = ledger.register(  # BEFORE the data is touched
    name="income~churn",
    family="bivariate",
    statistic="spearman",
    null="no monotone association",
    direction="two-sided",
    set="X",
)
result = ledger.test(h, frame_x)  # returns p-value / e-value
ledger.decision(h)  # reject / retain, wealth left
claims = confirm(ledger, custody, charter)  # unlock C once
```

Artifacts: `hypothesis_ledger.jsonl` (one line per hypothesis: registered
spec, hash of the code path, seed, p- or e-value, alpha spent, wealth
before/after, decision, previous-line hash), `negative_control.json`,
`confirmed_claims.json` (each with tier, test, p-value, adjusted decision),
and `error_budget.json` (from F1).

Rules: registration precedes evaluation; the ledger is append-only and
hash-chained so deletion or reordering is detectable; a hypothesis cannot be
evaluated on `C` unless it was rejected on `X` and registered as a survivor
before `C` was unlocked.

## Method (one rigorous method)

Two stages, kept separate on purpose (canon P2):

1. **Exploration triage: online FDR on `X`.** Default rule: **e-LOND**
   (Xu and Ramdas, online multiple testing with e-values). Reason: it is
   proposed to control FDR under arbitrary dependence between the e-values,
   which is the realistic case for many tests on one dataset. Permutation
   p-values are converted with a calibrator (`e = kappa * p^(kappa-1)`),
   which costs power; that cost is accepted for the dependence guarantee.
   Options, selectable in the charter: **LORD++** (Ramdas et al. 2017) and
   **SAFFRON** (Ramdas et al. 2018, adaptive). Assumptions as I recall them:
   LOND/LORD++/SAFFRON guarantee FDR control for independent (some LOND
   variants: PRDS) p-values with each level `alpha_t` a function only of past
   decisions; e-LOND needs valid e-values (`E[e] <= 1` under the null,
   conditionally on the past) and no independence. **These conditions are
   from memory and must be verified against the papers, and against the R
   `onlineFDR` package, at the start of Phase 3; the rule's docstring cites
   the exact theorem used.**
2. **Confirmation on `C`: BH on the survivors, once.** Survivors `S`
   (selected using `X` only) are tested on `C` with Benjamini-Hochberg at
   `alpha_confirm`, or Benjamini-Yekutieli if dependence is declared
   arbitrary, or Holm if the charter asks for FWER. Because `S` is a function
   of `X` and `C` is independent of `X`, the usual FDR theorem applies
   conditionally on `S`.

Cut: no combination of exploration and confirmation p-values, no
"learning" of alpha across stages (the stages do not add), no per-test
adaptive stopping.

## Negative controls (gate evidence)

- **Permuted-target control.** Re-run the whole exploration phase `B >= 20`
  times with the target permuted **within groups** (grouped data) or
  **within contiguous blocks** (temporal data), never globally, so
  dependence structure survives. Record the number of discoveries per
  permutation. Gate: observed discoveries on real data must exceed the 95th
  percentile of the permuted counts; otherwise the run reports "findings do
  not clearly exceed the false-discovery floor" and no claim is promoted.
- **Resampling control for feature-only findings.** Findings that do not use
  the target (drift, shape, pairwise feature structure) are not covered by
  target permutation. Control them by re-running on bootstrap or
  block-bootstrap resamples with the finding's structure broken (column-wise
  independent shuffles for association findings; time-shuffled blocks for
  drift). Same 95th-percentile gate.

## Unlock-once confirmation and failure policy

`C` is opened once through F2's custody API with the hash of `S`. The open
event is logged with the charter hash. If the confirmation run fails for a
pipeline reason (exception, not a statistical null), the policy fixed in the
charter applies: either (a) **declare unconfirmed**: all claims keep tier
`unverified` and the report says so, or (b) **reusable holdout**:
Thresholdout (Dwork et al. 2015) with a recorded noise scale and query
budget, after which `C` is spent. Default: (a). A statistical failure to
reject is a result, not a failure.

## Guarantees and assumptions

- `FDR(confirmed) <= alpha_confirm`, conditional on `S`, if: `C` rows are
  independent of `X` rows (grouped split by entity; embargo for temporal
  data, from F2); `C` is read once; each test on `C` is valid under its null
  (permutation or exact test; F7 checks size); the test family is exactly
  `S`, fixed before unlocking.
- Exploration FDR (e-LOND) bounds the false share of candidates, not of
  claims. It is a budget, not the guarantee.
- The negative control estimates the false-discovery floor; it is empirical
  (B permutations), so its 95th percentile has Monte Carlo error that is
  reported.

## Failure modes and degradation

| Situation | Behaviour |
|---|---|
| `|C|` too small for the minimum detectable effect (F1 feasibility) | Charter downgrades scope; all findings `unconfirmed`. |
| Hypotheses registered after seeing results | Detected by ledger order and hash chain; run marked invalid for claims. |
| Wealth exhausted | Remaining hypotheses are descriptive only; logged as not tested. |
| Dependence between tests unknown | Default to e-LOND / BY, accept lower power. |
| `C` accessed twice | Custody refuses; the first open stays in the log. |

## Repo integration

- Extends the existing evidence of structured findings: `CurationFinding`
  (`scmle/contracts/data.py:85`) carries `evidence`, `confidence` and
  `blocking`; ledger entries are proposed to be referenced from its
  `evidence` rather than replacing it. `CurationReport.release_ready`
  (`scmle/contracts/data.py:797`) is the existing gate the confirmation
  outcome feeds.
- The one existing multiplicity correction is Bonferroni inside
  `missingness.assess`: `ALPHA = 0.01` (`scmle/data/profiling/missingness.py:21`),
  `threshold = ALPHA / max(len(candidates), 1)` (`:104`). It becomes a ledger
  client and loses its private alpha.
- Sealed access pattern to mirror: `SealedTestSet` (`scmle/governance/vault.py:56`)
  and `seal` (`scmle/core/stages.py:551`, called at `:631`).
- Proposed new package: `scmle/data/ledger/`.

## Acceptance criteria (machine-checkable)

1. Under a global null (simulated, 10^4 runs), the e-LOND ledger's empirical
   FDR is <= `W0` with a one-sided binomial tolerance; same on correlated
   features (dependence test).
2. The confirmation stage's empirical FDR on semi-synthetic data (real `X`,
   planted effects) is <= `alpha_confirm` with tolerance; power is reported.
3. Ledger tamper test: deleting, reordering or editing any line fails
   verification.
4. A hypothesis evaluated before registration is rejected in code.
5. Permuted-target control on grouped and on temporal fixtures: permuted
   discoveries obey the nominal rate; a global permutation on the same data
   is shown to be anti-conservative (documents why within-block is required).
6. A second open of `C` raises and is logged.
7. Differential test: LORD++/SAFFRON levels equal R `onlineFDR` output on
   fixed p-value sequences to 1e-9.

## External referent

R `onlineFDR` (reference levels for LORD++/SAFFRON/LOND); the published
theorems for e-LOND; the semi-synthetic simulation corpus in F7; the
independent replication run by an outside party in Phase 6.
