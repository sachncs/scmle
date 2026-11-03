# F1: Charter and decision map

Canon: [`../../plan/00-canon.md`](../../plan/00-canon.md). Phase 2.

## Purpose

Fix, before any data is explored, what is being asked, of whom, for which
decision, and how much error the answer may carry. The charter is the one
document every later feature reads; nothing downstream may widen it.

## Decision it serves

Whether a run is **confirmatory** (claims can reach tier `confirmed`) or
**exploratory, unconfirmed** (n is too small to pay for confirmation), and
which action each gating check triggers. No check may gate a run unless the
decision map binds it to an action (P1).

## Behaviour

Today the nearest object is free text: `ProblemSpec.decision`,
`.population`, `.success_criteria` (`scmle/contracts/tasks.py:534-547`). The
charter replaces it with structured, validated fields (proposed).

```python
# proposed; mirrors the FrozenModel pattern (tasks.py:122-125)
class Estimand(FrozenModel):
    kind: Literal["predictive", "causal"]
    population: str  # who/what the claim is about
    target: str  # variable, with its measurement window
    prediction_moment: str  # when a prediction is issued (as-of rule)
    intercurrent_events: dict[str, str]  # event -> strategy (censor,
    # exclude, treat-as-outcome, ...)
    summary_measure: str  # e.g. "AUC", "mean difference"


class AnalysisCharter(FrozenModel):
    question: str
    estimand: Estimand
    unit: str  # unit of analysis, e.g. "customer-month"
    deployment_population: str  # reference sample URI or stated marginals
    decision_map: tuple[DecisionRow, ...]
    sets: SetSizes  # fractions for X / C / F (see F2)
    min_effect: float  # smallest standardised effect worth acting on
    error_budget: ErrorBudget  # alpha_confirm, W0, K_max, B, power
    multiverse: tuple[Choice, ...]  # declared before exploration (F6)
```

CLI (proposed): `scmle charter init | validate | sign | show`. Artifacts:
`analysis_charter.json`, `error_budget.json`, `feasibility.json`. The charter
hash is stamped on every ledger entry, certificate and claim.

`DecisionRow = (check_id, trigger, action, certificate_id)`. Actions are a
closed set: `drop_column`, `quarantine_column`, `reweight`, `switch_split`,
`block_run`, `downgrade_tier`. `trigger` is a test outcome at the budgeted
level, not a raw threshold.

## Method

**Closed-form power at a Bonferroni level, as an upper bound for BH.** With
`K_max` the most claims the run may carry into confirmation and effect
`delta = min_effect`, the confirmation set needs

- standardised mean effect: `n_C >= ((z_{1-a/(2K)} + z_{1-b}) / delta)^2`,
- correlation `r`: `n_C >= ((z_{1-a/(2K)} + z_{1-b}) / atanh(r))^2 + 3`,

with `a = alpha_confirm`, `1 - b` = target power (default 0.80). The inverse
gives the minimum detectable effect `MDE(n_C)`. For metric claims on
classification the binding quantity is the minority count: the Wilson
half-width at `n_minority` must not exceed the width the decision needs.

Why this and not simulation: the formulas are auditable by hand and have
published reference implementations. Bonferroni at `K_max` is conservative
for BH, so a pass here is a pass under the real procedure. `K_max` is the
bridge to F3: the ledger may forward at most `K_max` survivors, so the power
promised here is the power delivered there.

Cut: per-hypothesis simulation power, adaptive sample-size re-estimation,
Bayesian assurance. Each adds machinery without changing the gate.

## Guarantees and assumptions

- If `n_C >= n_required` and the test statistic is asymptotically normal, a
  true effect of size at least `min_effect` is confirmed with probability at
  least `1 - b` (assumes units are independent or blocks exchangeable, per F2).
- The charter fixes `alpha_confirm`, `W0`, `K_max`, `B` before any exploration
  reads data. An amendment after the first ledger entry is allowed only if it
  tightens the budget (smaller alpha or `K_max`); otherwise the run blocks.
- The charter hash (SHA-256 of canonical JSON) is always recorded. An HMAC
  signature (below) proves possession of the key, not the signer's identity.

## Failure modes and degradation

| Condition | Outcome |
|---|---|
| `n_C < n_required` | Scope becomes `exploratory_unconfirmed`; claims cap at tier `unverified`; the achievable MDE is reported. Run continues. |
| `min_effect` absent | Derived from `success_criteria` if present, else a flagged default; claims cap at `attested` at best. |
| Estimand attribute missing | `validate` fails; the run does not start. |
| No deployment sample | Representativeness is reported "unverifiable" (F6), not passed. |
| Gating check with no decision-map row or certificate | Registry lint fails the build. |
| Unsigned charter | Allowed; hash-only, noted in the report header. |

## Repo integration

- Verified: `ProblemSpec` and `ProjectSpec` (`tasks.py:534-570`); `leakage_policy:
  "warn" | "block"` is the current coarse gate (`tasks.py:570`, enforced at
  `scmle/core/stages.py:498`) and is superseded by the decision map (proposed).
- Verified: canonical JSON and HMAC-SHA256 helpers
  (`scmle/contracts/signing.py:26,36`), key from `resolve_key`
  (`signing.py:14`).
- Verified: the split happens in `make_split` (`stages.py:608`); the charter
  must exist before it. Seed and set sizes today live in `ValidationSpec`
  (`tasks.py:317-352`); the charter owns them going forward (proposed).
- Proposed location: `scmle/data/charter/`.

## Acceptance criteria

- Property test: for random `(K, delta, alpha, power)`, `MDE(n_required(...))`
  equals `delta` to 1e-9, and simulated power at `n_required` is >= target
  minus a binomial tolerance band.
- `validate` rejects a charter missing any of: population, target, prediction
  moment, intercurrent-event strategy, summary measure.
- Registry lint: every check with `gate=True` has one decision row and one
  certificate id; every decision row names a check that exists.
- Charter hash is identical across runs and platforms (canonical JSON), and a
  one-byte edit changes it and invalidates the signature.
- With `n_C` below requirement, the run completes and every claim has tier
  `unverified` and scope `exploratory_unconfirmed`.

## External referent

Power values are cross-checked against R `pwr` (`pwr.r.test`, `pwr.t.test`)
and G*Power on a fixed table of inputs, committed as fixtures. The required
estimand attributes are those of ICH E9(R1), not author invention. Charters
are reviewed for answerability by a domain owner (recorded as an attestation).
