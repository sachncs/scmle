"""The causal estimate on `C` as a registered claim of the ledger (F10).

This glue lives outside `scmle.data.identification` on purpose: that package
may not import the custody or confirmation APIs (an import test enforces it),
so the estimator code cannot reach `C` by itself.

The primary hypothesis is `theta = 0` for the charter's estimand, with the
strategy the selector chose from the charter and the declared DAG. It is
registered in the ledger by the hash of the charter's causal block, the
strategy, the DAG hash, the confirmation level and the exploration row count,
before the survivors are frozen. The runner receives the one frame the
confirmation releases and calls `run.identify` on it, so the estimate, the
falsification battery, the replication and the tier are computed from the same
opening that tests the bivariate family; nothing here opens `C`.

The five report files returned by `identify` are handed back to the ledger,
which writes their hashes into the confirmation row and so into the ledger
head. Subgroup hypotheses (charter format 9) are part of the registration:
the hash of the charter's causal block, which lists the declared levels, is
fixed before `C` opens. `identify` estimates the effect in each declared
subgroup on the released rows and adjusts the family by Holm's step-down at
the primary level; the result is reported in `identification_report.json`
and summarised in `causal_claims.json`, and it does not change the tier. The
replication over `strata` stays a falsifier that can only lower the tier.

Attestors (charter format 9) are named in the causal block, and the signed
attestations are read from the file it names before `C` opens. An attestation
counts only if its attestor is listed for that assumption, its scope is this
charter's hash, it is valid today and its signature verifies under the
signing key (`scmle.data.identification.attest`); what was accepted and why
the rest was not goes into `causal_claims.json`. A charter that names no
attestors still reports `unverified` ("no named attestor"), and so does one
whose attestations do not verify; the estimate and its interval are reported
either way.
"""

from collections.abc import Mapping
import datetime
import json
from pathlib import Path
from typing import Any

import pandas as pd

from scmle.contracts import errors
from scmle.data.admissibility import dag as dag_module
from scmle.data.charter import causal
from scmle.data.identification import attest
from scmle.data.identification import run
from scmle.data.identification import select
from scmle.data.ledger import budget as budget_module
from scmle.data.ledger import confirm
from scmle.data.ledger import ledger as ledger_module

KIND = "causal_primary"
CLAIM_ID = "causal_primary_theta_zero"
CLAIM_FILE = "causal_claims.json"
DOCUMENTS = run.DOCUMENTS


def load_dag(path: str | None) -> dag_module.Dag | None:
    """The DAG hypothesis a causal block declares, verified by its hash.

    Raises:
        ConfigError: If the file cannot be read or is not a stored DAG.
    """
    if not path:
        return None
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        return dag_module.Dag.from_artifact(document)
    except (OSError, ValueError, TypeError) as error:
        raise errors.ConfigError(
            f"cannot load the declared DAG {path}: {error}"
        ) from error


def registration(
    spec: causal.CausalSpec,
    plan: select.Plan,
    dag: dag_module.Dag | None,
    error_budget: budget_module.ErrorBudget,
    exploration_rows: int,
) -> confirm.RegisteredClaim:
    """The claim as it is fixed before `C` is opened."""
    body = {
        "hypothesis": f"{spec.estimand} of {spec.treatment} on "
        f"{spec.outcome} = 0",
        "causal": spec.to_dict(),
        "strategy": plan.primary,
        "dag": None if dag is None else dag.digest(),
        "alpha_confirm": str(error_budget.confirmation.alpha_confirm),
        "exploration_rows": exploration_rows,
    }
    return confirm.RegisteredClaim(
        CLAIM_ID,
        KIND,
        f"{spec.estimand} of {spec.treatment} on {spec.outcome} differs from "
        f"zero, by {plan.primary}, under the attested assumptions",
        ledger_module.digest(body),
    )


def jsonable(document: Mapping[str, Any]) -> dict[str, Any]:
    """A JSON round trip, so the hashed document is the written one."""
    return dict(json.loads(json.dumps(document, default=str)))


def verified_attestors(
    spec: causal.CausalSpec,
    plan: select.Plan,
    charter_hash: str,
    signing_key: str | bytes | None,
) -> attest.Applied:
    """Which attestations count for the assumptions the chosen strategy needs.

    Raises:
        ConfigError: If the attestation file cannot be read.
    """
    if plan.primary is None or spec.attestations is None:
        raise errors.ConfigError("the charter declares no attestations file")
    try:
        given = attest.load(spec.attestations)
    except ValueError as error:
        raise errors.ConfigError(str(error)) from error
    return attest.apply(
        spec.attestors,
        given,
        run.needed_assumptions(plan.primary, spec),
        charter_hash,
        datetime.date.today().isoformat(),
        signing_key,
    )


def build_runner(
    spec: causal.CausalSpec,
    x: pd.DataFrame,
    target_name: str,
    error_budget: budget_module.ErrorBudget,
    *,
    dag: dag_module.Dag | None = None,
    attestors: Mapping[str, str] | None = None,
    seed: int = 0,
    dag_report: Mapping[str, Any] | None = None,
    charter_hash: str | None = None,
    signing_key: str | bytes | None = None,
) -> confirm.ClaimRunner | None:
    """The registered primary claim, or None when no strategy identifies it.

    Args:
        spec: The charter's causal block.
        x: Exploration rows with the outcome column (design, nuisances).
        target_name: The outcome column's name in the released frame.
        error_budget: The charter's error budget.
        dag: The declared DAG, if any.
        attestors: Assumption to named attestor.
        seed: Learner seed.
        dag_report: What the admissibility stage found about the DAG on `X`,
            copied into the identification report.
        charter_hash: The charter's hash, the scope attestations must carry;
            needed when the block names attestors.
        signing_key: Key the attestations are verified with; defaults to the
            configured `SCMLE_SIGNING_KEY`.

    Returns:
        None for a charter nothing identifies: no estimate can exist, `C` is
        not opened for it, and the stage writes the refusal from `X`.
    """
    plan = select.select(spec, dag)
    if plan.primary is None:
        return None
    claim = registration(spec, plan, dag, error_budget, len(x))
    attestation_report: Mapping[str, Any] | None = None
    if spec.attestors and attestors is None:
        if charter_hash is None:
            raise errors.ConfigError("attestors need the charter hash")
        applied = verified_attestors(spec, plan, charter_hash, signing_key)
        attestors = applied.accepted
        attestation_report = applied.to_dict()

    def evaluate(features: pd.DataFrame, target: pd.Series) -> confirm.ClaimRun:
        released = features.copy()
        released[target_name] = target
        outcome = run.identify(
            spec,
            x,
            released,
            error_budget,
            attestors=attestors,
            dag=dag,
            seed=seed,
            dag_checks=dag_report,
            attestation_report=attestation_report,
        )
        documents = {k: jsonable(v) for k, v in outcome.documents.items()}
        return confirm.ClaimRun(
            {CLAIM_ID: str(documents[CLAIM_FILE]["status"])}, documents
        )

    return confirm.ClaimRunner((claim,), evaluate)
