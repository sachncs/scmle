"""Typed run result and the `evaluation.json` record's versions."""

from collections.abc import Mapping
import dataclasses
import json
import pathlib
from typing import Any

import numpy
import pandas

import scmle.artifacts.bundle
from scmle.contracts import experiments
from scmle.contracts import records
import scmle.contracts.data
from scmle.evaluation import fit_health
from scmle.evaluation import parameter_estimation

LEADERBOARD_DETAILS = (
    "folds",
    "access",
    "evaluation",
    "runtime",
    "training",
)
HIGHER = "higher"
LOWER = "lower"
GROUPS = records.nest(
    {},
    groups={
        "selected": {
            "selected_model": "model",
            "selected_trial_id": "trial",
            "selected_params": "params",
            "selected_strategy": "strategy",
            "selected_cv_score": "score",
            "selected_cv_std": "deviation",
        },
        "rows": {"n_dev": "development", "n_test": "test"},
        "digests": {"split_fingerprint": "split"},
        "release": {
            "release_attestation": "attestation",
            "release_policy": "policy",
            "release_evaluation": "evaluation",
            "release_evidence": "evidence",
        },
    },
)
"""Gathers the version-1 keys that share a prefix into objects (rule 2)."""


def orient(layout: records.Layout) -> records.Layout:
    """Gathers the grouped keys and turns the metric direction into a word.

    Args:
        layout: An evaluation record after the version-2 renames.

    Returns:
        The record with its `GROUPS` objects, and a boolean `direction` given
        as `higher` or `lower`.
    """
    layout = GROUPS.upgrade(layout)
    if isinstance(layout.get("direction"), bool):
        layout["direction"] = HIGHER if layout["direction"] else LOWER
    return layout


def disorient(layout: records.Layout) -> records.Layout:
    """Undoes `orient`: the metric direction as a flag, the groups spread.

    Args:
        layout: An evaluation record at version 2.

    Returns:
        The record, with `direction` True for `higher` and False for `lower`
        and the `GROUPS` objects spread back to top-level keys.
    """
    if layout.get("direction") in {HIGHER, LOWER}:
        layout["direction"] = layout["direction"] == HIGHER
    return GROUPS.downgrade(layout)


MIGRATIONS = (
    records.Migration(
        renames={
            "run_id": "run",
            "metric_greater_is_better": "direction",
            "selection_policy": "policy",
            "fit_health": "health",
            "cv_confidence": "confidence",
            "baseline_cv_score": "baseline",
            "test_metrics": "metrics",
            "weighted_test_metrics": "weighted",
            "semi_supervised": "unlabeled",
            "fairness_mitigation": "mitigation",
            "execution_backend": "backend",
            "pareto_front": "pareto",
            "evaluation_prevalence": "prevalence",
            "success_criteria": "criteria",
            "streamed_locked": "streamed",
            "split_strategy": "validation",
            "net_benefit": "benefit",
            "resource_limits": "limits",
            "skipped_for_time": "skipped",
            "test_set_opened": "opened",
            "elapsed_seconds": "elapsed",
            "learning_curve": "learning",
            "nested_cv": "nested",
            "bias_variance": "decomposition",
            "forecast_backtest": "backtest",
            "quantile_coverage": "quantiles",
            "model_card": "card",
            "reliability_plan": "reliability",
            "explanation_evidence": "explanations",
            "artifact_verification": "verification",
        },
        change=orient,
        revert=disorient,
    ),
)
"""The version steps of `evaluation.json`, oldest first: version 2 writes one
word per key, `selected.*`, `rows.*`, `digests.split`, `release.*`, and the
metric direction as `higher` or `lower`."""


def evaluation(raw: Mapping[str, Any]) -> records.Layout:
    """Reads an `evaluation.json` record of any version at version 2.

    Version 1 carried no version key; it is detected as version 1.

    Args:
        raw: The stored record, or a record built at version 1.

    Returns:
        The record at the current version, with `schema_version` set.

    Raises:
        InputError: If the stored version is not a positive integer or is
            newer than this build.
    """
    return records.load("evaluation", MIGRATIONS, raw)


@dataclasses.dataclass(slots=True)
class Run:
    """Everything a completed run produced.

    Attributes:
        run_id: Run identifier.
        run_dir: Directory holding all artifacts.
        state: Terminal state name.
        plan: The executed experiment plan.
        profile: Dataset profile.
        trials: All trials in execution order.
        selected_model: Winning model name.
        evaluation: Final evaluation record (also saved as JSON).
        bundle: The packaged inference pipeline.
        diagnostics: Post-selection diagnostic reports by name.
        estimation: Separate parameter-estimation evidence, when explicitly
            declared on the project; never used for model selection.
    """

    run_id: str
    run_dir: pathlib.Path
    state: str
    plan: experiments.Plan
    profile: scmle.contracts.data.Portrait
    trials: list[experiments.Trial]
    selected_model: str
    evaluation: dict[str, Any]
    bundle: scmle.artifacts.bundle.Bundle
    diagnostics: dict[str, dict[str, Any]] = dataclasses.field(
        default_factory=dict
    )
    estimation: parameter_estimation.Estimation | None = None

    def salience(self) -> pandas.DataFrame:
        """Returns permutation importance, most important feature first.

        Returns:
            One row per measured feature, ranked by mean metric drop, with the
            columns `feature`, `importance` and `std`. The diagnostic may cover
            only a random subset of the features when its time allowance ran
            out. The frame is empty if the diagnostic was not produced.
        """
        report = self.diagnostics.get("permutation_importance", {})
        return pandas.DataFrame(report.get("features", []))

    def diagnose(self, trial: experiments.Trial) -> str:
        """Gives the fit-health verdict of a trial.

        Args:
            trial: A trial of this run.

        Returns:
            The verdict, or an empty string for a trial that did not succeed or
            whose fit health cannot be assessed.
        """
        if trial.status is not experiments.Termination.OK:
            return ""
        health = fit_health.sanity(
            trial, self.trials, self.evaluation["direction"] == HIGHER
        )
        return "" if health is None else health.verdict

    def leaderboard(self) -> pandas.DataFrame:
        """Returns trials ranked best-first, failed/skipped trials last.

        Returns:
            One row per trial: the trial's fields plus its fit-health verdict
            and whether it was selected. The columns in `LEADERBOARD_DETAILS`
            are left out and the parameters are JSON text. Successful trials
            come first, ordered by mean score; the others follow in execution
            order.
        """
        board = pandas.DataFrame([t.to_dict() for t in self.trials])
        board["fit_health"] = [self.diagnose(trial) for trial in self.trials]
        board = board.drop(columns=list(LEADERBOARD_DETAILS))
        board["params"] = board["params"].map(json.dumps)
        board["selected"] = (
            board["identifier"] == self.evaluation["selected"]["trial"]
        )
        succeeded = board["status"] == str(experiments.Termination.OK)
        ranked = board[succeeded].sort_values(
            "score",
            ascending=self.evaluation["direction"] == LOWER,
        )
        return pandas.concat([ranked, board[~succeeded]]).reset_index(drop=True)

    def save(
        self,
        directory: str | pathlib.Path,
        signing_key: str | bytes | None = None,
    ) -> pathlib.Path:
        """Exports the model bundle to `directory`.

        Args:
            directory: Destination directory.
            signing_key: HMAC key; defaults to `$SCMLE_SIGNING_KEY`. Signed
                bundles load without `trusted=True`.

        Returns:
            The directory the bundle was written to.
        """
        return self.bundle.save(directory, signing_key)

    def predict(
        self, data: str | pathlib.Path | pandas.DataFrame
    ) -> numpy.ndarray:
        """Predicts with the best model.

        Args:
            data: DataFrame or path to a supported tabular dataset.

        Returns:
            Predictions, one per row.
        """
        return self.bundle.predict(data)

    def predict_proba(
        self, data: str | pathlib.Path | pandas.DataFrame
    ) -> numpy.ndarray:
        """Predicts class probabilities with the best model.

        Args:
            data: DataFrame or path to a supported tabular dataset.

        Returns:
            Class probabilities, one row per input row.
        """
        return self.bundle.predict_proba(data)

    def explain(
        self,
        data: str | pathlib.Path | pandas.DataFrame,
        background: str | pathlib.Path | pandas.DataFrame,
        class_label: Any = None,
    ) -> pandas.DataFrame:
        """Explains predictions of the best model against a background.

        Args:
            data: DataFrame or path to the rows to explain.
            background: DataFrame or path to the reference rows.
            class_label: Class to explain for a classifier; if None, the
                positive class of a binary model (a multiclass model needs it).

        Returns:
            The explanation as a frame.
        """
        return self.bundle.explain(data, background, class_label=class_label)

    @property
    def accomplished(self) -> bool:
        """Whether every stated success criterion was met (True if none)."""
        return all(c["met"] for c in self.evaluation["criteria"])

    def suspect(self) -> list[str]:
        """Lists flagged columns that are neither quarantined nor attested.

        A column the leakage screen flagged stays a suspicion until the run's
        `admissibility_table.json` records it as quarantined or attested.
        Without that file, every flagged column is listed.

        Returns:
            Sorted column names; empty when nothing is outstanding.
        """
        suspects = set(map(str, self.profile.leakage.get("suspects", ())))
        path = pathlib.Path(self.run_dir) / "admissibility_table.json"
        if suspects and path.exists():
            record = json.loads(path.read_text(encoding="utf-8"))
            for decision in record.get("decisions", ()):
                settled = decision.get("admission") == "quarantine" or (
                    decision.get("tier") == "attested"
                )
                if settled:
                    suspects.discard(str(decision.get("name")))
        return sorted(suspects)

    def tally(self) -> dict[str, int]:
        """Counts the claims in `confirmed_claims.json` by evidence tier.

        Returns:
            Tier name to count (a claim without a tier counts as
            `unverified`); empty if the run wrote no `confirmed_claims.json`.
        """
        path = pathlib.Path(self.run_dir) / "confirmed_claims.json"
        if not path.exists():
            return {}
        record = json.loads(path.read_text(encoding="utf-8"))
        counts: dict[str, int] = {}
        for claim in record.get("claims", ()):
            tier = str(claim.get("tier", "unverified"))
            counts[tier] = counts.get(tier, 0) + 1
        return counts

    def summary(self) -> str:
        """Returns a one-line human-readable result.

        Returns:
            The run id, selected model, cross-validation score, test metrics,
            whether the success criteria were met, claim tiers and any warning.
            A result with unattested flagged columns or an incomplete leakage
            screen is marked provisional.
        """
        evaluation = self.evaluation
        test = ", ".join(
            f"{name}={value:.4f}"
            for name, value in evaluation["metrics"].items()
        )
        unattested = self.suspect()
        warning = self.profile.admonition
        reasons = []
        if unattested:
            reasons.append(
                f"{len(unattested)} flagged column(s) not attested: "
                f"{', '.join(unattested)}"
            )
        if warning:
            reasons.append("leakage screen incomplete")
        label = f" (PROVISIONAL: {'; '.join(reasons)})" if reasons else ""
        text = (
            f"{self.run_id}: selected {self.selected_model} "
            f"(cv {evaluation['metric']}="
            f"{evaluation['selected']['score']:.4f}); test{label}: {test}"
        )
        missed = [
            f"{c['metric']} {c['achieved']:.4f} vs required {c['required']}"
            for c in evaluation["criteria"]
            if not c["met"]
        ]
        if evaluation["criteria"]:
            text += (
                "; success criteria met"
                if not missed
                else "; success criteria NOT met: " + ", ".join(missed)
            )
        tiers = self.tally()
        if tiers:
            text += "; claims: " + ", ".join(
                f"{count} {tier}" for tier, count in sorted(tiers.items())
            )
        if warning:
            text += f"; WARNING {warning}"
        return text
