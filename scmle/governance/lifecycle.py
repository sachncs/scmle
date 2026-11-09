"""Lifecycle automation: monitor, retrain a challenger, compare, promote."""

import dataclasses
import hashlib
import json
import pathlib
from typing import Any

import pandas as pd

import scmle
from scmle import durability
from scmle.api import results
from scmle.contracts import errors
from scmle.governance import registry as registry_module
from scmle.monitoring import champion
from scmle.monitoring import monitor as monitor_module

HOLDOUT_FRACTION = 0.3
STATE_SUFFIX = ".lifecycle.json"


def split_fresh(
    fresh: pd.DataFrame, schema: Any
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Splits fresh rows into training and holdout rows.

    Data with a time column (named by the schema and present in `fresh`) are
    split out of time: the rows are ordered by that column and the newest
    ones after the first ``int(n * (1 - HOLDOUT_FRACTION))`` are the holdout.
    Otherwise data with a group column are split by whole groups (about
    `HOLDOUT_FRACTION` of the groups, chosen by a seeded shuffle, form the
    holdout), unless that would leave one side empty, as with a single
    group. All other data, and that case, are split by a seeded shuffle of
    the rows. A random shuffle of temporal or grouped rows would leak the
    holdout into training and overstate the challenger.

    Args:
        fresh: The labelled fresh rows.
        schema: The champion's schema; only its `time_column` and
            `group_column` attributes are read, and either may be absent.

    Returns:
        The training and holdout frames.
    """
    time_name = getattr(schema, "time_column", None)
    group_name = getattr(schema, "group_column", None)
    if time_name and time_name in fresh.columns:
        order = pd.to_datetime(fresh[time_name], errors="coerce")
        ordered = fresh.iloc[order.argsort(kind="stable")].reset_index(
            drop=True
        )
        cut = int(len(ordered) * (1 - HOLDOUT_FRACTION))
        return ordered.iloc[:cut], ordered.iloc[cut:]
    if group_name and group_name in fresh.columns:
        names = pd.Series(fresh[group_name].astype(str).unique())
        names = names.sample(frac=1.0, random_state=0).reset_index(drop=True)
        held = set(names.iloc[int(len(names) * (1 - HOLDOUT_FRACTION)) :])
        mask = fresh[group_name].astype(str).isin(held)
        if mask.any() and not mask.all():
            return fresh[~mask].reset_index(drop=True), fresh[mask].reset_index(
                drop=True
            )
    shuffled = fresh.sample(frac=1.0, random_state=0).reset_index(drop=True)
    cut = int(len(shuffled) * (1 - HOLDOUT_FRACTION))
    return shuffled.iloc[:cut], shuffled.iloc[cut:]


@dataclasses.dataclass(frozen=True)
class RetrainPolicy:
    """When a drifting model may be retrained, so checks do not pile up.

    Attributes:
        min_rows: Labeled rows needed to train a challenger and compare it.
        cooldown_rows: After an attempt, fresh rows that must have arrived
            since (counting those offered now) before the next one (a
            rejected challenger is not retried on every check).
    """

    min_rows: int = 200
    cooldown_rows: int = 500


@dataclasses.dataclass(frozen=True)
class RollbackPolicy:
    """When a promoted model must be taken back.

    Attributes:
        min_rows: Labeled rows needed to judge a promotion.
        tolerance: Score the new champion may lose to the previous one, in
            the metric's units, before the loss counts. The rollback
            happens only when the whole 95% interval of its advantage lies
            below minus this tolerance. The default 0.01 keeps routine
            retraining noise (a model refit on a new sample nearly always
            differs by a little) from undoing promotions; set it to the
            smallest loss you would actually act on.
    """

    min_rows: int = 100
    tolerance: float = 0.01


@dataclasses.dataclass(frozen=True)
class RollbackOutcome:
    """What a rollback check concluded and did.

    Attributes:
        rolled_back: Whether the previous version is champion again.
        comparison: New champion (as challenger) against the previous one;
            None when no comparison was made (no previous version or too few
            rows).
        restored_version: The version now serving, if rolled back.
        reason: Human-readable explanation.
    """

    rolled_back: bool
    comparison: champion.Comparison | None
    restored_version: int | None
    reason: str


def fingerprint(frame: pd.DataFrame) -> str:
    """Short digest of the rows, to recognise data already acted on.

    Args:
        frame: The rows to digest; the index is ignored, row order is not.

    Returns:
        The first 16 hex characters of a SHA-256 over the row hashes.
    """
    hashes = pd.util.hash_pandas_object(frame, index=False).to_numpy()
    return hashlib.sha256(hashes.tobytes()).hexdigest()[:16]


@dataclasses.dataclass(frozen=True)
class LifecycleOutcome:
    """What one lifecycle review concluded and did.

    Attributes:
        monitoring: Monitoring report for the current champion.
        retrained: Whether a challenger was trained.
        comparison: Champion-challenger comparison, if a challenger exists.
        promoted_version: Always None as `LifecycleManager.review` returns
            it, since `review` only stages a challenger; promotion is a
            separate call.
        notes: Human-readable explanation.
        skipped: Why a recommended retrain did not run (`already_handled`,
            `cooldown` or `too_few_rows`), if it did not.
    """

    monitoring: monitor_module.MonitoringReport
    retrained: bool
    comparison: champion.Comparison | None
    promoted_version: int | None
    notes: list[str]
    skipped: str | None = None


class LifecycleManager:
    """Keeps a registered model fresh and safe to serve."""

    def __init__(
        self,
        registry: registry_module.ModelRegistryStore,
        model: str,
        work_dir: str,
        policy: RetrainPolicy | None = None,
    ) -> None:
        """Initializes the manager.

        Args:
            registry: Model registry.
            model: Registered model name.
            work_dir: Directory for challenger training runs.
            policy: When a recommended retrain may actually run.
        """
        self.registry = registry
        self.model = model
        self.work_dir = work_dir
        self.policy = policy or RetrainPolicy()
        self.rollout_decides = False

    def defer_to_rollout(self) -> None:
        """Makes the bootstrap promotion and rollback rules descriptive.

        Once a charter rollout is bound to this manager (`RolloutHook` calls
        this on construction), the e-processes of `scmle.decision.rollout`
        are the only deciders. `review` then stages the retrained challenger
        whatever its bootstrap comparison says, so the e-process can test
        it, and `check_rollback` never moves the alias; both audit the
        bootstrap result as a descriptive diagnostic (`descriptive: true`,
        `decides: false`). Without a bound rollout nothing changes.
        """
        self.rollout_decides = True

    @property
    def state_path(self) -> pathlib.Path:
        """Where the manager remembers its last attempt."""
        return self.registry.root / f"{self.model}{STATE_SUFFIX}"

    def read_state(self) -> dict[str, Any]:
        """Reads the last attempt's data digest and the rows seen since.

        Returns:
            A dict with `last_digest`, the digest of the data of the last
            retraining attempt (None if none), and `rows_since_attempt`;
            the saved content as it is when the state file exists.
        """
        with durability.path_lock(self.state_path):
            if not self.state_path.exists():
                return {"last_digest": None, "rows_since_attempt": 0}
            return json.loads(self.state_path.read_text(encoding="utf-8"))

    def write_state(self, state: dict[str, Any]) -> None:
        """Saves the state atomically.

        Args:
            state: The state to save, as returned by `read_state`.
        """
        durability.write_atomically(
            self.state_path, json.dumps(state).encode("utf-8")
        )

    def skip_reason(
        self, state: dict[str, Any], fresh: pd.DataFrame
    ) -> str | None:
        """Why a recommended retrain should not run now, if any reason holds.

        Args:
            state: The saved state, as returned by `read_state`.
            fresh: The fresh rows offered for retraining.

        Returns:
            `already_handled` if these rows were already acted on,
            `too_few_rows` if they are fewer than the policy's `min_rows`,
            `cooldown` if there was an attempt and the rows since it plus
            these are fewer than `cooldown_rows`, else None; the checks run
            in that order.
        """
        if fingerprint(fresh) == state["last_digest"]:
            return "already_handled"
        if len(fresh) < self.policy.min_rows:
            return "too_few_rows"
        attempted = state["last_digest"] is not None
        pending = state["rows_since_attempt"] + len(fresh)
        if attempted and pending < self.policy.cooldown_rows:
            return "cooldown"
        return None

    def register_run(
        self,
        result: results.RunResult,
        promote: bool = False,
        approval: registry_module.PromotionApproval | None = None,
        signing_key: str | bytes | None = None,
    ) -> registry_module.VersionRecord:
        """Registers a finished run and optionally makes it the champion.

        The run's bundle is first exported to ``export`` in its run
        directory. With `promote` the registry refuses (with an
        `ArtifactError`) a missing or invalid approval.

        Args:
            result: A completed run.
            promote: Whether to also point the champion alias at it. By
                default the version is only registered and no alias moves.
            approval: Independent approval for champion promotion.
            signing_key: HMAC key for the approval signature.

        Returns:
            The registered version.
        """
        bundle_dir = result.save(result.run_dir / "export")
        record = self.registry.register(
            self.model,
            bundle_dir,
            metrics=result.evaluation["test_metrics"],
            config=result.plan.to_dict() | {"project": read_run_config(result)},
            run_id=result.run_id,
            trusted=True,  # exported moments ago by this very process
        )
        if promote:
            self.registry.promote(
                self.model,
                record.version,
                approval=approval,
                signing_key=signing_key,
            )
        return record

    def promote_challenger(
        self,
        approval: registry_module.PromotionApproval,
        signing_key: str | bytes | None = None,
    ) -> registry_module.VersionRecord:
        """Promotes the staged challenger after independent review.

        With a guarded registry (`ModelRegistryStore.attach_guard`, e.g. via
        `Loop.guard_registry`) it also needs a human attestation for the exact
        model and version, else `AttestationRequired` is raised and nothing
        moves. An `ArtifactError` is raised if there is no staged challenger
        or the approval is invalid.

        Args:
            approval: The independent approval of the challenger bundle.
            signing_key: HMAC key for the approval signature.

        Returns:
            The record of the version now champion.
        """
        record = self.registry.resolve(self.model, registry_module.CHALLENGER)
        self.registry.promote(
            self.model,
            record.version,
            approval=approval,
            signing_key=signing_key,
        )
        self.registry.audit(
            "lifecycle.challenger_promoted",
            model=self.model,
            version=record.version,
            reviewer=approval.reviewer,
            approval_evidence=list(approval.evidence),
        )
        return record

    def review(
        self,
        fresh: pd.DataFrame,
        target: str,
        retrain: bool = True,
        project_options: dict[str, Any] | None = None,
    ) -> LifecycleOutcome:
        """Monitors the champion on fresh labeled data and acts on advice.

        If monitoring recommends retraining, a challenger is trained on part
        of the fresh data and compared with the champion on the held-out
        remainder. A challenger whose bootstrap comparison says `promote` is
        staged (the `challenger` alias), but champion promotion requires a
        separate call to `promote_challenger`. A recommended retrain is
        skipped, and noted in the outcome, when the rows were already acted
        on, are too few or arrive within the cooldown. When a charter
        rollout is bound (`defer_to_rollout`) the challenger is staged
        whatever the comparison says. The attempt is recorded in the state
        file and the audit trail.

        Args:
            fresh: Recent served rows including the true target column.
            target: Name of the target column.
            retrain: Whether a recommended retrain may run; when False the
                champion is only monitored.
            project_options: Extra `MLProject` options for the challenger.

        Returns:
            The outcome.
        """
        with durability.path_lock(self.state_path):  # One attempt at a time.
            return self.review_locked(fresh, target, retrain, project_options)

    def review_locked(
        self,
        fresh: pd.DataFrame,
        target: str,
        retrain: bool,
        project_options: dict[str, Any] | None,
    ) -> LifecycleOutcome:
        """Does the work of `review`; the caller holds the state lock.

        Args:
            fresh: Fresh labelled rows, including the target.
            target: Name of the target column.
            retrain: Whether a recommended retrain may run.
            project_options: Options for the retraining project.

        Returns:
            The outcome: the monitoring report, whether a retrain ran, the
            champion-challenger comparison if there was one, and the notes.
        """
        current = self.registry.load(self.model)
        features = fresh.drop(columns=[target])
        report = monitor_module.ModelMonitor(current).assess(
            features, fresh[target]
        )
        notes = list(report.recommendation.reasons)
        state = self.read_state()
        self.registry.audit(
            "lifecycle.review",
            model=self.model,
            advice=report.recommendation.action,
            reasons=notes,
            rows=len(fresh),
        )
        if report.recommendation.action != "retrain" or not retrain:
            self.note_rows(state, fresh)
            return LifecycleOutcome(
                report, False, None, None, notes or ["healthy"]
            )
        skip = self.skip_reason(state, fresh)
        if skip is not None:
            self.registry.audit(
                "lifecycle.retrain_skipped",
                model=self.model,
                reason=skip,
                rows=len(fresh),
            )
            self.note_rows(state, fresh)
            notes.append(f"retraining skipped: {skip}")
            return LifecycleOutcome(
                report, False, None, None, notes, skipped=skip
            )
        digest = fingerprint(fresh)
        self.registry.audit(
            "lifecycle.retrain_started",
            model=self.model,
            rows=len(fresh),
            data_digest=digest,
            reasons=notes,
        )
        self.write_state({"last_digest": digest, "rows_since_attempt": 0})
        train, holdout = split_fresh(fresh, current.schema)
        options = dict(project_options or {})
        options.setdefault("task", current.schema.task)
        options.setdefault("metric", current.info.metric)
        challenger_run = scmle.MLProject(
            train, target, output_dir=self.work_dir, **options
        ).run()
        comparison = champion.ChampionChallenger(
            current, challenger_run.bundle
        ).compare(holdout.drop(columns=[target]), holdout[target])
        evidence = dataclasses.asdict(comparison)
        if self.rollout_decides:
            self.registry.audit(
                "lifecycle.bootstrap_diagnostic",
                model=self.model,
                stage="review",
                descriptive=True,
                decides=False,
                **evidence,
            )
        if comparison.decision != "promote" and not self.rollout_decides:
            self.registry.audit(
                "lifecycle.challenger_rejected",
                model=self.model,
                champion_version=self.registry.resolve(self.model).version,
                **evidence,
            )
            notes.append("challenger was not clearly better; champion kept")
            return LifecycleOutcome(report, True, comparison, None, notes)
        record = self.register_run(challenger_run)
        self.registry.promote(
            self.model, record.version, registry_module.CHALLENGER
        )
        self.registry.audit(
            "lifecycle.challenger_staged",
            model=self.model,
            version=record.version,
            **evidence,
        )
        if self.rollout_decides:
            notes.append(
                f"bootstrap comparison says {comparison.decision!r} "
                "(descriptive only); the rollout e-process decides promotion"
            )
        notes.append(
            f"staged challenger as v{record.version}; independent approval "
            "required before champion promotion"
        )
        return LifecycleOutcome(report, True, comparison, None, notes)

    def check_rollback(
        self,
        fresh: pd.DataFrame,
        target: str,
        policy: RollbackPolicy | None = None,
        approval: registry_module.PromotionApproval | None = None,
        signing_key: str | bytes | None = None,
    ) -> RollbackOutcome:
        """Takes the previous champion back if the new one is clearly worse.

        With a charter rollout bound (`defer_to_rollout`) this is a
        descriptive diagnostic: it audits the comparison and never moves the
        alias or asks for approval.

        Both versions score the same fresh labeled rows. The previous version
        returns only when the whole 95% interval of the new champion's
        advantage over it lies below minus the policy's tolerance; anything
        less clear leaves the new champion in place. Once there is a previous
        version and enough rows, the evidence goes to the audit trail either
        way.

        With a guarded registry the rollback also needs a human attestation
        naming this rollback; without one `AttestationRequired` is raised
        after the evidence is audited and the champion stays. An invalid
        approval raises `ArtifactError` the same way.

        Args:
            fresh: Recent served rows including the true target.
            target: Name of the target column.
            policy: Evidence required; defaults to `RollbackPolicy()`.
            approval: Independent approval for the rollback promotion.
            signing_key: HMAC key for the approval signature.

        Returns:
            The outcome: whether the previous version was restored, the
            comparison, the restored version and the explanation. It is not
            restored when there is no previous version, too few rows, the new
            champion holds up, no approval was given, or a rollout decides.
        """
        rule = policy or RollbackPolicy()
        aliases = self.registry.aliases(self.model)
        previous = aliases.get(registry_module.PREVIOUS)
        if previous is None:
            return RollbackOutcome(False, None, None, "no previous version")
        if len(fresh) < rule.min_rows:
            return RollbackOutcome(
                False, None, None, f"fewer than {rule.min_rows} labeled rows"
            )
        current = self.registry.load(self.model)
        old = self.registry.load(self.model, previous)
        comparison = champion.ChampionChallenger(old, current).compare(
            fresh.drop(columns=[target]), fresh[target]
        )
        worse = comparison.interval[1] < -rule.tolerance
        if self.rollout_decides:
            self.registry.audit(
                "lifecycle.rollback_check",
                model=self.model,
                current=aliases[registry_module.CHAMPION],
                previous=previous,
                rollback_required=worse,
                rolled_back=False,
                descriptive=True,
                decides=False,
                **dataclasses.asdict(comparison),
            )
            return RollbackOutcome(
                False,
                comparison,
                None,
                "descriptive: the bootstrap interval "
                f"[{comparison.interval[0]:.4f}, {comparison.interval[1]:.4f}]"
                f" {'is' if worse else 'is not'} below the tolerance; "
                "nothing moved, the rollback e-process decides",
            )
        restore = worse and approval is not None
        restored = False
        try:
            if restore and approval is not None:
                self.restore_previous(
                    approval,
                    signing_key,
                    aliases[registry_module.CHAMPION],
                    previous,
                )
                restored = True
        finally:
            self.registry.audit(
                "lifecycle.rollback_check",
                model=self.model,
                current=aliases[registry_module.CHAMPION],
                previous=previous,
                rollback_required=worse,
                rolled_back=restored,
                **dataclasses.asdict(comparison),
            )
        if not worse:
            return RollbackOutcome(
                False, comparison, None, "the new champion holds up"
            )
        if not restore:
            self.registry.audit(
                "lifecycle.rollback_pending_approval",
                model=self.model,
                version=previous,
                reason="new champion is clearly worse",
            )
            return RollbackOutcome(
                False,
                comparison,
                None,
                "rollback requires independent approval",
            )
        return RollbackOutcome(
            True,
            comparison,
            previous,
            f"restored v{previous}: the new champion trails it by "
            f"{-comparison.advantage:.4f} (interval "
            f"[{comparison.interval[0]:.4f}, {comparison.interval[1]:.4f}])",
        )

    def restore_previous(
        self,
        approval: registry_module.PromotionApproval,
        signing_key: str | bytes | None = None,
        current: int | None = None,
        previous: int | None = None,
    ) -> int:
        """Points the champion alias back at the previous version.

        The decision to do so (the bootstrap interval of `check_rollback` or
        the rollback e-process of `scmle.governance.rollout_hook`) is made by
        the caller; this is the one place the alias moves, with the same
        independent approval and the guarded registry's attestation naming
        this rollback.

        Args:
            approval: Independent approval for the previous version.
            signing_key: HMAC key for the approval signature.
            current: The version being replaced (read when omitted).
            previous: The version restored (read when omitted).

        Returns:
            The restored version.

        Raises:
            ArtifactError: If there is no previous version or the approval
                is invalid.
        """
        aliases = self.registry.aliases(self.model)
        target = (
            previous
            if previous is not None
            else aliases.get(registry_module.PREVIOUS)
        )
        if target is None:
            raise errors.ArtifactError("no previous version to restore")
        replaced = (
            current
            if current is not None
            else aliases.get(registry_module.CHAMPION)
        )
        self.registry.promote(
            self.model,
            target,
            approval=approval,
            signing_key=signing_key,
            guard_detail={
                "rollback": {
                    "model": self.model,
                    "from_version": replaced,
                    "to_version": target,
                }
            },
        )
        return int(target)

    def note_rows(self, state: dict[str, Any], fresh: pd.DataFrame) -> None:
        """Counts fresh rows toward the cooldown after an attempt.

        Does nothing if there was no attempt or these rows are the ones last
        acted on.

        Args:
            state: The saved state, updated in place and written.
            fresh: The fresh rows seen.
        """
        if (
            state["last_digest"] is None
            or fingerprint(fresh) == (state["last_digest"])
        ):
            return
        state["rows_since_attempt"] += len(fresh)
        self.write_state(state)


def read_run_config(result: results.RunResult) -> dict[str, Any]:
    """Reads the saved project configuration of a run.

    Args:
        result: The completed run.

    Returns:
        The content of the run's `config.json`.

    Raises:
        ArtifactError: If the run has no `config.json`.
    """
    path = result.run_dir / "config.json"
    if not path.exists():
        raise errors.ArtifactError("run has no config.json")
    return json.loads(path.read_text(encoding="utf-8"))
