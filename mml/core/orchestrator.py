"""Orchestrator: compiles a project spec into the staged workflow and runs it.

Stages (state machine): VALIDATING -> PROFILING -> PLANNING -> TRAINING -> SELECTING ->
TESTING -> PACKAGING -> COMPLETED, with FAILED / CANCELLED / BLOCKED as exits. Every
transition emits an event; finished trials are checkpointed to disk as they complete.
"""

from __future__ import annotations

import html
import json
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import distributions
from pathlib import Path

import pandas as pd
from sklearn.model_selection import cross_validate

from mml.api.results import RunResult
from mml.artifacts.bundle import ModelBundle
from mml.contracts.data import DatasetManifest, DatasetProfile
from mml.contracts.errors import BlockedError, MagicMLError
from mml.contracts.events import Event, EventLog
from mml.contracts.experiments import ExperimentPlan, TrialResult
from mml.contracts.tasks import ProjectSpec, TaskType
from mml.core.state import RunState, StateMachine
from mml.data.preprocessing import build_pipeline
from mml.data.profiling import profile_dataset, resolve_task
from mml.data.readers import build_manifest, read_table
from mml.data.splitting import make_split
from mml.evaluation.metrics import resolve_metric, score_estimator
from mml.intelligence.planner import make_plan
from mml.models.registry import get_model


class RunCancelled(MagicMLError):
    category = "cancelled"


@dataclass
class Prepared:
    df: pd.DataFrame
    manifest: DatasetManifest
    task: TaskType
    profile: DatasetProfile
    plan: ExperimentPlan


def _json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str))


def _new_run_dir(root: Path) -> tuple[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d")
    n = 1
    while True:
        run_id = f"run_{stamp}_{n:03d}"
        try:
            (root / run_id).mkdir()
            return run_id, root / run_id
        except FileExistsError:
            n += 1


class Orchestrator:
    def __init__(self, spec: ProjectSpec, frame: pd.DataFrame | None = None) -> None:
        self.spec = spec
        self._frame = frame
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    # ---- control plane ------------------------------------------------
    def _ingest(self) -> tuple[pd.DataFrame, DatasetManifest]:
        df = self._frame.copy() if self._frame is not None else read_table(self.spec.data.uri)
        return df, build_manifest(df, self.spec.data.target, self.spec.data.uri)

    def _profile(self, df, manifest) -> tuple[TaskType, DatasetProfile]:
        task = resolve_task(self.spec, df[self.spec.data.target])
        resolve_metric(self.spec.metric, task)  # fail early on incompatible metric
        return task, profile_dataset(df, self.spec, manifest, task)

    def prepare(self, allowed_models: tuple[str, ...] | None = None) -> Prepared:
        df, manifest = self._ingest()
        task, profile = self._profile(df, manifest)
        plan = make_plan(self.spec, profile, task, allowed_models)
        return Prepared(df, manifest, task, profile, plan)

    # ---- full run -----------------------------------------------------
    def run(
        self, output_dir: str | Path = "runs", approved: ExperimentPlan | None = None
    ) -> RunResult:
        spec = self.spec
        run_id, run_dir = _new_run_dir(Path(output_dir))
        log = EventLog(run_dir / "events.jsonl")
        sm = StateMachine(run_id, log, spec.config_hash())
        _json(run_dir / "config.json", json.loads(spec.canonical_json()))
        (run_dir / "environment.lock").write_text(
            "\n".join(sorted({f"{d.metadata['Name']}=={d.version}" for d in distributions()}))
        )
        try:
            return self._run(sm, run_dir, approved)
        except RunCancelled:
            sm.transition(RunState.CANCELLED, "cancelled by user")
            raise
        except KeyboardInterrupt as exc:
            sm.transition(RunState.CANCELLED, "interrupted")
            raise RunCancelled("interrupted", {"run_dir": str(run_dir)}) from exc
        except BlockedError as exc:
            exc.details["run_dir"] = str(run_dir)
            sm.transition(RunState.BLOCKED, str(exc))
            raise
        except MagicMLError as exc:
            exc.details["run_dir"] = str(run_dir)
            sm.transition(RunState.FAILED, f"{exc.category}: {exc}")
            raise
        except Exception as exc:
            sm.transition(RunState.FAILED, f"internal: {exc!r}")
            raise MagicMLError(str(exc), {"run_dir": str(run_dir)}) from exc

    def _check_cancel(self, run_dir: Path) -> None:
        if self._cancel.is_set():
            raise RunCancelled("run cancelled", {"run_dir": str(run_dir)})

    def _run(self, sm: StateMachine, run_dir: Path, approved: ExperimentPlan | None) -> RunResult:
        spec = self.spec
        t0 = time.monotonic()
        deadline = t0 + spec.resources.time_budget_seconds

        sm.transition(RunState.VALIDATING, "ingest dataset")
        df, manifest = self._ingest()
        _json(run_dir / "dataset_manifest.json", vars(manifest))

        sm.transition(RunState.PROFILING, "profile dataset")
        task, profile = self._profile(df, manifest)
        _json(
            run_dir / "dataset_profile.json",
            {
                "manifest": vars(manifest),
                "columns": profile.column_statistics,
                "quality_findings": profile.quality_findings,
                "risk_findings": profile.risk_findings,
                "target": profile.target_summary,
                "feature_plan": {
                    "numeric": profile.feature_plan.numeric,
                    "categorical": profile.feature_plan.categorical,
                    "dropped": profile.feature_plan.dropped,
                },
            },
        )
        if spec.leakage_policy == "block" and profile.risk_findings:
            leaks = [r for r in profile.risk_findings if "leakage" in r]
            if leaks:
                raise BlockedError("unresolved leakage risk: " + "; ".join(leaks))

        sm.transition(RunState.PLANNING, "build split manifest and execution plan")
        manifest_split, dev_X, dev_y, locked = make_split(df, spec, task)
        _json(run_dir / "validation_manifest.json", manifest_split.to_dict())
        allowed = approved.candidate_models if approved is not None else None
        plan = make_plan(spec, profile, task, allowed)
        _json(run_dir / "execution_plan.json", plan.to_dict())
        (run_dir / "diagnostics").mkdir()
        _json(run_dir / "diagnostics" / "decision_log.json", [vars(d) for d in plan.decisions])
        metric = resolve_metric(spec.metric, task)
        fplan = profile.feature_plan
        folds = [(list(a), list(b)) for a, b in manifest_split.folds]

        sm.transition(RunState.TRAINING, f"{len(plan.candidate_models)} candidates")
        trials_dir = run_dir / "trials"
        trials_dir.mkdir()
        trials: list[TrialResult] = []
        for i, name in enumerate(plan.candidate_models, 1):
            self._check_cancel(run_dir)
            tr = TrialResult(f"trial_{i:03d}", name, "skipped", metric.name)
            if time.monotonic() >= deadline and i > 1:
                tr.error = "time budget exhausted before this trial started"
            else:
                start = time.monotonic()
                try:
                    pipe = build_pipeline(fplan, get_model(name), task, spec.validation.seed)
                    res = cross_validate(
                        pipe, dev_X, dev_y, cv=folds, scoring=metric.scorer, error_score="raise"
                    )
                    scores = [metric.to_natural(float(s)) for s in res["test_score"]]
                    tr.fold_scores = scores
                    tr.score_mean = float(pd.Series(scores).mean())
                    tr.score_std = float(pd.Series(scores).std(ddof=0))
                    tr.status = "ok"
                except Exception as exc:  # failure isolation: one bad trial never ends the run
                    tr.status, tr.error = "failed", f"{type(exc).__name__}: {exc}"
                tr.fit_seconds = time.monotonic() - start
            trials.append(tr)
            _json(trials_dir / f"{tr.trial_id}.json", tr.to_dict())
            sm.log.emit(
                Event(
                    sm.run_id,
                    f"trial.{tr.status}",
                    tr.model_name,
                    spec.config_hash(),
                    (f"trials/{tr.trial_id}.json",),
                )
            )

        sm.transition(RunState.SELECTING, "apply selection policy")
        ok = [t for t in trials if t.status == "ok"]
        if not ok:
            raise MagicMLError("all trials failed", {"errors": [t.error for t in trials]})
        pool = [t for t in ok if t.model_name != "dummy"] or ok
        sign = 1 if metric.greater_is_better else -1
        best = min(
            pool,
            key=lambda t: (
                -sign * round(t.score_mean, 12),
                get_model(t.model_name).capabilities.cost,
                t.model_name,
            ),
        )
        policy = (
            "best mean CV score; ties -> cheaper model; dummy baseline excluded when others ran"
        )

        sm.transition(RunState.TESTING, f"final one-time test evaluation of {best.model_name}")
        self._check_cancel(run_dir)
        final = build_pipeline(fplan, get_model(best.model_name), task, spec.validation.seed)
        final.fit(dev_X, dev_y)
        test_X, test_y = locked.open(locked.issue_token())
        test_metrics = score_estimator(final, test_X, test_y, task)
        baseline = next((t for t in ok if t.model_name == "dummy"), None)
        evaluation = {
            "run_id": sm.run_id,
            "task": task.value,
            "metric": metric.name,
            "metric_greater_is_better": metric.greater_is_better,
            "selection_policy": policy,
            "selected_model": best.model_name,
            "selected_cv_score": best.score_mean,
            "selected_cv_std": best.score_std,
            "baseline_cv_score": baseline.score_mean if baseline else None,
            "test_metrics": test_metrics,
            "n_dev": len(dev_X),
            "n_test": len(test_X),
            "split_fingerprint": manifest_split.fingerprint,
            "split_strategy": manifest_split.strategy,
            "test_set_opened": 1,
            "elapsed_seconds": time.monotonic() - t0,
            "reproducibility": "same seed/data/environment replays; threaded backends may differ",
        }

        sm.transition(RunState.PACKAGING, "write bundle and reports")
        bundle = ModelBundle.create(
            final,
            fplan,
            dev_X,
            spec.data.target,
            task.value,
            metric.name,
            best.model_name,
            list(getattr(final, "classes_", [])) or None,
        )
        bundle.save(run_dir / "model")
        _json(run_dir / "evaluation.json", evaluation)
        result = RunResult(
            sm.run_id,
            run_dir,
            "COMPLETED",
            plan,
            profile,
            trials,
            best.model_name,
            evaluation,
            bundle,
            [vars(d) for d in plan.decisions],
        )
        board = result.leaderboard()
        board.to_parquet(run_dir / "leaderboard.parquet", index=False)
        (run_dir / "report.html").write_text(_report(result, board))
        sm.transition(RunState.COMPLETED, "run complete", ("model/", "evaluation.json"))
        return result


def _report(r: RunResult, board: pd.DataFrame) -> str:
    esc = html.escape
    metrics = "".join(
        f"<tr><td>{esc(k)}</td><td>{v:.4f}</td></tr>"
        for k, v in r.evaluation["test_metrics"].items()
    )
    warn = "".join(
        f"<li>{esc(w)}</li>" for w in (*r.profile.quality_findings, *r.profile.risk_findings)
    )
    return (
        f"<!doctype html><meta charset=utf-8><title>{esc(r.run_id)}</title>"
        f"<h1>{esc(r.run_id)}</h1><p>{esc(r.plan.summary()).replace(chr(10), '<br>')}</p>"
        f"<h2>Leaderboard</h2>{board.to_html(index=False, border=0)}"
        f"<h2>Final test metrics ({esc(r.selected_model)})</h2><table>{metrics}</table>"
        f"<h2>Findings</h2><ul>{warn or '<li>none</li>'}</ul>"
    )
