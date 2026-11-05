"""Generates the Phase 18 local evidence by running the real code.

    uv run python scripts/make_phase18_evidence.py plan/evidence/phase-18/local

Follows one model from a business question to observed outcomes:

1. `scmle frame` drafts a charter and a decision-cost model from a request
   about the Titanic extract (`tests/fixtures/titanic_extract.csv`, with a
   row number and a constant boarding date added because the extract has
   neither a key nor a moment; both are stated assumptions);
2. `scmle charter sign`, then `scmle run --charter` (a real run);
3. `scmle package` of the run's bundle;
4. the local registry registers the bundle, a signed approval promotes it;
5. `scmle loop init|observe|mature|step` feeds batches of the held-out test
   rows and their outcomes through the closed loop;
6. `scmle trail build|verify|show` joins all of it, and
   `scripts/trace_model.py` walks the result.

Writes ``<output>/trail`` (the only folder a reviewer needs), ``keys.json``
(demonstration keys), ``commands.json`` and ``summary.json``.

The three roles (analyst, stakeholder, reviewer) were played by the person
running this script with three different keys. No independent person signed
anything, the observed outcomes are a replay of held-out rows and not
production traffic, and nothing ran on a cloud.
"""

import json
import os
from pathlib import Path
import shutil
import subprocess  # nosec B404
import sys
import tempfile

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scmle.contracts import events  # noqa: E402
from scmle.governance import registry  # noqa: E402
from scmle.governance import release_state  # noqa: E402
from scmle.trail import inputs as trail_inputs  # noqa: E402
from scmle.trail import outcomes  # noqa: E402
from scmle.trail import records  # noqa: E402

KEYS = {
    "analyst": "phase18-demonstration-analyst-key",
    "stakeholder": "phase18-demonstration-stakeholder-key",
    "reviewer": "phase18-demonstration-reviewer-key",
}
MODEL = "titanic_survival"
REQUEST = (
    "Predict `survived` for each passenger at boarding, so the purser can "
    "decide whom to brief first."
)
COSTS = {
    "c_00": 0.0,
    "c_01": 5.0,
    "c_10": 1.0,
    "c_11": 0.0,
    "t_lo": 0.05,
    "t_hi": 0.5,
    "sources": ["human", "human", "human", "human"],
    "range_source": "human",
}
BATCH = 36
RAN = [
    "scmle frame, charter sign and a real charter run on the Titanic extract",
    "scmle package of the run's bundle",
    "the local registry: register, a signed approval, release state "
    "approved, champion alias moved",
    "scmle loop init, observe, mature, step and verify over seven batches of "
    "the held-out test rows",
    "scmle trail build, verify and show, then scripts/trace_model.py",
]
NOT_RUN = [
    "any reviewer who did not build the model: the analyst, stakeholder and "
    "reviewer roles were played by the author with three different keys, so "
    "the gate (an independent reviewer traces the model) is not met",
    "production traffic: the observed outcomes replay the run's held-out "
    "test rows through the loop, so they agree with the expectation by "
    "construction and show the mechanism, not the model's real-world cost",
    "a runtime that serves the model: the release state is approved, not "
    "production, because no control plane confirmed a deployment",
    "any cloud account (AWS or GCP)",
    "the F14 P0 items on both providers, so no production claim is made",
    "a prospective acceptance criterion: the cost ceiling was set after the "
    "held-out evaluation was seen",
]
COMMANDS: list[list[str]] = []


def scmle(*arguments: str, check: bool = True) -> str:
    """Runs the real command line and records the command."""
    shown = ["scmle", *arguments]
    COMMANDS.append(shown)
    environment = dict(os.environ, SCMLE_SIGNING_KEY=KEYS["analyst"])
    done = subprocess.run(  # nosec B603
        [sys.executable, "-m", "scmle.cli", *arguments],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if check and done.returncode != 0:
        raise SystemExit(f"{shown} failed:\n{done.stderr[-2000:]}")
    return done.stdout


def prepare(work: Path) -> Path:
    """The extract with a row number and a constant boarding date."""
    frame = pd.read_csv(ROOT / "tests/fixtures/titanic_extract.csv")
    frame.insert(0, "passenger_id", range(len(frame)))
    frame["boarding_date"] = "1912-04-10"
    path = work / "data.csv"
    frame.to_csv(path, index=False)
    (work / "request.txt").write_text(REQUEST + "\n", encoding="utf-8")
    (work / "cost.json").write_text(json.dumps(COSTS), encoding="utf-8")
    (work / "source.json").write_text(
        json.dumps(
            {
                "uri": "file://titanic_extract.csv",
                "columns": list(frame.columns),
                "primary_key": ["passenger_id"],
            }
        ),
        encoding="utf-8",
    )
    return path


def run_pipeline(work: Path, data: Path) -> dict:
    """Frame, sign, run, package, register, promote."""
    framing = work / "framing"
    scmle(
        "frame",
        str(work / "request.txt"),
        "--source",
        str(work / "source.json"),
        "--data",
        str(data),
        "--parse-dates",
        "boarding_date",
        "--cost-model",
        str(work / "cost.json"),
        "--output",
        str(framing),
    )
    scmle("charter", "sign", str(framing / "charter.json"))
    scmle(
        "run",
        str(data),
        "--target",
        "survived",
        "--charter",
        str(framing / "charter.json"),
        "--id-columns",
        "passenger_id",
        "--exclude",
        "boarding_date",
        "--unavailable-at-prediction",
        "boat",
        "body",
        "--time-budget",
        "40",
        "--workers",
        "2",
        "--top-models",
        "2",
        "--configs-per-model",
        "2",
        "--seed",
        "7",
        "--output-dir",
        str(work / "runs"),
    )
    run_dir = sorted((work / "runs").glob("run_*"))[-1]
    scmle("package", str(run_dir / "model"), "--output", str(work / "service"))
    return {"framing": framing, "run": run_dir}


def register(
    work: Path, run_dir: Path
) -> tuple[registry.ModelRegistryStore, int]:
    """Registers the bundle and promotes it on a signed approval."""
    store = registry.ModelRegistryStore(work / "registry")
    evaluation = json.loads((run_dir / "evaluation.json").read_text())
    record = store.register(
        MODEL,
        run_dir / "model",
        metrics=evaluation["test_metrics"],
        config={"seed": 7, "charter": "framing/charter.json"},
        run_id=evaluation["run_id"],
        signing_key=KEYS["analyst"],
    )
    approval = registry.PromotionApproval(
        model=MODEL,
        version=record.version,
        content_id=record.content_id,
        reviewer="reviewer-role (the author, with the reviewer key)",
        decision=registry.PromotionDecision.APPROVED,
        reason="test ROC AUC and the held-out cost per case met the "
        "stakeholder's acceptance criteria",
        evidence=("run/evaluation.json", "reports/experiment_comparison.json"),
        approved_at=events.utc_timestamp(),
    ).sign(KEYS["reviewer"])
    releases = release_state.ReleaseRegistry(store)
    releases.approve(MODEL, record.version, approval, KEYS["reviewer"])
    store.promote(
        MODEL,
        record.version,
        approval=approval,
        signing_key=KEYS["reviewer"],
    )
    return store, record.version


def replay(
    work: Path,
    run_dir: Path,
    data: Path,
    store: registry.ModelRegistryStore,
    version: int,
) -> dict:
    """Feeds the held-out test rows through the loop, batch by batch."""
    record = store.resolve(MODEL, version)
    bundle = store.load(MODEL, version, KEYS["analyst"])
    frame = pd.read_csv(data)
    split = json.loads((run_dir / "split_manifest.json").read_text())
    rows = frame.iloc[sorted(split["final_test_ids"])].reset_index(drop=True)
    features = rows.drop(columns=["survived", "boat", "body"])
    probability = bundle.predict_proba(features)[:, 1]
    costs = json.loads((work / "framing/decision_cost_model.json").read_text())
    cut = outcomes.optimal_threshold(costs)
    label = rows["survived"].astype(int).tolist()
    spent = outcomes.case_costs(costs, cut, probability.tolist(), label)
    scale = max(spent) or 1.0
    reference = sum(spent) / len(spent) / scale
    policy = {
        "reference_risk": round(reference, 4),
        "tolerance": 0.1,
        "scope": f"{record.content_id}:titanic-replay",
        "initial_tau": 0.5,
        "min_matured": 5,
        "r_star": 0.6,
        "outcome_window": 0,
        "alarm_gates_retrain": False,
        "challenger_budget": 100,
        "reply_timeout": 10000,
        "delta_promo": 0.0,
        "loss_bound": 1.0,
    }
    deploy = work / "deployment"
    deploy.mkdir()
    (work / "policy.json").write_text(json.dumps(policy), encoding="utf-8")
    charter_hash = json.loads((run_dir / "analysis_charter.json").read_text())[
        "charter_hash"
    ]
    scmle(
        "loop",
        "init",
        str(deploy),
        "--policy",
        str(work / "policy.json"),
        "--charter-hash",
        charter_hash,
    )
    batches = []
    steps = []
    for number, start in enumerate(range(0, len(rows), BATCH)):
        part = slice(start, start + BATCH)
        ids = rows["passenger_id"].iloc[part].astype(int).tolist()
        p_part = [round(float(v), 6) for v in probability[part]]
        decision = [int(v >= cut) for v in p_part]
        pd.DataFrame(
            {"case_id": ids, "prediction": p_part, "decision": decision}
        ).to_csv(work / "batch.csv", index=False)
        scmle(
            "loop", "observe", str(deploy), "--batch", str(work / "batch.csv")
        )
        part_costs = outcomes.case_costs(costs, cut, p_part, label[part])
        pd.DataFrame(
            {"case_id": ids, "loss": [c / scale for c in part_costs]}
        ).to_csv(work / "losses.csv", index=False)
        scmle(
            "loop", "mature", str(deploy), "--labels", str(work / "losses.csv")
        )
        steps.append(json.loads(scmle("loop", "step", str(deploy))))
        batches.append(
            {
                "batch": number,
                "case_ids": ids,
                "probability": p_part,
                "label": label[part],
            }
        )
    feed = {
        "source": "replay of the held-out test rows of the run, in "
        "passenger_id order, scored by the registered champion; not "
        "production traffic",
        "threshold": cut,
        "stream": "titanic-replay",
        "batches": batches,
    }
    (deploy / "observed_outcomes.json").write_text(
        json.dumps(feed, indent=1) + "\n", encoding="utf-8"
    )
    scmle("loop", "verify", str(deploy))
    mean, low, high = outcomes.interval(spent)
    return {
        "deploy": deploy,
        "steps": steps,
        "interval": [low, high],
        "mean": mean,
    }


def write_inputs(
    work: Path, run_dir: Path, replayed: dict, version: int, charter_hash: str
) -> Path:
    """The stakeholder's and reviewer's statements, signed."""
    low, high = replayed["interval"]
    ceiling = round(high + 0.005, 2)
    reviewed = {
        name: records.sha256_file(run_dir / name.removeprefix("run/"))
        for name in (
            "run/evaluation.json",
            "run/feasibility.json",
            "run/curation_report.json",
            "run/confirmed_claims.json",
            "run/dataset_manifest.json",
        )
    }
    reviewed["framing/decision_cost_model.json"] = records.sha256_file(
        work / "framing/decision_cost_model.json"
    )
    record = {
        "model": MODEL,
        "version": version,
        "charter_hash": charter_hash,
        "builder": "sachin",
        "stakeholder": {
            "name": "stakeholder-role (the author, with the stakeholder key)",
            "kpis": [
                {
                    "kpi": "Expected cost of mis-prioritised briefings per "
                    "passenger (missing a survivor costs 5, briefing a "
                    "non-survivor costs 1)",
                    "owner": "purser",
                    "metric": "outcomes#mean_cost",
                    "direction": "lower_is_better",
                },
                {
                    "kpi": "Survivors ranked ahead of non-survivors",
                    "owner": "purser",
                    "metric": "run/evaluation.json#test_metrics.roc_auc",
                    "direction": "higher_is_better",
                },
            ],
            "acceptance": [
                {
                    "id": "A1",
                    "path": "run/evaluation.json#test_metrics.roc_auc",
                    "op": ">=",
                    "value": 0.8,
                    "reason": "ranking must beat a coin by a wide margin",
                },
                {
                    "id": "A2",
                    "path": "outcomes#cases",
                    "op": ">=",
                    "value": 200,
                    "reason": "enough matured outcomes to judge",
                },
                {
                    "id": "A3",
                    "path": "outcomes#mean_cost",
                    "op": "<=",
                    "value": ceiling,
                    "reason": "set after seeing the held-out evaluation, so "
                    "it is a post-hoc ceiling, not a prospective one",
                },
            ],
            "expectation": {
                "kpi": "cost per case",
                "metric": "outcomes#mean_cost",
                "low": round(low - 0.005, 2),
                "high": ceiling,
                "level": 0.95,
                "basis": "95% normal interval of the cost per case of the "
                "rule on the 217 held-out test rows, widened by 0.005 and "
                "rounded; fixed before the replay was scored through the loop",
            },
        },
        "reviewer": {
            "name": "reviewer-role (the author, with the reviewer key)",
            "thresholds": [
                {
                    "id": "Q1",
                    "path": "run/dataset_manifest.json#row_count",
                    "op": ">=",
                    "value": 1000,
                    "reason": "fewer rows than this make the interval too wide",
                },
                {
                    "id": "Q2",
                    "path": "run/evaluation.json#fit_health.verdict",
                    "op": "==",
                    "value": "healthy",
                    "reason": "training gain must carry over to validation",
                },
                {
                    "id": "Q3",
                    "path": "run/evaluation.json#cv_confidence.interval.0",
                    "op": ">=",
                    "value": 0.75,
                    "reason": "lower end of the cross-validated ROC AUC",
                },
                {
                    "id": "Q4",
                    "path": "run/evaluation.json#n_test",
                    "op": ">=",
                    "value": 200,
                    "reason": "rows in the final test",
                },
            ],
            "evidence": {
                "statement": "Read the listed records. The confirmation set "
                "(273 rows) was smaller than the 333 the chartered power "
                "needs, so the charter run's scope is exploratory_unconfirmed "
                "and its claims are tiered accordingly. The extract has no "
                "timestamp or key, so the row number and boarding date are "
                "assumptions, and the row number correlates with the target "
                "because the extract is ordered. Played by the author.",
                "reviewed": reviewed,
            },
        },
    }
    signed = trail_inputs.sign_inputs(record, KEYS)
    path = work / "inputs.json"
    records.write_json(path, signed)
    return path


def main(arguments: list[str]) -> int:
    """Runs the pipeline and writes the evidence folder."""
    output = Path(arguments[0] if arguments else "plan/evidence/phase-18/local")
    output = output.resolve()
    with tempfile.TemporaryDirectory() as scratch:
        work = Path(os.environ.get("PHASE18_WORK", scratch))
        work.mkdir(parents=True, exist_ok=True)
        data = prepare(work)
        made = run_pipeline(work, data)
        store, version = register(work, made["run"])
        replayed = replay(work, made["run"], data, store, version)
        charter_hash = json.loads(
            (made["run"] / "analysis_charter.json").read_text()
        )["charter_hash"]
        inputs_path = write_inputs(
            work, made["run"], replayed, version, charter_hash
        )
        keys_path = work / "keys.json"
        keys_path.write_text(json.dumps(KEYS), encoding="utf-8")
        trail = output / "trail"
        if trail.exists():
            shutil.rmtree(trail)
        output.mkdir(parents=True, exist_ok=True)
        scmle(
            "trail",
            "build",
            str(trail),
            "--run",
            str(made["run"]),
            "--framing",
            str(work / "framing"),
            "--registry",
            str(work / "registry"),
            "--deployment",
            str(replayed["deploy"]),
            "--inputs",
            str(inputs_path),
            "--keys",
            str(keys_path),
        )
        verified = scmle(
            "trail", "verify", str(trail), "--keys", str(keys_path)
        )
        shown = scmle("trail", "show", str(trail))
        (output / "keys.json").write_text(
            json.dumps(KEYS, indent=2) + "\n", encoding="utf-8"
        )
        (output / "trail-show.txt").write_text(shown, encoding="utf-8")
        (output / "commands.json").write_text(
            json.dumps(COMMANDS, indent=1) + "\n", encoding="utf-8"
        )
        summary = {
            "ran": RAN,
            "not_run": NOT_RUN,
            "independent_reviewer": False,
            "production_traffic": False,
            "cloud": False,
            "model": MODEL,
            "version": version,
            "charter_hash": charter_hash,
            "verify_output": verified.strip().splitlines()[-1],
            "loop_steps": replayed["steps"],
            "replay_mean_cost": replayed["mean"],
            "replay_interval": replayed["interval"],
        }
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
