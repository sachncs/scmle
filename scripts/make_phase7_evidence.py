"""Generates the Phase 7 evidence artifacts by running the real code.

    uv run python scripts/make_phase7_evidence.py plan/evidence/phase-7

Writes, under the output folder:

- `signed-sample-run/`: the artifacts of a charter run on the scikit-learn
  breast-cancer table, with a charter signed by a published demonstration
  key. The key is public, so the signature shows tamper evidence only; it is
  not a statement by an independent holder.
- `access-attempt-log/`: a custody log (with MACs) from a scripted attack on a
  real vault process, and what each attempt returned. The attacker is this
  script, not an independent red team.
- `split-manifests/`: the run's manifest and the manifests of the grouped and
  temporal fixtures that `tests/unit/test_custody_golden.py` pins.

Nothing here is edited by hand: re-running it replaces the folder.
"""

from collections.abc import Callable
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import pandas as pd
from sklearn import datasets

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scmle  # noqa: E402
from scmle.contracts import errors  # noqa: E402
from scmle.contracts import tasks  # noqa: E402
from scmle.data.charter import sealing  # noqa: E402
from scmle.data.custody import manifest as manifest_module  # noqa: E402
from scmle.data.custody import sealed  # noqa: E402
from scmle.data.ledger import ledger as ledger_module  # noqa: E402
from scmle.data.splitting import splitters  # noqa: E402
from scmle.governance import vault  # noqa: E402
from tests.unit import test_custody_golden as golden  # noqa: E402
from tests.unit.test_charter_contracts import make_charter  # noqa: E402

DEMONSTRATION_KEY = "scmle-phase7-demonstration-key-not-secret"
KEPT = (
    "analysis_charter.json",
    "split_manifest.json",
    "error_budget.json",
    "feasibility.json",
    "custody_log.jsonl",
    "hypothesis_ledger.jsonl",
    "negative_control.json",
    "confirmed_claims.json",
)


def verify_run(folder: Path) -> dict[str, str]:
    """Checks the kept artifacts with the library's own verifiers."""
    charter = sealing.read_verified(
        folder / "analysis_charter.json", signing_key=DEMONSTRATION_KEY
    )
    claims = json.loads((folder / "confirmed_claims.json").read_text())
    ledger = ledger_module.Ledger.load(
        folder / "hypothesis_ledger.jsonl", expected_head=claims["ledger_head"]
    )
    log = sealed.OpenLog.verify_file(
        folder / "custody_log.jsonl", signing_key=DEMONSTRATION_KEY
    )
    manifest = manifest_module.CustodyManifest.from_dict(
        json.loads((folder / "split_manifest.json").read_text())
    )
    return {
        "charter_signature": "valid" if charter.signed else "unsigned",
        "ledger_chain_and_head": f"valid, {len(ledger.hypotheses)} hypotheses",
        "custody_log_chain_and_macs": f"valid, {len(log.entries)} entries",
        "split_manifest_hash": f"valid, sha256 {manifest.sha256}",
        "outcome": str(claims["outcome"]),
    }


def sample_run(out: Path) -> Path:
    """Runs a charter run and keeps its small, verifiable artifacts."""
    raw = datasets.load_breast_cancer(as_frame=True)
    frame = raw.frame.rename(columns={"target": "y"})
    with tempfile.TemporaryDirectory() as scratch:
        charter_path = Path(scratch) / "charter.json"
        sealing.write_charter(
            charter_path,
            make_charter(min_effect=0.5, k_max=3),
            signing_key=DEMONSTRATION_KEY,
        )
        os.environ["SCMLE_SIGNING_KEY"] = DEMONSTRATION_KEY
        result = scmle.MLProject(
            frame,
            "y",
            time_budget=300,
            output_dir=Path(scratch) / "out",
            charter=charter_path,
            enforcement="cooperative",
            search=tasks.SearchSpec(top_models=0),
            strategies=tasks.StrategySpec(top_models=0),
            ensemble=tasks.EnsembleSpec(size=0),
            models=tasks.ModelSpec(
                exclude_families=["bagging", "deep_learning", "foundation"]
            ),
        ).run()
        run_dir = Path(result.run_dir)
        target = out / "signed-sample-run"
        shutil.rmtree(target, ignore_errors=True)
        target.mkdir(parents=True)
        for name in KEPT:
            if (run_dir / name).exists():
                shutil.copy(run_dir / name, target / name)
    steps = verify_run(target)
    (target / "verification.json").write_text(
        json.dumps(steps, indent=2, sort_keys=True) + "\n"
    )
    return target


def attack(out: Path) -> Path:
    """Scripted attempts on a real vault; returns the folder."""
    target = out / "access-attempt-log"
    shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True)
    log_path = target / "custody_log.jsonl"
    log = sealed.OpenLog(log_path, signing_key=DEMONSTRATION_KEY)
    held = sealed.SealedSet(
        "C",
        vault.SealedTestSet(
            pd.DataFrame({"a": [1.0, 2.0, 3.0]}),
            pd.Series([0, 1, 0]),
            (10, 11, 12),
        ),
        log,
    )
    held.commit("claims-sha-A", "ledger-head-A")
    token = held.issue_token()
    attempts: list[dict[str, str]] = []

    def attempt(label: str, call: Callable[[], object]) -> None:
        try:
            call()
            outcome = "released"
        except errors.LeakageError as refusal:
            outcome = f"refused: {refusal}"
        attempts.append({"attempt": label, "outcome": outcome})

    attempt("guessed token", lambda: held.open("guess", "attacker"))
    attempt("right token, no binding", lambda: held.open(token, "attacker"))
    attempt(
        "right token, other ledger state",
        lambda: held.open(token, "attacker", "claims-sha-A", "ledger-head-B"),
    )
    attempt(
        "right token and binding",
        lambda: held.open(
            token, "confirmation", "claims-sha-A", "ledger-head-A"
        ),
    )
    attempt(
        "second open, same token and binding",
        lambda: held.open(token, "attacker", "claims-sha-A", "ledger-head-A"),
    )
    held.locked.close()
    sealed.OpenLog.verify_file(log_path, signing_key=DEMONSTRATION_KEY)
    (target / "attempts.json").write_text(
        json.dumps(attempts, indent=2) + "\n", encoding="utf-8"
    )
    return target


def manifests(out: Path, run_folder: Path) -> Path:
    """Writes the run's manifest and the two fixtures' manifests."""
    target = out / "split-manifests"
    shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True)
    shutil.copy(run_folder / "split_manifest.json", target / "run.json")
    for name, build in (
        ("grouped-fixture.json", golden.grouped),
        ("temporal-fixture.json", golden.temporal),
    ):
        (target / name).write_text(
            json.dumps(build().to_dict(), indent=2, sort_keys=True) + "\n"
        )
    return target


def main() -> int:
    """Builds every artifact under the folder named on the command line."""
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    run_folder = sample_run(out)
    attack(out)
    manifests(out, run_folder)
    assert splitters  # imported so the fixtures register their strategies
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
