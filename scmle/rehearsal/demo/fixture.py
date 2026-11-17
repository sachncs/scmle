"""The published demo fixture: its key, data, thresholds and fast training.

The demo signing key below is a published fixture, not a secret.
"""

import pathlib
import tempfile
import time
from typing import Any

import pandas

import scmle.api.project
import scmle.artifacts.results
from scmle.contracts import tasks
import scmle.governance.registry

DEMO_SIGNING_KEY = "scmle-local-demo-signing-key-published-fixture"
MODEL = "churn"
TARGET = "churned"
DATA_DIRECTORY = pathlib.Path("deploy/local/data")
GOLDEN_DIRECTORY = pathlib.Path("deploy/local/golden")
REVIEWER = "demo-reviewer"
READY_TIMEOUT_SECONDS = 120.0
LATENCY_REQUESTS = 120
EXPECTED_SAMPLES = 4  # metric samples the demo ships to the sink
MINIMUM_ROC_AUC = 0.8  # a trained demo model must beat this on the test set
PROBABILITY_TOLERANCE = 1e-9  # online and batch probabilities must agree
PROMOTED_VERSION = 2  # the bad release is registered second and promoted
FAST_OPTIONS: dict[str, Any] = {
    "search": tasks.Search(top_models=0),
    "strategies": tasks.Strategies(top_models=0),
    "ensemble": tasks.Stacking(size=0),
    "models": tasks.Candidates(
        exclude_families=["bagging", "deep_learning", "foundation"]
    ),
    "enforcement": "cooperative",
    "time_budget": 300,
    "seed": 7,
}


def now() -> str:
    """Formats the current time.

    Returns:
        The current UTC time as an ISO-8601 string with a `+00:00` offset and
        second resolution.
    """
    return time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())


def train(name: str, data: pathlib.Path) -> scmle.artifacts.results.Run:
    """Trains one model on a CSV with the demo's fixed, fast options.

    Args:
        name: Prefix of the temporary output directory.
        data: CSV file holding the features and the target column.

    Returns:
        The run result, including the model bundle and its evaluation.
    """
    frame = pandas.read_csv(data)
    return scmle.api.project.Project(
        frame,
        TARGET,
        output_dir=str(tempfile.mkdtemp(prefix=name)),
        **FAST_OPTIONS,
    ).run()


def sanction(
    record: scmle.governance.registry.Release, reason: str, evidence: str
) -> scmle.governance.registry.Sanction:
    """Builds an approval bound to one registered version.

    Args:
        record: The registered version being approved.
        reason: Why the reviewer approves.
        evidence: Name of the evidence the reviewer cites.

    Returns:
        An approved promotion by the demo reviewer, signed with the demo key.
    """
    return scmle.governance.registry.Sanction(
        model=record.model,
        version=record.version,
        content_id=record.content_id,
        reviewer=REVIEWER,
        decision=scmle.governance.registry.Verdict.APPROVED,
        reason=reason,
        evidence=(evidence,),
        approved_at=now(),
    ).sign(DEMO_SIGNING_KEY)
