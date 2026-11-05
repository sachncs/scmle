"""Synthetic traffic, references and signal windows for operations tests."""

import numpy as np
import pandas as pd

from scmle.monitoring import reference
from scmle.operations import signals

START = 1_000_000.0


def frame(rows=400, seed=0, shift=0.0, missing=0.0, unseen=0.0):
    """Features shaped like the golden model's: two numbers and a plan."""
    rng = np.random.default_rng(seed)
    age = rng.normal(40.0 + shift, 10.0, rows)
    balance = rng.lognormal(7.0, 0.5, rows) * (1.0 + shift / 40.0)
    plan = rng.choice(["basic", "pro"], rows)
    if unseen:
        plan = np.where(rng.random(rows) < unseen, "enterprise", plan)
    out = pd.DataFrame({"age": age, "balance": balance, "plan": plan})
    if missing:
        hole = rng.random(rows) < missing
        out.loc[hole, "age"] = np.nan
    return out


def profile(rows=2000, seed=99):
    base = frame(rows, seed)
    score = (base["age"] - 40.0) / 10.0
    return reference.ReferenceProfile(
        features={
            "age": reference.NumericReference.fit(base["age"]),
            "balance": reference.NumericReference.fit(base["balance"]),
            "plan": reference.CategoricalReference.fit(base["plan"]),
        },
        output=reference.NumericReference.fit(score),
        output_name="score",
        reference_metrics={},
    )


def monitor(window=300.0, minimum=30, **thresholds):
    return signals.SignalMonitor(
        signals.WindowPolicy(window_seconds=window, min_samples=minimum),
        signals.SignalThresholds(**thresholds),
        profile(),
    )


def healthy_traffic(watch, count=200, at=START, status=200, seconds=0.05):
    for step in range(count):
        watch.record_request(at + step * 0.1, "/predict", status, seconds)


def named(rows, name):
    return next(row for row in rows if row.name == name)
