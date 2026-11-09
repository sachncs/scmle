"""Trivial stand-in detectors that exercise the harness end to end.

There is no real detector in this module. These three do just enough to run
the whole chain (plant, detect, certify, store, go stale) on a real table
that ships with scikit-learn. They are NOT part of the product and must
never be marked gating; their certificates live in
`sample_certificates.json` only as evidence that the machinery works.

The policy below is fixed before any measurement and is not tuned to the
results.
"""

from collections.abc import Callable
from collections.abc import Sequence
import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from benchmarks.qualification import certify
from benchmarks.qualification import inject
from scmle.data.detectors import certificates

TARGET = "target"
SAMPLE_PATH = Path(__file__).with_name("sample_certificates.json")
SENTINEL_NUMBERS = (-999.0, -9999.0)
SENTINEL_TEXT = ("N/A", "NA", "null", "?")
CONFIDENT_DISAGREEMENT = 0.2
FOLDS = 5
SENTINEL_COLUMNS = ["mean radius", "mean texture"]
FLIP_RATE = 0.10
SENTINEL_RATE = 0.10
DUPLICATE_RATE = 0.10
# Fixed in advance: 50 percent recall floor, 10 percent false-alarm ceiling,
# 95 percent Wilson intervals.
POLICY = certificates.CertificationPolicy(
    recall_floor=0.5, false_alarm_ceiling=0.10, confidence=0.95
)
DEFAULT_SEEDS = (0, 1, 2)


def demo_frame() -> pd.DataFrame:
    """The breast-cancer table bundled with scikit-learn (no download).

    Returns:
        The features and the `target` column, with a fresh 0..n-1 index.
    """
    return load_breast_cancer(as_frame=True).frame.reset_index(drop=True)


def label_flip_standin(frame: pd.DataFrame) -> pd.Series:
    """Flags rows whose label a cross-validated model confidently rejects.

    Args:
        frame: Table with feature columns and the `target` label column.

    Returns:
        A boolean Series with the index of `frame`; True where the
        cross-validated probability of the observed label is below
        `CONFIDENT_DISAGREEMENT`.
    """
    features = frame.drop(columns=[TARGET])
    labels = frame[TARGET]
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    folds = StratifiedKFold(FOLDS, shuffle=True, random_state=0)
    probability = cross_val_predict(
        model, features, labels, cv=folds, method="predict_proba"
    )
    classes = np.sort(labels.unique())
    observed = np.searchsorted(classes, labels.to_numpy())
    own = probability[np.arange(len(frame)), observed]
    return pd.Series(own < CONFIDENT_DISAGREEMENT, index=frame.index)


def sentinel_standin(frame: pd.DataFrame) -> pd.Series:
    """Flags rows that hold a common missing-value code.

    Args:
        frame: Table to scan; numeric columns are checked against
            `SENTINEL_NUMBERS` and the others against `SENTINEL_TEXT`.

    Returns:
        A boolean Series with the index of `frame`; True for a row with a
        sentinel in any column.
    """
    flags = pd.Series(False, index=frame.index)
    for name in frame.columns:
        column = frame[name]
        if pd.api.types.is_numeric_dtype(column):
            flags |= column.isin(SENTINEL_NUMBERS)
        else:
            flags |= column.isin(SENTINEL_TEXT)
    return flags


def duplicate_standin(frame: pd.DataFrame) -> pd.Series:
    """Flags every copy after the first occurrence of a row.

    Args:
        frame: Table to scan.

    Returns:
        A boolean Series with the index of `frame`; True for each repeated
        row after its first occurrence.
    """
    return pd.Series(frame.duplicated(keep="first"), index=frame.index)


@dataclasses.dataclass(frozen=True)
class StandIn:
    """A stand-in detector with the defect it is measured on.

    Attributes:
        detector: Maps a frame to one boolean flag per row.
        plan: How to plant the defect it should find.
        helpers: Extra callables whose source is part of the code hash.
    """

    detector: Callable[[pd.DataFrame], pd.Series]
    plan: certify.DefectPlan
    helpers: tuple[Any, ...] = ()

    def code_hash(self) -> str:
        """Hash of the detector and its helpers' source.

        Returns:
            The code hash that certificates are filed under.
        """
        return certificates.code_hash(self.detector, *self.helpers)


STANDINS: dict[str, StandIn] = {
    "standin-label-flip": StandIn(
        label_flip_standin,
        certify.DefectPlan(
            "label_flip",
            lambda frame, seed: inject.flip_labels(
                frame, TARGET, FLIP_RATE, seed
            ),
        ),
    ),
    "standin-sentinel": StandIn(
        sentinel_standin,
        certify.DefectPlan(
            "sentinel",
            lambda frame, seed: inject.inject_sentinels(
                frame, SENTINEL_COLUMNS, SENTINEL_RATE, seed
            ),
        ),
    ),
    "standin-duplicate": StandIn(
        duplicate_standin,
        certify.DefectPlan(
            "duplicate_exact",
            lambda frame, seed: inject.inject_duplicates(
                frame, DUPLICATE_RATE, seed
            ),
        ),
    ),
}


def certify_all(
    seeds: Sequence[int] = DEFAULT_SEEDS,
    frame: pd.DataFrame | None = None,
) -> certificates.CertificateStore:
    """Certifies every stand-in on the demo table.

    Args:
        seeds: Run seeds; counts are pooled over them.
        frame: Table to plant into; defaults to `demo_frame()`.

    Returns:
        A store holding one certificate per stand-in.
    """
    data = demo_frame() if frame is None else frame
    store = certificates.CertificateStore()
    for name, standin in STANDINS.items():
        store.add(
            certify.certify_detector(
                name,
                standin.detector,
                standin.code_hash(),
                data,
                [standin.plan],
                seeds,
                POLICY,
            )
        )
    return store


def sample_problems(store: certificates.CertificateStore) -> list[str]:
    """Why a store of stand-in certificates is not current (empty if fine).

    Args:
        store: Certificate store to audit.

    Returns:
        One message per stand-in whose certificate is not current for its
        present code; empty when all are current.
    """
    problems = []
    for name, standin in STANDINS.items():
        digest = standin.code_hash()
        state = store.state(name, digest)
        if state is not certificates.CertificateState.CURRENT:
            problems.append(f"{name}: certificate is {state.value}")
    return problems
