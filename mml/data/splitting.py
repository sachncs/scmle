"""Leakage-aware splitting: immutable split manifest + sealed test partition.

Row identifiers are positional indices into the ingested frame. The test partition is held
behind LockedTestSet, which can be opened exactly once, only with the token the orchestrator
issues after model selection.
"""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
    train_test_split,
)

from mml.contracts.errors import BlockedError, LeakageError
from mml.contracts.tasks import ProjectSpec, TaskType


@dataclass(frozen=True)
class SplitManifest:
    strategy: str
    seed: int
    train_ids: tuple[int, ...]
    test_ids: tuple[int, ...]
    # folds hold positions *within the development (train) partition*
    folds: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]
    notes: tuple[str, ...] = ()

    @property
    def fingerprint(self) -> str:
        payload = repr((self.strategy, self.seed, self.train_ids, self.test_ids, self.folds))
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "seed": self.seed,
            "fingerprint": self.fingerprint,
            "n_train": len(self.train_ids),
            "n_test": len(self.test_ids),
            "train_ids": list(self.train_ids),
            "test_ids": list(self.test_ids),
            "folds": [{"train": list(t), "val": list(v)} for t, v in self.folds],
            "notes": list(self.notes),
        }


class LockedTestSet:
    """Holds the test partition; unavailable to search, openable once after selection."""

    def __init__(self, X: pd.DataFrame, y: pd.Series, ids: tuple[int, ...]) -> None:
        self._X, self._y = X, y
        self.ids = ids
        self._token = secrets.token_hex(8)
        self._opened = False

    def issue_token(self) -> str:
        """Called by the orchestrator when selection is frozen."""
        return self._token

    def open(self, token: str) -> tuple[pd.DataFrame, pd.Series]:
        if token != self._token:
            raise LeakageError("test set is locked: invalid token")
        if self._opened:
            raise LeakageError("test set was already used for final evaluation")
        self._opened = True
        return self._X, self._y

    def __len__(self) -> int:
        return len(self.ids)

    def __repr__(self) -> str:
        return f"LockedTestSet(n={len(self.ids)}, opened={self._opened})"


def resolve_strategy(spec: ProjectSpec, task: TaskType) -> str:
    s = spec.validation.strategy
    if s != "auto":
        if s == "stratified" and not task.is_classification:
            raise BlockedError("stratified validation requires a classification task")
        return s
    if spec.data.time_column:
        return "temporal"
    if spec.data.group_column:
        return "group"
    return "stratified" if task.is_classification else "random"


def _ids(a: np.ndarray) -> tuple[int, ...]:
    return tuple(int(i) for i in a)


def make_split(
    df: pd.DataFrame, spec: ProjectSpec, task: TaskType
) -> tuple[SplitManifest, pd.DataFrame, pd.Series, LockedTestSet]:
    """Return (manifest, dev features, dev target, locked test)."""
    d, v = spec.data, spec.validation
    strategy = resolve_strategy(spec, task)
    n = len(df)
    y_all = df[d.target]
    idx = np.arange(n)
    notes: list[str] = []
    folds_n = v.folds

    if strategy == "temporal":
        t = pd.to_datetime(df[d.time_column], errors="coerce")
        if t.isna().any():
            raise BlockedError(
                f"time column '{d.time_column}' has unparseable/missing values: ordering ambiguous"
            )
        order = np.argsort(t.to_numpy(), kind="stable")
        n_test = max(1, int(round(n * v.test_fraction)))
        dev_ids, test_ids = order[: n - n_test], order[n - n_test :]
        if len(dev_ids) < folds_n + 1:
            raise BlockedError("too few rows for temporal cross-validation")
        tscv = TimeSeriesSplit(n_splits=folds_n)
        folds = [(tr, va) for tr, va in tscv.split(np.arange(len(dev_ids)))]
    elif strategy == "group":
        groups = df[d.group_column].to_numpy()
        if pd.isna(groups).any():
            raise BlockedError(f"group column '{d.group_column}' has missing values")
        if len(set(groups)) < folds_n + 2:
            raise BlockedError(f"need at least {folds_n + 2} groups for group validation")
        gss = GroupShuffleSplit(n_splits=1, test_size=v.test_fraction, random_state=v.seed)
        dev_ids, test_ids = next(gss.split(idx, groups=groups))
        dev_groups = groups[dev_ids]
        folds_n = min(folds_n, len(set(dev_groups)))
        folds = list(GroupKFold(n_splits=folds_n).split(np.arange(len(dev_ids)), groups=dev_groups))
    else:
        stratify = y_all if strategy == "stratified" else None
        if strategy == "stratified":
            if (y_all.value_counts() < 2).any():
                raise BlockedError("a class has fewer than 2 examples: cannot stratify")
        dev_ids, test_ids = train_test_split(
            idx, test_size=v.test_fraction, random_state=v.seed, stratify=stratify
        )
        dev_ids, test_ids = np.sort(dev_ids), np.sort(test_ids)
        if strategy == "stratified":
            min_class = int(y_all.iloc[dev_ids].value_counts().min())
            if min_class < 2:
                raise BlockedError("insufficient minority-class examples after test split")
            if min_class < folds_n:
                notes.append(f"folds reduced {folds_n}->{min_class} (minority class size)")
                folds_n = min_class
            skf = StratifiedKFold(n_splits=folds_n, shuffle=True, random_state=v.seed)
            folds = list(skf.split(np.arange(len(dev_ids)), y_all.iloc[dev_ids]))
        else:
            kf = KFold(n_splits=folds_n, shuffle=True, random_state=v.seed)
            folds = list(kf.split(np.arange(len(dev_ids))))

    manifest = SplitManifest(
        strategy=strategy,
        seed=v.seed,
        train_ids=_ids(dev_ids),
        test_ids=_ids(test_ids),
        folds=tuple((_ids(a), _ids(b)) for a, b in folds),
        notes=tuple(notes),
    )
    drop = [d.target]
    X = df.drop(columns=drop)
    dev_X = X.iloc[dev_ids].reset_index(drop=True)
    dev_y = y_all.iloc[dev_ids].reset_index(drop=True)
    locked = LockedTestSet(
        X.iloc[test_ids].reset_index(drop=True),
        y_all.iloc[test_ids].reset_index(drop=True),
        manifest.test_ids,
    )
    return manifest, dev_X, dev_y, locked
