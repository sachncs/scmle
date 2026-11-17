"""Leakage-aware splitting: strategies, immutable manifest, sealed test set.

Row identifiers are positional indices into the ingested frame. Each
`Partitioner` subclass owns one validation protocol; `Splitter` selects
and applies it. The test partition lives behind `Lockbox`, which opens
exactly once, with the token the orchestrator issues after selection.
"""

from scmle.data.splitting.splitters.chronology import Chronology
from scmle.data.splitting.splitters.chronology import RollingSplit
from scmle.data.splitting.splitters.grouping import GroupSplit
from scmle.data.splitting.splitters.partitions import DUPLICATE_AWARE
from scmle.data.splitting.splitters.partitions import Fold
from scmle.data.splitting.splitters.partitions import Lockbox
from scmle.data.splitting.splitters.partitions import Manifest
from scmle.data.splitting.splitters.partitions import MIN_FOLDS
from scmle.data.splitting.splitters.partitions import MIN_ROLLING_FOLDS
from scmle.data.splitting.splitters.partitions import MIN_ROLLING_TRAIN
from scmle.data.splitting.splitters.partitions import MIN_STRATUM_ROWS
from scmle.data.splitting.splitters.partitions import Partition
from scmle.data.splitting.splitters.partitions import Partitioner
from scmle.data.splitting.splitters.partitions import ResidentTestSet
from scmle.data.splitting.splitters.partitions import Split
from scmle.data.splitting.splitters.partitions import tuplify
from scmle.data.splitting.splitters.splitter import scheme
from scmle.data.splitting.splitters.splitter import Splitter
from scmle.data.splitting.splitters.splitter import STRATEGIES
from scmle.data.splitting.splitters.splitter import TASK_STRATEGIES
from scmle.data.splitting.splitters.stratification import EventStratifiedSplit
from scmle.data.splitting.splitters.stratification import Multilabel
from scmle.data.splitting.splitters.stratification import Preset
from scmle.data.splitting.splitters.stratification import RandomSplit
from scmle.data.splitting.splitters.stratification import StratifiedSplit
from scmle.data.splitting.splitters.stratification import (
    TreatmentStratifiedSplit,
)

__all__ = [
    "DUPLICATE_AWARE",
    "MIN_FOLDS",
    "MIN_ROLLING_FOLDS",
    "MIN_ROLLING_TRAIN",
    "MIN_STRATUM_ROWS",
    "STRATEGIES",
    "TASK_STRATEGIES",
    "Chronology",
    "EventStratifiedSplit",
    "Fold",
    "GroupSplit",
    "Lockbox",
    "Manifest",
    "Multilabel",
    "Partition",
    "Partitioner",
    "Preset",
    "RandomSplit",
    "ResidentTestSet",
    "RollingSplit",
    "Split",
    "Splitter",
    "StratifiedSplit",
    "TreatmentStratifiedSplit",
    "scheme",
    "tuplify",
]
