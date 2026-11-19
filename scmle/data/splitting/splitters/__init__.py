"""Leakage-aware splitting: strategies, immutable manifest, sealed test set.

Row identifiers are positional indices into the ingested frame. Each
strategy in `splits` (`splits.Random`, `splits.Group` ...) and `temporal`
(`temporal.Chronology`, `temporal.Rolling`) owns one validation protocol;
`Splitter` selects and applies it. The test partition lives behind a
`lockbox.Lockbox`, which opens exactly once, with the token the
orchestrator issues after selection.

The multi-word names the strategies had before they moved into `splits` and
`lockbox` stay importable from here for one release; see `docs/RENAMES.md`.
"""

from scmle.data.splitting.splitters import lockbox
from scmle.data.splitting.splitters import splits
from scmle.data.splitting.splitters.lockbox import Lockbox
from scmle.data.splitting.splitters.partitions import DUPLICATE_AWARE
from scmle.data.splitting.splitters.partitions import Fold
from scmle.data.splitting.splitters.partitions import Manifest
from scmle.data.splitting.splitters.partitions import MIN_FOLDS
from scmle.data.splitting.splitters.partitions import MIN_ROLLING_FOLDS
from scmle.data.splitting.splitters.partitions import MIN_ROLLING_TRAIN
from scmle.data.splitting.splitters.partitions import MIN_STRATUM_ROWS
from scmle.data.splitting.splitters.partitions import Partition
from scmle.data.splitting.splitters.partitions import Partitioner
from scmle.data.splitting.splitters.partitions import Split
from scmle.data.splitting.splitters.partitions import tuplify
from scmle.data.splitting.splitters.splits import Multilabel
from scmle.data.splitting.splitters.splits import Preset
from scmle.data.splitting.splitters.splitter import scheme
from scmle.data.splitting.splitters.splitter import Splitter
from scmle.data.splitting.splitters.splitter import STRATEGIES
from scmle.data.splitting.splitters.splitter import TASK_STRATEGIES
from scmle.data.splitting.splitters.temporal import Chronology

# Names before the family moved into `splits` and `lockbox`; kept for one
# release so existing imports keep working (docs/RENAMES.md).

__all__ = [
    "DUPLICATE_AWARE",
    "MIN_FOLDS",
    "MIN_ROLLING_FOLDS",
    "MIN_ROLLING_TRAIN",
    "MIN_STRATUM_ROWS",
    "STRATEGIES",
    "TASK_STRATEGIES",
    "Chronology",
    "Fold",
    "Lockbox",
    "Manifest",
    "Multilabel",
    "Partition",
    "Partitioner",
    "Preset",
    "Split",
    "Splitter",
    "lockbox",
    "scheme",
    "splits",
    "tuplify",
]
