"""Pinned split manifests: a platform that differs shows as a new hash.

The fixtures are generated from fixed seeds and a fixed data hash, carved
with the in-process protocol (no vault process), and hashed. The expected
hashes are constants. If a platform, a numpy, pandas or scikit-learn release
changes a row id, a fold or a row-content root, this test fails there. One
machine cannot compare platforms with each other: the constants were pinned
on macOS arm64 with Python 3.14.7, numpy 2.5.4, pandas 3.0.6 and
scikit-learn 1.9.1. Equality on Windows and under the lowest supported
dependencies is established only when the `cross-platform` and
`minimal-dependencies` jobs of `.github/workflows/ci.yml` (which run
`tests/unit`) pass; it has not been observed from this machine.
"""

import numpy
import pandas

from scmle.contracts import tasks
from scmle.data.custody import pipeline
from scmle.data.custody import sealed
from scmle.data.splitting import splitters
from tests.unit.test_charter_contracts import make_charter

DATA_HASH = "7" * 64
GROUPED_SHA256 = (
    "7fc52bbd5aac330519c90b65fe3bd9c7d9ed33c23351d451a825f5bd7b948238"
)
TEMPORAL_SHA256 = (
    "f85653d27f4c12653ec69d8d994166afdf54cd8d622584928abd8434958ef0af"
)


def classification_frame(rows=1200):
    rng = numpy.random.default_rng(2024)
    return pandas.DataFrame(
        {
            "t": numpy.arange(rows),
            "g": numpy.arange(rows) // 4,
            "x": rng.normal(size=rows),
            "z": rng.integers(0, 50, size=rows),
            "y": (rng.random(rows) < 0.3).astype(int),
        }
    )


def manifest_of(spec, task, frame):
    split = splitters.Splitter(spec, task).split(frame)
    vault = sealed.Vault(sealed.OpenLog(), process=False)
    return pipeline.apply_custody(
        charter=make_charter(min_effect=0.3, k_max=5),
        split=split,
        frame=frame,
        spec=spec,
        task=task,
        data_hash=DATA_HASH,
        vault=vault,
    ).manifest


def grouped():
    spec = tasks.Config(
        data=tasks.Dataset(uri="m", target="y", group_column="g"),
        validation=tasks.Validation(strategy="group", seed=5),
    )
    return manifest_of(spec, tasks.Task.BINARY, classification_frame())


def temporal():
    spec = tasks.Config(
        data=tasks.Dataset(uri="m", target="y", time_column="t"),
        validation=tasks.Validation(strategy="temporal", seed=5),
    )
    return manifest_of(spec, tasks.Task.BINARY, classification_frame())


def test_the_grouped_manifest_has_its_pinned_hash():
    manifest = grouped()
    assert manifest.confirmation_ids and manifest.row_roots
    assert manifest.sha256 == GROUPED_SHA256


def test_the_temporal_manifest_has_its_pinned_hash():
    manifest = temporal()
    assert manifest.confirmation_ids and manifest.row_roots
    assert manifest.sha256 == TEMPORAL_SHA256


def test_the_two_fixtures_differ_and_each_is_repeatable():
    assert grouped().sha256 != temporal().sha256
    assert grouped().sha256 == grouped().sha256
