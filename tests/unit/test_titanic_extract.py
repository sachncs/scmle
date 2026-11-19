"""The frozen Titanic extract: the screen flags `boat` and `body`."""

import hashlib
import json
from pathlib import Path

import pandas
import pytest

from benchmarks.qualification import corpora
from scmle.assurance.admissibility import leakage
from scmle.assurance.admissibility import screen

FIXTURE = Path(__file__).parents[1] / "fixtures" / "titanic_extract.csv"
ENTRY = "titanic-boat-body"


@pytest.fixture(scope="module")
def frame():
    return pandas.read_csv(FIXTURE)


@pytest.fixture(scope="module")
def report(frame):
    return screen.screen_table(
        frame,
        "survived",
        "classification",
        [c for c in frame.columns if c != "survived"],
        leakage.CERTIFIED.run_seed,
    )


def test_the_extract_matches_the_hash_in_the_manifest():
    manifest = json.loads(corpora.MANIFEST.read_text(encoding="utf-8"))
    entry = next(c for c in manifest["corpora"] if c["id"] == ENTRY)
    data = FIXTURE.read_bytes()
    assert hashlib.sha256(data).hexdigest() == entry["sha256"]
    assert len(data) == entry["size_bytes"]


def test_the_extract_has_the_recording_structure(frame):
    # Enforces README.md:199.
    # Enforces website/src/pages/docs/benchmarks.astro:773.
    assert len(frame) == 1309
    assert frame["boat"].isna().mean() == pytest.approx(0.629, abs=0.001)
    assert frame["body"].isna().mean() == pytest.approx(0.908, abs=0.001)
    survivors = frame["survived"] == 1
    assert frame.loc[~survivors, "boat"].isna().mean() > 0.9
    assert frame.loc[survivors, "body"].isna().all()


def test_the_certified_screen_blocks_boat_and_body(report):
    # Enforces README.md:199.
    # Enforces website/src/pages/docs/benchmarks.astro:773.
    assert {f.feature for f in report.blocking} >= {"boat", "body"}
    mechanisms = {f.mechanism for f in report.blocking if f.feature == "body"}
    assert mechanisms == {"record_purity"}


def test_ordinary_columns_are_not_blocked(report):
    blocked = {f.feature for f in report.blocking}
    assert blocked.isdisjoint(
        {"pclass", "sex", "age", "sibsp", "parch", "fare"}
    )
