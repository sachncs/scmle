"""Characterization goldens for modality runs, one file per modality.

Each modality's golden lives in `tests/fixtures/modality/<modality>.json`,
so the agents that move one modality onto the shared stages never share a
golden. A golden records what a run of the lifecycle leaves behind and
nothing incidental to one machine:

- `stages`: the lifecycle stages in the order the report evidences them;
- `report`: the keys and value types of the signed report (a list is
  described by its first item, a value by its type name), because the
  numbers depend on the machine and the shape is the contract;
- `files`: every file the run writes, package and checkpoint included.

A deliberate change regenerates the goldens in the same commit, naming the
register rows it closes:

    SCMLE_WRITE_GOLDENS=1 uv run pytest tests/integration/test_modality_characterization.py
"""

import json
import os
import pathlib

import pytest

from scmle.pipeline import modality_lifecycle
from tests.unit import test_modality_lifecycle

GOLDENS = pathlib.Path(__file__).parents[1] / "fixtures" / "modality"
WRITE = os.environ.get("SCMLE_WRITE_GOLDENS") == "1"
SEED = 17
KEY = "characterization-key"
FACTORIES = {
    "document": test_modality_lifecycle.document_adapter,
    "text": test_modality_lifecycle.text_adapter,
    "image": test_modality_lifecycle.image_adapter,
    "audio": test_modality_lifecycle.audio_adapter,
    "video": test_modality_lifecycle.video_adapter,
    "graph": test_modality_lifecycle.graph_adapter,
    "geospatial": test_modality_lifecycle.geospatial_adapter,
    "raster": test_modality_lifecycle.raster_adapter,
}


def shape(value):
    """The keys and value types of JSON data, without its values."""
    if isinstance(value, dict):
        return {str(key): shape(item) for key, item in sorted(value.items())}
    if isinstance(value, list):
        return [shape(value[0])] if value else []
    return type(value).__name__


def capture(root):
    """Everything a golden records about one modality run."""
    report = modality_lifecycle.Lifecycle().run(
        FACTORIES[root.name](),
        root / "package",
        signing_key=KEY,
        seed=SEED,
        report_path=root / "modality_lifecycle.json",
    )
    return {
        "stages": [evidence.stage.value for evidence in report.evidence],
        "report": shape(report.to_dict()),
        "files": sorted(
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file()
        ),
    }


@pytest.mark.parametrize("modality", sorted(FACTORIES))
def test_a_modality_run_produces_what_its_golden_records(modality, tmp_path):
    root = tmp_path / modality
    root.mkdir()
    found = capture(root)
    golden = GOLDENS / f"{modality}.json"
    if WRITE:
        golden.write_text(json.dumps(found, indent=1, sort_keys=True) + "\n")
        return
    expected = json.loads(golden.read_text(encoding="utf-8"))
    assert found["stages"] == expected["stages"]
    assert found["files"] == expected["files"]
    assert found["report"] == expected["report"]


def test_every_modality_golden_belongs_to_a_modality_run():
    names = {path.stem for path in GOLDENS.glob("*.json")}
    assert names - {"checkpoint_v1", "checkpoint_v2"} == set(FACTORIES)
