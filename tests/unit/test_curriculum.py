"""Every example in docs/CURRICULUM.md runs."""

import pathlib
import re

import pytest

PAGE = pathlib.Path(__file__).parents[2] / "docs" / "CURRICULUM.md"
BLOCKS = re.findall(r"```python\n(.*?)```", PAGE.read_text(), re.S)


def test_every_stage_has_an_example():
    assert len(BLOCKS) == 8


@pytest.mark.parametrize("index", range(8))
def test_the_example_runs(index):
    if "import torch" in BLOCKS[index]:
        pytest.importorskip("torch")
    exec(compile(BLOCKS[index], f"curriculum-stage-{index + 1}", "exec"), {})
