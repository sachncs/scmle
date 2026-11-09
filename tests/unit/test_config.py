"""Tests for project configuration files."""

import json

import pytest

from scmle.contracts import errors
from scmle.core import config


def test_sections_build_typed_specs_and_reject_unknown_names(tmp_path):
    path = tmp_path / "sections.json"
    path.write_text(json.dumps({"search": {"top_models": 1}}))
    assert config.load_sections(path)["search"].top_models == 1
    assert config.load_sections(None) == {}
    path.write_text(json.dumps({"mystery": {}}))
    with pytest.raises(errors.ConfigError, match="unknown config sections"):
        config.load_sections(path)
