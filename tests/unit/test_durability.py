"""Tests for filesystem durability boundaries."""

import pathlib

from scml import durability


def test_sync_directory_flushes_a_real_directory(tmp_path: pathlib.Path):
    """A directory handle can flush an atomic replacement on Unix."""
    durability.sync_directory(tmp_path)
