"""The framing draft replays to the same bytes on every interpreter."""

import json
import os
from pathlib import Path

from tests.unit import replay_digests

FIXTURE = Path(__file__).parents[1] / "fixtures" / "replay_golden.json"


def test_the_seeded_framing_draft_matches_the_stored_digests():
    found = replay_digests.digests()
    if os.environ.get("SCMLE_WRITE_GOLDEN"):
        FIXTURE.write_text(json.dumps(found, indent=2, sort_keys=True) + "\n")
    stored = json.loads(FIXTURE.read_text())
    assert found == stored
