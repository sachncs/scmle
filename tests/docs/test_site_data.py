"""Data and modality statements of the site that no other test enforced."""

import pytest

from scmle.contracts import data
from scmle.contracts import modalities
from scmle.pipeline import modality_lifecycle
from tests.unit.test_modality_lifecycle import document_adapter
from tests.unit.test_modality_lifecycle import image_adapter


def passport(adapter, **changes):
    """A manifest that matches the adapter unless a field is changed.

    Args:
        adapter: The adapter the manifest describes.
        **changes: Fields to replace (`modality`, `samples` or `digests`).

    Returns:
        The manifest.
    """
    fields = {
        "identifier": "document-dataset",
        "snapshot": "document-snapshot",
        "modality": "document",
        "task": "binary_classification",
        "samples": len(adapter.samples),
        "digests": {"content": adapter.digest},
        "source": data.Origin(
            owner="records-team", acquisition="export", license="internal"
        ),
    }
    return data.Passport(**(fields | changes))


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"modality": "image"}, "modality must match"),
        ({"digests": {"content": "0" * 64}}, "content_hash must match"),
        ({"samples": 1}, "sample_count must match"),
    ],
)
def test_the_lifecycle_refuses_a_manifest_that_does_not_match_the_adapter(
    tmp_path, changes, message
):
    """A mismatched manifest stops the run before any package is written.

    Also enforces website/src/pages/docs/data.astro:199.
    Also enforces website/src/pages/docs/extensions.astro:151.
    """
    adapter = document_adapter()
    target = tmp_path / "package"
    with pytest.raises(ValueError, match=message):
        modality_lifecycle.Lifecycle().run(
            adapter,
            target,
            signing_key="test-signing-key",
            manifest=passport(adapter, **changes),
        )
    assert not target.exists()


def test_image_evidence_says_it_is_not_saliency_or_localization(tmp_path):
    """The image baseline states that it is neither saliency nor location.

    Also enforces website/src/pages/docs/data.astro:342.
    Also enforces website/src/pages/docs/modalities.astro:138.
    Also enforces website/src/pages/docs/modalities.astro:123.
    """
    report = modality_lifecycle.Lifecycle().run(
        image_adapter(),
        tmp_path / "image-package",
        signing_key="test-signing-key",
    )
    evidence = {item.stage: item.evidence for item in report.evidence}[
        modalities.Phase.MODEL_EXPLAINABILITY
    ]
    limits = " ".join(evidence["modality_specific"]["limitations"])
    assert "not object localization or validated saliency" in limits
    assert evidence["modality_specific"]["feature_space_only"] is True
