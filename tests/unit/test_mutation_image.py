"""The Pillow guard of the image package, case by case."""

import pytest

from scmle.data.modalities import image


def test_the_default_operation_is_image_data(monkeypatch) -> None:
    monkeypatch.setattr(image, "PILLOW_AVAILABLE", False)
    with pytest.raises(image.DependencyError) as raised:
        image.pillow()
    assert raised.value.details["operation"] == "image data"


def test_a_flag_without_a_module_still_refuses(monkeypatch) -> None:
    monkeypatch.setattr(image, "PILLOW_AVAILABLE", True)
    monkeypatch.setattr(image, "PillowModule", None)
    with pytest.raises(image.DependencyError) as raised:
        image.pillow("decode")
    assert raised.value.details["operation"] == "decode"


def test_a_module_without_the_flag_still_refuses(monkeypatch) -> None:
    monkeypatch.setattr(image, "PILLOW_AVAILABLE", False)
    monkeypatch.setattr(image, "PillowModule", object())
    with pytest.raises(image.DependencyError):
        image.pillow("decode")


def test_an_available_module_is_returned(monkeypatch) -> None:
    module = object()
    monkeypatch.setattr(image, "PILLOW_AVAILABLE", True)
    monkeypatch.setattr(image, "PillowModule", module)
    assert image.pillow("decode") is module
