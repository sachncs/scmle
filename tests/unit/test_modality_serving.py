"""Reference HTTP serving tests for verified modality packages."""

import json
import urllib.error
import urllib.request

import pytest

from scml.contracts import errors
from scml.data.modalities import document
from scml.inference import modality_serving
from scml.models import modality_lifecycle


class DocumentDecoder(modality_serving.ModalitySampleDecoder):
    """Decode the deliberately small JSON document test contract."""

    def decode(self, payload: object) -> tuple[document.DocumentSample, ...]:
        """Decode labelled document records from a request object."""
        if not isinstance(payload, dict) or not isinstance(
            payload.get("samples"), list
        ):
            raise TypeError("payload must contain a samples list")
        samples = []
        for record in payload["samples"]:
            if not isinstance(record, dict):
                raise TypeError("every sample must be an object")
            samples.append(
                document.DocumentSample(
                    document_id=str(record["document_id"]),
                    content=str(record["content"]),
                    label=str(record.get("label", "unknown")),
                )
            )
        return tuple(samples)

    def schema(self) -> dict[str, object]:
        """Return the JSON sample contract."""
        return {
            "type": "object",
            "required": ["samples"],
            "sample_fields": ["document_id", "content", "label"],
        }


def adapter() -> modality_lifecycle.DocumentAdapter:
    """Build a small, labelled document fixture."""
    samples = tuple(
        document.DocumentSample(
            document_id=f"document-{index}",
            content=("alpha " if index % 2 else "beta ") * (index + 2),
            label="alpha" if index % 2 else "beta",
        )
        for index in range(12)
    )
    return modality_lifecycle.DocumentAdapter(document.DocumentDataset(samples))


def post(url: str, payload: object) -> tuple[int, dict[str, object]]:
    """Post JSON and return status plus decoded body."""
    request = urllib.request.Request(
        url,
        json.dumps(payload).encode(),
        {"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def test_modality_http_serving_verifies_health_schema_and_predictions(
    tmp_path,
):
    """A caller decoder can expose a signed modality package safely."""
    reference = adapter()
    report = modality_lifecycle.ModalityLifecycle().run(
        reference,
        tmp_path / "package",
        signing_key="serving-key",
    )
    predictor = modality_lifecycle.ModalityPredictor.from_package(
        reference,
        report.package.path,
        "serving-key",
    )
    serving = modality_serving.ModalityHttpServing(
        predictor,
        DocumentDecoder(),
        max_samples=2,
    )
    serving.start()
    try:
        with urllib.request.urlopen(f"{serving.address}/health") as response:
            health = json.loads(response.read())
        assert health["package_verified"] is True
        assert health["modality"] == "document"

        with urllib.request.urlopen(f"{serving.address}/schema") as response:
            schema = json.loads(response.read())
        assert schema["decoder"]["sample_fields"] == [
            "document_id",
            "content",
            "label",
        ]

        payload = {
            "samples": [
                {
                    "document_id": "request-1",
                    "content": "alpha alpha",
                    "label": "alpha",
                },
                {
                    "document_id": "request-2",
                    "content": "beta beta beta",
                    "label": "beta",
                },
            ]
        }
        status, body = post(f"{serving.address}/predict", payload)
        assert status == 200
        assert body["package_verified"] is True
        assert body["sample_count"] == 2
        assert len(body["predictions"]) == 2
        assert len(body["probabilities"]) == 2

        status, body = post(f"{serving.address}/predict", {"samples": []})
        assert status == 422
        assert body["category"] == "samples"

        status, body = post(
            f"{serving.address}/predict",
            {"samples": [payload["samples"][0]] * 3},
        )
        assert status == 413
        assert body["category"] == "request"

        status, body = post(f"{serving.address}/unknown", {})
        assert status == 404
        assert body["error"] == "not found"
    finally:
        serving.stop()

    with pytest.raises(errors.ScmlError, match="not running"):
        print(serving.address)


def test_modality_http_serving_rejects_invalid_request_bounds():
    """The reference server refuses unsafe zero-sized request limits."""
    with pytest.raises(ValueError, match="request bounds"):
        modality_serving.ModalityHttpServing(
            predictor=None,
            decoder=DocumentDecoder(),
            max_request_bytes=0,
        )
