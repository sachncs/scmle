"""Reference HTTP serving tests for verified modality packages."""

import json
import re
from typing import cast
import urllib.error
import urllib.request

import pytest

import scmle
from scmle.contracts import errors
from scmle.data.modalities import document
from scmle.inference import modality_serving
from scmle.models import modality_lifecycle


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


class ExplodingDecoder(DocumentDecoder):
    """Raise an unexpected decoder error for server-boundary testing."""

    def decode(self, payload: object) -> tuple[document.DocumentSample, ...]:
        """Raise a representative unexpected decoder failure."""
        raise RuntimeError("decoder failed")


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


def request(
    url: str, body: bytes, content_type: str, request_id: str | None = None
) -> tuple[int, dict[str, object]]:
    """Send a request body and return status plus decoded JSON."""
    headers = {"Content-Type": content_type}
    if request_id is not None:
        headers["X-Request-ID"] = request_id
    request = urllib.request.Request(
        url,
        body,
        headers,
    )
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def post(
    url: str, payload: object, request_id: str | None = None
) -> tuple[int, dict[str, object]]:
    """Post JSON and return status plus decoded body."""
    return request(
        url,
        json.dumps(payload).encode(),
        "application/json",
        request_id=request_id,
    )


def test_modality_http_serving_verifies_health_schema_and_predictions(
    tmp_path,
):
    """A caller decoder can expose a signed modality package safely."""
    assert scmle.ModalityHttpServing is modality_serving.ModalityHttpServing
    assert scmle.ModalitySampleDecoder is modality_serving.ModalitySampleDecoder
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
        assert re.fullmatch(r"srv-[0-9a-f]{32}", cast(str, body["request_id"]))
        assert body["package_verified"] is True
        assert body["sample_count"] == 2
        assert len(cast(list[object], body["predictions"])) == 2
        assert len(cast(list[object], body["probabilities"])) == 2

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

        status, body = request(
            f"{serving.address}/predict", b"{}", "text/plain"
        )
        assert status == 415
        assert body["category"] == "request"
        assert re.fullmatch(r"srv-[0-9a-f]{32}", cast(str, body["request_id"]))

        status, body = request(
            f"{serving.address}/predict", b"not-json", "application/json"
        )
        assert status == 400
        assert body["category"] == "request"
        assert re.fullmatch(r"srv-[0-9a-f]{32}", cast(str, body["request_id"]))
    finally:
        serving.stop()

    with pytest.raises(errors.ScmleError, match="not running"):
        print(serving.address)


def test_release_bound_serving_publishes_verified_lifecycle_identity(tmp_path):
    """Serving can require the exact signed lifecycle report for its package."""
    reference = adapter()
    report_path = tmp_path / "modality_lifecycle.json"
    report = modality_lifecycle.ModalityLifecycle().run(
        reference,
        tmp_path / "package",
        signing_key="serving-key",
        report_path=report_path,
    )
    predictor = modality_lifecycle.ModalityPredictor.from_package(
        reference,
        report.package.path,
        "serving-key",
        lifecycle_report_path=report_path,
    )
    serving = modality_serving.ModalityHttpServing(
        predictor,
        DocumentDecoder(),
        require_lifecycle_report=True,
    )

    health = predictor.health()

    assert health["lifecycle_report_verified"] is True
    assert isinstance(health["lifecycle_report_digest"], str)
    assert serving.require_lifecycle_report is True


def test_release_bound_serving_rejects_unbound_predictor(tmp_path):
    """The production-boundary mode must not accept package-only loading."""
    reference = adapter()
    report = modality_lifecycle.ModalityLifecycle().run(
        reference,
        tmp_path / "package",
        signing_key="serving-key",
    )
    predictor = modality_lifecycle.ModalityPredictor.from_package(
        reference, report.package.path, "serving-key"
    )

    with pytest.raises(ValueError, match="verified lifecycle report"):
        modality_serving.ModalityHttpServing(
            predictor,
            DocumentDecoder(),
            require_lifecycle_report=True,
        )


def test_report_binding_rejects_tampered_lifecycle_evidence(tmp_path):
    """A changed lifecycle report cannot be used to authorize serving."""
    reference = adapter()
    report_path = tmp_path / "modality_lifecycle.json"
    report = modality_lifecycle.ModalityLifecycle().run(
        reference,
        tmp_path / "package",
        signing_key="serving-key",
        report_path=report_path,
    )
    record = json.loads(report_path.read_text(encoding="utf-8"))
    record["dataset_hash"] = "tampered"
    report_path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="signature does not verify"):
        modality_lifecycle.ModalityPredictor.from_package(
            reference,
            report.package.path,
            "serving-key",
            lifecycle_report_path=report_path,
        )


def test_modality_http_serving_rejects_invalid_request_bounds():
    """The reference server refuses unsafe zero-sized request limits."""
    with pytest.raises(ValueError, match="request bounds"):
        modality_serving.ModalityHttpServing(
            predictor=cast(modality_lifecycle.ModalityPredictor, None),
            decoder=DocumentDecoder(),
            max_request_bytes=0,
        )


def test_modality_http_serving_classifies_unexpected_decoder_errors(tmp_path):
    """Unexpected decoder errors become a bounded structured response."""
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
        ExplodingDecoder(),
    )
    serving.start()
    try:
        status, body = post(
            f"{serving.address}/predict",
            {"samples": [{"document_id": "request", "content": "alpha"}]},
        )
        assert status == 500
        assert body["error"] == "internal modality serving failure"
        assert body["category"] == "internal"
        assert re.fullmatch(r"srv-[0-9a-f]{32}", cast(str, body["request_id"]))
    finally:
        serving.stop()


def test_modality_http_serving_echoes_valid_request_identity(tmp_path):
    """A valid caller identity is echoed in the body and response header."""
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
    serving = modality_serving.ModalityHttpServing(predictor, DocumentDecoder())
    serving.start()
    try:
        request = urllib.request.Request(
            f"{serving.address}/health",
            headers={"X-Request-ID": "caller.request-42"},
        )
        with urllib.request.urlopen(request) as response:
            body = json.loads(response.read())
            assert response.headers["X-Request-ID"] == "caller.request-42"
        assert body["request_id"] == "caller.request-42"
    finally:
        serving.stop()


@pytest.mark.parametrize("request_id", ["", "bad value", "é", "x" * 129])
def test_modality_http_serving_rejects_unsafe_request_identity(
    tmp_path, request_id
):
    """Unsafe caller identities are rejected without echoing their contents."""
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
    serving = modality_serving.ModalityHttpServing(predictor, DocumentDecoder())
    serving.start()
    try:
        status, body = post(
            f"{serving.address}/predict",
            {"samples": []},
            request_id=request_id,
        )
        assert status == 400
        assert body["category"] == "request"
        assert "request_id" in body
        if request_id:
            assert request_id not in json.dumps(body)
        assert re.fullmatch(r"srv-[0-9a-f]{32}", cast(str, body["request_id"]))
    finally:
        serving.stop()
