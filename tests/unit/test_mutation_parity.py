"""The hand-written parity model, its comparison and its remote check."""

import io
import json
import pathlib
import urllib.error

from scmle.contracts import serving
from scmle.deploy import parity

ROOT = pathlib.Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "deploy" / "local" / "golden"
URL = "http://parity.invalid"


def test_the_preprocessing_names_its_steps_and_missing_category() -> None:
    prep = parity.assemble(["x"], ["c"])
    numeric = prep.transformers[0][1]
    categorical = prep.transformers[1][1]
    assert list(numeric.named_steps) == ["cast", "impute", "scale"]
    assert list(categorical.named_steps) == ["cast", "impute", "encode"]
    fill = categorical.named_steps["impute"].fill_value
    assert fill == parity.MISSING_CATEGORY


def test_the_hand_written_model_records_its_metric() -> None:
    golden = parity.goldens(GOLDEN)
    assert parity.handwrite(golden.spec).info.metric == "roc_auc"


def test_a_differing_label_is_named_by_record() -> None:
    expected = {"labels": [1], "probability_of_class_1": [0.9]}
    tolerance = parity.Tolerance(probability_abs=0.01)
    found = parity.compare(expected, [0], [[0.1, 0.9]], tolerance)
    assert found.to_dict() == {
        "passed": False,
        "records": 1,
        "error": 0.0,
        "tolerance": 0.01,
        "mismatches": ["record 0: label differs"],
    }


def test_an_http_error_fails_with_its_status(monkeypatch) -> None:
    def refuse(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 503, "busy", {}, None)

    monkeypatch.setattr(parity.urllib.request, "urlopen", refuse)
    assert parity.crosscheck(URL, GOLDEN) == {
        "passed": False,
        "mismatches": ["HTTP 503"],
    }


def test_a_reply_without_the_contract_keys_fails(monkeypatch) -> None:
    expected = json.loads((GOLDEN / "expected.json").read_text("utf-8"))
    body = {
        "predictions": expected["labels"],
        "probabilities": [
            [1.0 - p, p] for p in expected["probability_of_class_1"]
        ],
    }

    def answer(request, timeout):
        return io.BytesIO(json.dumps(body).encode())

    monkeypatch.setattr(parity.urllib.request, "urlopen", answer)
    report = parity.crosscheck(URL, GOLDEN)
    absent = [key for key in serving.RESPONSE_KEYS if key not in body]
    assert report["passed"] is False
    assert report["mismatches"] == [f"response lacks {absent}"]


def test_the_goldens_read_each_file_by_its_exact_name(monkeypatch) -> None:
    read = []
    real = parity.read_json

    def spy(path):
        read.append(pathlib.Path(path).name)
        return real(path)

    monkeypatch.setattr(parity, "read_json", spy)
    parity.goldens(GOLDEN)
    assert sorted(read) == [
        "model_spec.json",
        "records.json",
        "tolerances.json",
    ]
