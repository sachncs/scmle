"""The static check finds every deserialiser call and judges where its bytes
come from (C9: no unauthenticated deserialisation path)."""

import json
import pathlib
import textwrap

from scmle.security import deserialisation

ROOT = pathlib.Path(__file__).resolve().parents[2]


def package(tmp_path, **modules):
    root = tmp_path / "pkg"
    root.mkdir()
    (root / "__init__.py").write_text("", encoding="utf-8")
    for name, text in modules.items():
        (root / f"{name}.py").write_text(
            textwrap.dedent(text), encoding="utf-8"
        )
    return root


def verdicts(root):
    return [(s.key, s.sink, s.verdict) for s in deserialisation.scan(root)]


def test_a_path_or_unknown_bytes_is_unauthenticated(tmp_path):
    root = package(
        tmp_path,
        loose="""
        import io
        import pickle
        import joblib
        from yaml import load
        import numpy

        def path(p):
            return joblib.load(p)

        def raw(data):
            return pickle.loads(data)

        def unknown(blob):
            buffer = blob + b""
            return joblib.load(io.BytesIO(buffer))

        def handle(p):
            with open(p, "rb") as stream:
                return pickle.load(stream)

        def yamlish(text):
            return load(text)

        def arrays(p):
            return numpy.load(p, allow_pickle=True)

        STATE = pickle.loads(b"")
        """,
    )
    assert verdicts(root) == [
        ("pkg.loose:path", "joblib.load", "unauthenticated"),
        ("pkg.loose:raw", "pickle.loads", "unauthenticated"),
        ("pkg.loose:unknown", "joblib.load", "unauthenticated"),
        ("pkg.loose:handle", "pickle.load", "unauthenticated"),
        ("pkg.loose:yamlish", "yaml.load", "unauthenticated"),
        ("pkg.loose:arrays", "numpy.load", "unauthenticated"),
        ("pkg.loose:<module>", "pickle.loads", "unauthenticated"),
    ]


def test_calls_the_library_makes_safe_are_not_sites(tmp_path):
    root = package(
        tmp_path,
        safe="""
        import numpy
        import torch
        import yaml

        def arrays(p):
            return numpy.load(p), numpy.load(p, allow_pickle=False)

        def weights(p):
            return torch.load(p, weights_only=True)

        def config(text):
            return yaml.load(text, Loader=yaml.SafeLoader)
        """,
    )
    assert verdicts(root) == []


def test_bytes_from_an_authenticating_reader_are_authenticated(tmp_path):
    root = package(
        tmp_path,
        good="""
        import io
        import joblib
        from scmle import envelope

        class Store:
            def load(self, path):
                meta, payload = envelope.read(path)
                return joblib.load(io.BytesIO(payload))

        def package(path, digest):
            data = envelope.fetch(path, digest)
            return joblib.load(io.BytesIO(data))

        def wrong(path):
            payload, meta = envelope.read(path)
            return joblib.load(io.BytesIO(payload))
        """,
    )
    assert verdicts(root) == [
        ("pkg.good:Store.load", "joblib.load", "authenticated"),
        ("pkg.good:package", "joblib.load", "authenticated"),
        ("pkg.good:wrong", "joblib.load", "unauthenticated"),
    ]


def test_a_helper_is_forwarded_only_when_every_caller_authenticates(
    tmp_path,
):
    helper = """
    import io
    import joblib
    from scmle import envelope

    def unpickle(data, name):
        return joblib.load(io.BytesIO(data))

    def good(path, digest):
        data = envelope.fetch(path, digest)
        return unpickle(data, "x")
    """
    root = package(tmp_path, helper=helper)
    assert verdicts(root) == [
        ("pkg.helper:unpickle", "joblib.load", "forwarded")
    ]
    (root / "other.py").write_text(
        textwrap.dedent(
            """
            import pathlib
            from pkg import helper

            def bad(path):
                return helper.unpickle(pathlib.Path(path).read_bytes(), "y")
            """
        ),
        encoding="utf-8",
    )
    [site] = deserialisation.scan(root)
    assert site.verdict == "unauthenticated"
    assert site.reason == "data is not authenticated in pkg.other:bad"


def test_an_unused_helper_is_unauthenticated(tmp_path):
    root = package(
        tmp_path,
        lone="""
        import io
        import joblib

        def unpickle(data):
            return joblib.load(io.BytesIO(data))
        """,
    )
    [site] = deserialisation.scan(root)
    assert (site.verdict, site.reason) == (
        "unauthenticated",
        "data comes from no caller",
    )


def test_a_buffer_the_function_dumped_is_a_round_trip(tmp_path):
    root = package(
        tmp_path,
        trip="""
        import io
        import joblib

        def check(model):
            buffer = io.BytesIO()
            joblib.dump(model, buffer)
            buffer.seek(0)
            return joblib.load(buffer)

        def fake(model, given):
            buffer = given
            joblib.dump(model, buffer)
            return joblib.load(buffer)
        """,
    )
    assert verdicts(root) == [
        ("pkg.trip:check", "joblib.load", "roundtrip"),
        ("pkg.trip:fake", "joblib.load", "unauthenticated"),
    ]


def test_import_aliases_and_from_imports_are_resolved(tmp_path):
    root = package(
        tmp_path,
        names="""
        import cloudpickle
        from pickle import Unpickler
        from pandas import read_pickle

        def a(x):
            return cloudpickle.loads(x)

        def b(x):
            return Unpickler(x).load()

        def c(x):
            return read_pickle(x)
        """,
    )
    assert [v[1] for v in verdicts(root)] == [
        "cloudpickle.loads",
        "pickle.Unpickler",
        "pandas.read_pickle",
    ]


def test_every_site_in_the_package_is_authenticated():
    found = deserialisation.scan(ROOT / "scmle")
    assert len(found) == 8
    assert [s for s in found if s.verdict == "unauthenticated"] == []
    assert {s.key: s.verdict for s in found} == {
        "scmle.artifacts.bundle.archive:unpickle": "forwarded",
        "scmle.models.conformance:smoke": "roundtrip",
        "scmle.models.deeplearning.estimators:NetworkEstimator.restore": (
            "authenticated"
        ),
        "scmle.models.deeplearning.jepa:Embedder.restore": "authenticated",
        "scmle.models.documents.archive:Archive.load": "authenticated",
        "scmle.models.images.archive:Archive.load": "authenticated",
        "scmle.pipeline.modality_lifecycle.verification:load_package": (
            "authenticated"
        ),
        "scmle.pipeline.products:Kept.load": "authenticated",
    }


def test_the_command_prints_sites_and_fails_on_an_unauthenticated_one(
    tmp_path, capsys
):
    root = package(
        tmp_path,
        bad="""
        import joblib

        def f(p):
            return joblib.load(p)
        """,
    )
    assert deserialisation.main([str(root)]) == 1
    printed = json.loads(capsys.readouterr().out)
    assert printed == [
        {
            "module": "pkg.bad",
            "function": "f",
            "line": 5,
            "sink": "joblib.load",
            "verdict": "unauthenticated",
            "reason": "reads p, not authenticated bytes",
        }
    ]
    assert deserialisation.main([str(ROOT / "scmle")]) == 0
