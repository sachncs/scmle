"""The data Kind: one protocol, the tabular kind as its first implementation."""

import numpy
import pandas
import pytest

from scmle.contracts import capabilities
from scmle.contracts import errors
from scmle.contracts import tasks
import scmle.contracts.data
from scmle.data.readers import tabular
from scmle.data.splitting import splitters
from scmle.evaluation import metrics
from scmle.evaluation import selection
import scmle.models.registry
from scmle.pipeline import engine
from scmle.pipeline import kinds
from scmle.pipeline import products
from scmle.pipeline import profiler
from scmle.pipeline import roster
import scmle.pipeline.services
from scmle.pipeline.stages import ingestion
from scmle.runtime import limits


def rows(count=60):
    generator = numpy.random.default_rng(0)
    return pandas.DataFrame(
        {
            "a": generator.normal(size=count),
            "b": generator.choice(["x", "y", "z"], size=count),
            "y": numpy.arange(count) % 2,
        }
    )


def config(uri="x.csv", **data):
    return tasks.Config(data=tasks.Dataset(uri=uri, target="y", **data))


def test_the_tabular_kind_is_a_kind_and_is_registered():
    kind = kinds.pick(config().data)
    assert isinstance(kind, kinds.Tabular)
    assert isinstance(kind, kinds.Kind)
    assert kind.name == "tabular"
    assert set(kinds.KINDS) == {
        "tabular",
        "document",
        "graph",
        "image",
        "video",
        "audio",
        "raster",
        "geospatial",
    }


def test_every_registered_kind_is_a_kind_named_as_registered():
    for name, kind in kinds.KINDS.items():
        assert isinstance(kind, kinds.Kind)
        assert kind.name == name


def test_the_roster_enrols_exactly_what_every_kind_declares():
    declared = {
        record.key for kind in kinds.KINDS.values() for record in kind.declare()
    }
    assert {record.key for record in roster.enrol().catalogue()} == declared


def test_an_unregistered_file_modality_is_refused_with_its_details():
    data = config("notes.txt", modality="text").data
    with pytest.raises(
        capabilities.UnsupportedError,
        match="file-backed modality has no registered ingestion adapter",
    ) as failure:
        kinds.pick(data)
    assert failure.value.details == {
        "status": "unsupported",
        "reason": "file-backed modality has no ingestion adapter",
        "modality": "text",
        "uri": "notes.txt",
        "supported_modalities": sorted(kinds.KINDS),
    }


def test_the_reader_reads_a_whole_file_without_a_stream(tmp_path):
    path = tmp_path / "data.csv"
    rows().to_csv(path, index=False)
    spec = config(str(path))
    frame, stream = kinds.Tabular().read(spec)
    assert stream is None
    pandas.testing.assert_frame_equal(frame, tabular.read(str(path)))


def test_the_reader_streams_a_file_over_the_memory_budget(tmp_path):
    path = tmp_path / "data.csv"
    rows(400).to_csv(path, index=False)
    spec = config(str(path), memory_budget_gb=1e-9)
    assert kinds.Tabular().overflows(spec.data)
    frame, stream = kinds.Tabular().read(spec)
    assert stream is not None
    assert stream[1].rows == 400
    assert "split_side" in frame.columns


def test_the_findings_are_the_profilers_portrait():
    spec = config()
    frame = rows()
    manifest = tabular.manifested(frame, "y", "x.csv", "tabular")
    found = kinds.Tabular().survey(
        spec, frame, manifest, tasks.Task.BINARY, 0.5
    )
    expected = profiler.Profiler(spec).profile(
        frame, manifest, tasks.Task.BINARY, 0.5
    )
    assert isinstance(found, scmle.contracts.data.Portrait)
    assert found.plan == expected.plan
    assert found.quality == expected.quality


def test_the_split_constraints_are_the_splitter_of_the_spec():
    spec = config(group_column="g")
    splitter = kinds.Tabular().constrain(spec, tasks.Task.BINARY)
    assert isinstance(splitter, splitters.Splitter)
    assert splitter.strategy == splitters.scheme(spec, tasks.Task.BINARY)


def test_the_featuriser_drops_every_column_outside_the_plan():
    plan = scmle.contracts.data.Blueprint(numeric=("a",), categorical=("b",))
    featuriser = kinds.Tabular().featurise(
        plan, scmle.contracts.data.Preprocessing()
    )
    values = featuriser.fit_transform(rows())
    assert values.shape[0] == 60
    assert "y" not in set(featuriser.get_feature_names_out())


def test_the_decoder_keeps_ordinal_labels_and_defers_otherwise():
    assert kinds.Tabular().decode(("low", "high")) == ("low", "high")
    assert kinds.Tabular().decode(None) is None


def test_the_featurise_cost_is_the_profilers_bound():
    kind = kinds.Tabular()
    assert kind.cost(1000, 10, 2, 600.0) == profiler.estimate(
        1000, 10, 2, 600.0
    )
    with pytest.raises(errors.InputError):
        kind.cost(-1, 10, 2, 600.0)


def provide(spec):
    return scmle.pipeline.services.Services(
        spec,
        scmle.models.registry.Catalog.stock(),
        metrics.CATALOG,
        selection.Averager(spec.selection),
        limits.Governor(spec.resources, 100.0),
        frame=rows(),
    )


def test_a_board_holds_each_applied_product_by_type():
    board = engine.Board.start(provide(config()))
    with pytest.raises(errors.OrderError, match="Intake"):
        board.get(ingestion.Intake)
    assert board.find(ingestion.Intake) is None
    made = ingestion.Ingest().execute(board)
    assert board.get(ingestion.Intake) is made
    assert board.find(ingestion.Intake) is made
    assert made.kind is kinds.TABULAR


def test_a_product_applied_to_a_board_without_products_changes_nothing():
    class Board:
        pass

    target = Board()
    made = ingestion.Intake(
        rows(),
        tabular.manifested(rows(), "y", "x.csv", "tabular"),
        [],
        0.0,
        None,
    )
    made.apply(target)
    assert vars(target) == {}
    assert isinstance(made, products.Derived)
