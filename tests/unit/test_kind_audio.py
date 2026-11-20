"""The audio kind on the shared stages (plan commit 40, register rows MM-6,
MM-10, MM-12, MM-13, MM-16 and MM-18, and the A9 split-leakage property).

Clips are listed in a manifest, described whole in bounded chunks, tied
into cohorts (copies, near-copies, declared groups) and run through the
same engine and gates as a table: planning, search, cross-validation, the
locked test set, explanations and the package.
"""

import importlib.util
import json
import math
import os
import pathlib
import tracemalloc

import hypothesis
from hypothesis import strategies
import numpy
import pandas
import pytest
from sklearn import linear_model

import scmle.api.project
from scmle.contracts import capabilities
from scmle.contracts import errors
from scmle.contracts import tasks
from scmle.data import modalities
from scmle.data.modalities.audio import spectrum
from scmle.pipeline import kinds
from scmle.pipeline import modality_lifecycle
from scmle.pipeline.kinds import audio
from scmle.pipeline.kinds import rows
from tests import support

RATE = 16_000
GOLDENS = pathlib.Path(__file__).parents[1] / "fixtures" / "modality" / "audio"
WRITE = os.environ.get("SCMLE_WRITE_GOLDENS") == "1"


def clip(rng, label, seconds=1.0, rate=RATE):
    """A clip of three tones with random on/off envelopes.

    The first tone sits near a centre frequency the label picks, so the
    label is learnable from the spectrum; the envelopes make every clip's
    band energies over time its own.
    """
    count = int(rate * seconds)
    times = numpy.arange(count) / rate
    signal = numpy.zeros(count)
    centre = (300.0, 1_500.0, 3_500.0)[label % 3]
    for tone in range(3):
        frequency = (
            centre * rng.uniform(0.8, 1.25)
            if tone == 0
            else rng.uniform(100.0, 6_000.0)
        )
        steps = rng.uniform(0.0, 1.0, 12) ** 2
        envelope = numpy.repeat(steps, math.ceil(count / 12))[:count]
        weight = 1.0 if tone == 0 else 0.4
        signal += (
            weight * envelope * numpy.sin(2 * numpy.pi * frequency * times)
        )
    signal += rng.normal(0.0, 0.01, count)
    return 0.5 * signal / numpy.abs(signal).max()


def write(path, values, rate=RATE):
    """Writes frames (or frames by channels) as 16-bit WAV."""
    matrix = values[:, None] if values.ndim == 1 else values
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        modalities.audio.encode(
            (matrix.shape[1], 2, rate, len(matrix), "NONE", "not compressed"),
            matrix,
        )
    )
    return path


def noisy(rng, values, decibels):
    """Adds white noise `decibels` below the signal's power."""
    level = numpy.sqrt(numpy.mean(numpy.square(values)))
    return values + rng.normal(0.0, level / 10 ** (decibels / 20), len(values))


def collection(root, count=90, seed=0):
    """A labelled manifest of `count` distinct clips."""
    rng = numpy.random.default_rng(seed)
    entries = []
    for index in range(count):
        write(root / "clips" / f"c{index:03d}.wav", clip(rng, index))
        entries.append(
            {
                "path": f"clips/c{index:03d}.wav",
                "label": ("low", "mid", "high")[index % 3],
            }
        )
    manifest = root / "manifest.csv"
    pandas.DataFrame(entries).to_csv(manifest, index=False)
    return manifest


def specification(uri, target="label", **validation):
    """A run specification for an audio manifest."""
    return tasks.Config(
        data=tasks.Dataset(uri=str(uri), target=target, modality="audio"),
        validation=tasks.Validation(**validation),
    )


@pytest.fixture
def registered():
    """The audio kind, as `kinds.KINDS` registers it."""
    assert kinds.KINDS[audio.AUDIO.name] is audio.AUDIO
    return audio.AUDIO


def test_a_clip_is_described_whole_not_its_first_frames(tmp_path):
    """MM-13: a change after frame 100,000 changes the description."""
    rng = numpy.random.default_rng(1)
    head = clip(rng, 0, seconds=8.0)
    tail = head.copy()
    tail[110_000:] = 0.0
    first = write(tmp_path / "first.wav", head)
    second = write(tmp_path / "second.wav", tail)

    old = modalities.audio.Dataset(
        (
            modalities.audio.Sample.decode(first.read_bytes(), source="first"),
            modalities.audio.Sample.decode(
                second.read_bytes(), source="second"
            ),
        )
    ).prepare()
    prefix = old.materialize()[:, :6]
    assert (prefix[0] == prefix[1]).all()  # the defect: only a prefix counts

    described = audio.AUDIO.describe([first, second])
    assert described.loc[0, "audio_rms"] != described.loc[1, "audio_rms"]
    assert described.loc[0, "audio_silence"] < described.loc[1, "audio_silence"]


def test_describing_a_long_clip_holds_a_bounded_working_set(tmp_path):
    """MM-6: memory follows the chunk, not the clip.

    Decoding 3,000,000 stereo frames at once would hold 48 MB of float64;
    the description must stay within a few chunks (16 chunk-sized arrays).
    """
    rng = numpy.random.default_rng(2)
    frames = 3_000_000
    values = 0.1 * rng.standard_normal((frames, 2))
    path = write(tmp_path / "long.wav", values)
    del values
    bound = 16 * spectrum.CHUNK * 2 * 8

    tracemalloc.start()
    try:
        trace = spectrum.describe(path)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    assert trace.features["audio_duration"] == frames / RATE
    assert peak < bound, f"peak {peak} bytes over the bound {bound}"


def test_a_clip_without_frames_is_a_validation_error(tmp_path):
    """MM-16: a zero-frame WAV is bad data, not a missing dependency."""
    path = write(tmp_path / "empty.wav", numpy.zeros((0, 1)))
    with pytest.raises(modalities.audio.ValidationError) as caught:
        spectrum.describe(path)
    assert caught.value.details["code"] == "empty_audio_frames"


def test_an_unreadable_clip_is_named_in_the_error(tmp_path):
    (tmp_path / "clips").mkdir()
    (tmp_path / "clips" / "bad.wav").write_bytes(b"RIFF\0\0\0\0WAVEjunk")
    pandas.DataFrame({"path": ["clips/bad.wav"], "label": ["a"]}).to_csv(
        tmp_path / "m.csv", index=False
    )
    with pytest.raises(modalities.audio.ValidationError) as caught:
        audio.AUDIO.read(specification(tmp_path / "m.csv"))
    assert caught.value.details["source"].endswith("bad.wav")


def test_a_compressed_clip_without_soundfile_names_the_library(tmp_path):
    if importlib.util.find_spec("soundfile") is not None:
        pytest.skip("soundfile is installed, so a FLAC clip is decoded")
    path = tmp_path / "clip.flac"
    path.write_bytes(b"fLaC" + bytes(64))
    with pytest.raises(modalities.audio.DecoderError, match="soundfile"):
        spectrum.describe(path)


def test_a_missing_file_or_path_column_is_a_data_error(tmp_path):
    pandas.DataFrame({"file": ["a.wav"], "label": ["a"]}).to_csv(
        tmp_path / "m.csv", index=False
    )
    with pytest.raises(errors.DataError, match="'path' column"):
        audio.AUDIO.read(specification(tmp_path / "m.csv"))
    pandas.DataFrame({"path": ["gone.wav"], "label": ["a"]}).to_csv(
        tmp_path / "m.csv", index=False
    )
    with pytest.raises(errors.DataError, match="do not exist"):
        audio.AUDIO.read(specification(tmp_path / "m.csv"))


def test_the_order_of_the_manifest_changes_nothing(tmp_path):
    """MM-12: rows, cohorts and the split do not follow caller order."""
    manifest = collection(tmp_path, count=24)
    listed = pandas.read_csv(manifest)
    listed.iloc[::-1].to_csv(tmp_path / "reversed.csv", index=False)

    forward = audio.AUDIO.read(specification(manifest))[0]
    backward = audio.AUDIO.read(specification(tmp_path / "reversed.csv"))[0]

    pandas.testing.assert_frame_equal(forward, backward)


def test_a_labelled_directory_reads_like_a_manifest(tmp_path):
    rng = numpy.random.default_rng(3)
    for index in range(6):
        label = ("low", "high")[index % 2]
        write(tmp_path / "data" / label / f"{index}.wav", clip(rng, index))

    frame = audio.AUDIO.read(specification(tmp_path / "data"))[0]

    assert sorted(frame["label"]) == ["high"] * 3 + ["low"] * 3
    assert "path" not in frame.columns


def test_copies_and_near_copies_share_a_cohort(tmp_path):
    """MM-10: exact, louder and noisy copies fall in their source's cohort."""
    rng = numpy.random.default_rng(4)
    entries = []
    for index in range(12):
        write(tmp_path / f"s{index}.wav", clip(rng, index))
        entries.append({"path": f"s{index}.wav", "label": index % 2})
    source = clip(numpy.random.default_rng(99), 0)
    copies = {
        "orig.wav": source,
        "copy.wav": source,
        "quiet.wav": 0.3 * source,
        "noisy.wav": noisy(rng, source, spectrum.NOISE),
        "cleaner.wav": noisy(rng, source, spectrum.NOISE + 10),
    }
    for name, values in copies.items():
        write(tmp_path / name, values)
        entries.append({"path": name, "label": 1})
    pandas.DataFrame(entries).to_csv(tmp_path / "m.csv", index=False)
    listed = pandas.read_csv(tmp_path / "m.csv")

    frame = audio.AUDIO.read(specification(tmp_path / "m.csv"))[0]
    names = sorted(listed["path"])
    cohort = dict(zip(names, frame[rows.KEY], strict=True))

    assert len({cohort[name] for name in copies}) == 1
    others = {cohort[f"s{index}.wav"] for index in range(12)}
    assert cohort["orig.wav"] not in others
    assert len(others) >= 10  # distinct clips stay apart


def test_a_declared_group_joins_its_clips(tmp_path):
    manifest = collection(tmp_path, count=120)
    listed = pandas.read_csv(manifest).assign(
        speaker=[f"s{index // 4}" for index in range(120)]
    )
    listed.to_csv(manifest, index=False)
    spec = specification(manifest)
    spec = spec.model_copy(
        update={
            "data": spec.data.model_copy(update={"group_column": "speaker"})
        }
    )

    frame = audio.AUDIO.read(spec)[0]
    split = audio.AUDIO.constrain(spec, tasks.Task.MULTICLASS).split(frame)

    speakers = frame["speaker"].to_numpy()
    train = numpy.asarray(split.manifest.training)
    test = numpy.asarray(split.manifest.test)
    assert not set(speakers[train]) & set(speakers[test])
    for fit, held in split.manifest.folds:
        assert not set(speakers[train[list(fit)]]) & set(
            speakers[train[list(held)]]
        )


@hypothesis.settings(max_examples=40, deadline=None)
@hypothesis.given(
    sizes=strategies.lists(strategies.integers(1, 5), min_size=12, max_size=40),
    seed=strategies.integers(0, 2**16),
    classification=strategies.booleans(),
)
def test_no_cohort_straddles_the_locked_set_or_a_fold(
    sizes, seed, classification
):
    """A9: every cohort lies wholly in one side of the split and each fold."""
    rng = numpy.random.default_rng(seed)
    owners = numpy.repeat(numpy.arange(len(sizes)), sizes)
    cohort = [f"cohort-{owner}" for owner in owners]
    target = (
        (owners % 2).tolist()
        if classification
        else rng.normal(size=len(cohort)).tolist()
    )
    frame = pandas.DataFrame(
        {
            "x": rng.normal(size=len(cohort)),
            "y": target,
            rows.KEY: cohort,
        }
    )
    spec = tasks.Config(
        data=tasks.Dataset(uri="clips.csv", target="y", modality="audio"),
        validation=tasks.Validation(folds=3, seed=seed),
    )
    task = tasks.Task.BINARY if classification else tasks.Task.REGRESSION

    try:
        split = audio.AUDIO.constrain(spec, task).split(frame)
    except errors.BlockedError:
        hypothesis.assume(False)
        return

    labels = numpy.asarray(cohort)
    train = numpy.asarray(split.manifest.training)
    test = numpy.asarray(split.manifest.test)
    assert not set(labels[train]) & set(labels[test])
    for fit, held in split.manifest.folds:
        assert not set(labels[train[list(fit)]]) & set(
            labels[train[list(held)]]
        )
    assert rows.KEY not in split.features.columns
    assert rows.KEY not in split.test.features.columns


def test_a_protocol_that_could_split_a_cohort_becomes_the_group_split():
    frame = pandas.DataFrame(
        {
            "x": numpy.arange(100.0),
            "y": numpy.arange(100) % 2,
            rows.KEY: [f"cohort-{index // 2}" for index in range(100)],
        }
    )
    spec = specification("clips.csv", target="y", strategy="stratified")
    split = audio.AUDIO.constrain(spec, tasks.Task.BINARY).split(frame)
    assert split.manifest.strategy == "group"
    assert any(
        "stratified split replaced by the group split" in note
        for note in split.manifest.notes
    )


def test_the_kind_declares_the_tabular_tasks_and_bounds_its_cost():
    declared = audio.AUDIO.declare()
    assert {item.key.task for item in declared} == set(rows.TASKS)
    assert {item.key.modality for item in declared} == {
        capabilities.Modality.AUDIO
    }
    assert {item.status for item in declared} == {
        capabilities.Status.EXPERIMENTAL
    }
    small = audio.AUDIO.cost(100, 20, 2, 60.0)
    assert audio.AUDIO.cost(1_000, 20, 2, 60.0) > small > 100 * audio.SECONDS


def test_the_noise_probe_decodes_only_the_frames_it_scores(tmp_path):
    """MM-18: the lifecycle's noise probe no longer decodes a whole file.

    The probe scores features of the first 100,000 frames; decoding all
    2,000,000 frames of the clip held about 64 MB of float64 arrays.
    """
    rng = numpy.random.default_rng(5)
    values = 0.1 * rng.standard_normal(2_000_000)
    sample = modalities.audio.Sample.decode(
        write(tmp_path / "long.wav", values).read_bytes(),
        source="long",
        label="a",
    )
    other = modalities.audio.Sample.decode(
        write(tmp_path / "short.wav", values[:4_000] * 0.5).read_bytes(),
        source="short",
        label="b",
    )
    preparer = modality_lifecycle.Audio(
        modalities.audio.Dataset((sample, other))
    )
    batch = preparer.features((sample, other))
    model = linear_model.LogisticRegression()
    model.fit(batch.values, [0, 1])
    bound = 16 * 1024 * 1024

    tracemalloc.start()
    try:
        probe = preparer.perturb(sample, model, 0.0)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    assert probe["status"] == "available"
    assert peak < bound, f"peak {peak} bytes over the bound {bound}"


def outcome(run):
    """What a media run must leave, independent of the machine.

    The artifacts of the shared stages, the cohort split, the evaluation's
    baseline and cross-validated interval, the held-out coverage read once
    from the locked rows, and the custody log of the one opening.
    """
    evaluation = json.loads(
        (run.run_dir / "evaluation.json").read_text(encoding="utf-8")
    )
    validation = json.loads(
        (run.run_dir / "validation_manifest.json").read_text(encoding="utf-8")
    )
    coverage = json.loads(
        (run.run_dir / "coverage_report.json").read_text(encoding="utf-8")
    )
    custody = run.run_dir.parent / "custody" / f"{run.run_id}.jsonl"
    return {
        "artifacts": sorted(
            path.name for path in run.run_dir.iterdir() if path.name in KEPT
        ),
        "strategy": validation["strategy"],
        "evaluation": sorted(key for key in evaluation if key in EVALUATED),
        "interval": sorted(evaluation["confidence"]),
        "coverage": coverage["rows_read"],
        "custody": [
            json.loads(line)["kind"]
            for line in custody.read_text(encoding="utf-8").splitlines()
        ],
    }


KEPT = {
    "complete.json",
    "coverage_report.json",
    "dataset_manifest.json",
    "dataset_profile.json",
    "evaluation.json",
    "explanations",
    "model",
    "validation_manifest.json",
}
"""Artifacts every run on the shared stages writes."""

EVALUATED = {"baseline", "confidence", "metrics", "opened", "task"}
"""Evaluation entries the golden pins: the baseline, the held-out interval,
the test metrics and the custody record of the one opening."""


def test_a_classification_run_takes_the_shared_stages(registered, tmp_path):
    manifest = collection(tmp_path)
    run = scmle.api.project.Project(
        data=str(manifest),
        target="label",
        modality="audio",
        models=support.optin(tasks.Task.MULTICLASS),
        time_budget=60,
        output_dir=str(tmp_path / "out"),
    ).run()

    found = outcome(run)
    golden = GOLDENS / "engine.json"
    if WRITE:
        golden.write_text(json.dumps(found, indent=1) + "\n", encoding="utf-8")
    assert found == json.loads(golden.read_text(encoding="utf-8"))
    assert run.evaluation["opened"]
    assert rows.KEY not in run.profile.feature_plan.numeric

    rng = numpy.random.default_rng(7)
    fresh = [
        write(tmp_path / f"new{index}.wav", clip(rng, index))
        for index in range(3)
    ]
    predicted = run.predict(registered.describe(fresh))
    assert len(predicted) == 3
    assert set(predicted) <= {"low", "mid", "high"}


def test_a_regression_run_takes_the_shared_stages(registered, tmp_path):
    manifest = collection(tmp_path)
    frame = audio.AUDIO.read(specification(manifest))[0]
    listed = (
        pandas.read_csv(manifest).sort_values("path").reset_index(drop=True)
    )
    listed["level"] = frame["audio_loudness"] + numpy.random.default_rng(
        8
    ).normal(0.0, 0.5, len(frame))
    listed.drop(columns=["label"]).to_csv(manifest, index=False)

    run = scmle.api.project.Project(
        data=str(manifest),
        target="level",
        modality="audio",
        models=support.optin(tasks.Task.REGRESSION),
        time_budget=60,
        output_dir=str(tmp_path / "out"),
    ).run()

    assert run.evaluation["task"] == tasks.Task.REGRESSION.value
    assert run.evaluation["metrics"]


def test_a_flac_clip_is_described_through_soundfile(tmp_path):
    soundfile = pytest.importorskip(
        "soundfile", reason="the optional soundfile library is not installed"
    )
    values = clip(numpy.random.default_rng(12), 0)
    soundfile.write(tmp_path / "clip.flac", values, RATE)
    write(tmp_path / "clip.wav", values)

    flac = spectrum.describe(tmp_path / "clip.flac")
    wave = spectrum.describe(tmp_path / "clip.wav")

    assert flac.features["audio_duration"] == wave.features["audio_duration"]
    assert flac.features["audio_centroid"] == pytest.approx(
        wave.features["audio_centroid"], rel=0.01
    )
