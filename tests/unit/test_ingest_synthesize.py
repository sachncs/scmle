"""As-of view and synthesis: point-in-time correctness and bookkeeping."""

import ast
import datetime
import fractions
import pathlib

import numpy
import pandas
import pytest

from benchmarks.qualification import ingest_checks
from scmle.contracts import qualification
from scmle.data.admissibility import provenance
from scmle.data.ingest import asof_view
from scmle.data.ingest import synthesize
from scmle.data.ledger import ledger as ledger_module
from scmle.data.ledger import online_fdr

ROOT = pathlib.Path(__file__).resolve().parents[2]


def small_view(rule=None, permissive=False):
    frame = pandas.DataFrame(
        {
            "cid": [1, 1, 1, 2],
            "ev": [1.0, 5.0, 9.0, 3.0],
            "rec": [1.5, 20.0, 9.5, 3.5],
            "val": [10.0, 20.0, 30.0, 40.0],
        }
    )
    rule = rule or asof_view.TimeRule(asof_view.TRANSACTION, "rec")
    return asof_view.AsOfView(
        frame, "cid", rule, "ev", "events", permissive=permissive
    )


def asked(cutoff, key=1):
    return pandas.DataFrame(
        {
            "cid": [key],
            asof_view.TARGET_COLUMN: [0],
            asof_view.CUTOFF_COLUMN: [cutoff],
        }
    )


def test_late_arrival_is_excluded_by_recorded_time():
    seen = small_view().visible(asked(10.0))
    # event 5.0 is before the cutoff but was recorded at 20.0
    assert seen["val"].tolist() == [10.0, 30.0]


def test_event_delay_mode_uses_the_declared_delay():
    rule = asof_view.TimeRule(asof_view.EVENT_DELAY, None, 2.0)
    seen = small_view(rule).visible(asked(10.0))
    assert seen["val"].tolist() == [10.0, 20.0]  # 9.0 + 2.0 > 10
    assert small_view(rule).tier is qualification.Tier.UNVERIFIED


def test_non_versioned_table_is_refused_unless_permissive():
    rule = asof_view.TimeRule(asof_view.NON_VERSIONED)
    frame = pandas.DataFrame({"cid": [1], "segment": ["gold"]})
    view = asof_view.AsOfView(frame, "cid", rule, name="dim")
    with pytest.raises(asof_view.NoAsOfRuleError):
        view.visible(asked(10.0))
    open_view = asof_view.AsOfView(
        frame, "cid", rule, name="dim", permissive=True
    )
    assert open_view.visible(asked(10.0))["segment"].tolist() == ["gold"]
    assert open_view.tier is qualification.Tier.UNVERIFIED


def test_time_rule_validation_and_tiers():
    with pytest.raises(ValueError):
        asof_view.TimeRule(asof_view.TRANSACTION)
    with pytest.raises(ValueError):
        asof_view.TimeRule(asof_view.EVENT_DELAY)
    with pytest.raises(ValueError):
        asof_view.TimeRule("nope")
    with pytest.raises(ValueError):
        asof_view.TimeRule(asof_view.EVENT_DELAY, None, -1.0)
    good = provenance.LogCheck(100, 0, 0.01)
    bad = provenance.LogCheck(100, 50, 0.01)
    assert (
        asof_view.TimeRule(
            asof_view.TRANSACTION, "rec", None, None, good
        ).tier()
        is qualification.Tier.CONFIRMED
    )
    assert (
        asof_view.TimeRule(asof_view.TRANSACTION, "rec", None, None, bad).tier()
        is qualification.Tier.UNVERIFIED
    )
    assert (
        asof_view.TimeRule(asof_view.EVENT_DELAY, None, 1.0, None, good).tier()
        is qualification.Tier.UNVERIFIED
    )


def test_reserved_and_missing_columns_are_rejected():
    frame = pandas.DataFrame({"cid": [1], "ev": [1.0], "event_days": [3]})
    with pytest.raises(ValueError):
        asof_view.AsOfView(
            frame,
            "cid",
            asof_view.TimeRule(asof_view.EVENT_DELAY, None, 0),
            "ev",
        )
    with pytest.raises(ValueError):
        asof_view.AsOfView(
            pandas.DataFrame({"cid": [1]}),
            "cid",
            asof_view.TimeRule(asof_view.EVENT_DELAY, None, 0),
            "ev",
        )


def test_datetime_columns_become_days_and_null_keys_never_match():
    frame = pandas.DataFrame(
        {
            "cid": [1, None],
            "ev": pandas.to_datetime(["2024-01-01", "2024-01-02"], utc=True),
        }
    )
    view = asof_view.AsOfView(
        frame, "cid", asof_view.TimeRule(asof_view.EVENT_DELAY, None, 0.0), "ev"
    )
    seen = view.visible(asked(20_000.0, key=1))
    assert len(seen) == 1
    assert seen[asof_view.EVENT_COLUMN].iloc[0] == pytest.approx(19_723.0)
    assert asof_view.to_days(pandas.Series(["2024-01-01"]))[0] == pytest.approx(
        19_723.0
    )


def test_scope_boundaries_match_the_reference_on_a_small_table():
    source = synthesize.Source(small_view(), "cid")
    targets = pandas.DataFrame({"cid": [1], "moment": [11.0]})
    spec = synthesize.FeatureSpec("f", "aggregate", 0, "sum", "val", None)
    got = synthesize.evaluate(spec, targets, "moment", [source], [], 1.0)
    assert got.tolist() == [40.0]  # cutoff 10: values 10 and 30


@pytest.mark.parametrize("seed", range(150))
def test_differential_against_the_naive_reference(seed):
    assert ingest_checks.differential_equal(seed), seed


@pytest.mark.parametrize("seed", range(150))
def test_metamorphic_rewriting_unknown_rows_changes_nothing(seed):
    assert ingest_checks.metamorphic_unchanged(seed), seed


def test_late_arrival_worlds_leak_for_the_baseline_and_not_for_the_view():
    results = [ingest_checks.late_leaks(seed) for seed in range(40)]
    assert all(r.view == 0 for r in results)
    assert sum(r.baseline > 0 for r in results) >= 35


def test_synthesis_declares_family_size_and_drops_with_reasons():
    rng = numpy.random.default_rng(0)
    targets = pandas.DataFrame(
        {"cid": numpy.arange(30), "moment": numpy.full(30, 80.0)}
    )
    events = pandas.DataFrame(
        {
            "cid": rng.integers(0, 30, 400),
            "ev": rng.uniform(0, 100, 400),
            "amt": rng.normal(size=400),
            "flat": numpy.ones(400),
        }
    )
    view = asof_view.AsOfView(
        events,
        "cid",
        asof_view.TimeRule(asof_view.EVENT_DELAY, None, 0.0),
        "ev",
        "events",
    )
    found = synthesize.synthesize(
        targets, "moment", [synthesize.Source(view, "cid")]
    )
    reasons = {m["dropped"] for m in found.manifest if not m["kept"]}
    assert "constant" in reasons
    assert any(r.startswith("duplicate of") for r in reasons)
    assert found.declared_family_size == found.frame.shape[1]
    assert found.declared_family_size == len(found.kept())
    capped = synthesize.synthesize(
        targets, "moment", [synthesize.Source(view, "cid")], max_features=5
    )
    assert sum(m["dropped"] == "cap" for m in capped.manifest) == (
        len(capped.manifest) - 5
    )
    assert capped.declared_family_size <= 5


def test_mutable_dimension_is_excluded_by_default_and_unverified_when_opened():
    targets = pandas.DataFrame({"cid": [1, 2, 3], "moment": [10.0, 10.0, 10.0]})
    dim = pandas.DataFrame({"cid": [1, 2, 3], "segment": ["a", "b", "a"]})
    rule = asof_view.TimeRule(asof_view.NON_VERSIONED)
    closed = asof_view.AsOfView(dim, "cid", rule, name="dim")
    out = synthesize.synthesize(
        targets, "moment", [synthesize.Source(closed, "cid", synthesize.PARENT)]
    )
    assert out.declared_family_size == 0
    assert out.manifest[0]["dropped"].startswith("no as-of rule")
    opened = asof_view.AsOfView(dim, "cid", rule, name="dim", permissive=True)
    sources = [synthesize.Source(opened, "cid", synthesize.PARENT)]
    permissive = synthesize.synthesize(targets, "moment", sources)
    assert permissive.manifest[0]["tier"] == "unverified"
    assert permissive.declared_family_size == 1
    strict = synthesize.synthesize(targets, "moment", sources, strict=True)
    assert strict.declared_family_size == 0
    assert strict.manifest[0]["dropped"] == "unverified (strict)"
    table = synthesize.feature_provenance(permissive, 0.0, [opened])
    assert table.fields[0].tier() is qualification.Tier.UNVERIFIED


def attested_rule(kind=asof_view.EVENT_DELAY):
    note = provenance.Attestation(
        "A. Owner",
        "data owner",
        "recorded within 2 days",
        "ticket 1",
        "2024-01-01",
        "scope",
        "2099-01-01",
    )
    if kind == asof_view.EVENT_DELAY:
        return asof_view.TimeRule(kind, None, 2.0, note)
    return asof_view.TimeRule(
        kind, "rec", 2.0, note, provenance.LogCheck(100, 0, 0.01)
    )


def test_feature_tier_is_the_weakest_source_tier():
    targets = pandas.DataFrame({"cid": [1, 2, 1], "moment": [10.0, 10.0, 2.0]})
    strong = small_view(attested_rule(asof_view.TRANSACTION))
    weak = asof_view.AsOfView(
        pandas.DataFrame({"cid": [1, 2], "ev": [1.0, 1.0], "x": [1.0, 2.0]}),
        "cid",
        asof_view.TimeRule(asof_view.EVENT_DELAY, None, 0.0),
        "ev",
        "weak",
    )
    out = synthesize.synthesize(
        targets,
        "moment",
        [synthesize.Source(strong, "cid"), synthesize.Source(weak, "cid")],
    )
    tiers = {m["tables"][0]: m["tier"] for m in out.kept()}
    assert tiers["events"] == "confirmed"
    assert tiers["weak"] == "unverified"
    table = synthesize.feature_provenance(out, 0.0, [strong, weak])
    by_name = table.tiers()
    kept_strong = [m["name"] for m in out.kept() if m["tables"] == ["events"]]
    assert by_name[kept_strong[0]] is qualification.Tier.CONFIRMED
    attested = small_view(attested_rule())
    solo = synthesize.synthesize(
        targets, "moment", [synthesize.Source(attested, "cid")]
    )
    solo_table = synthesize.feature_provenance(solo, 1.0, [attested])
    assert set(solo_table.tiers().values()) == {qualification.Tier.ATTESTED}
    assert solo_table.fields[0].as_of.lead_seconds == 86_400.0


def test_nested_features_are_depth_two_and_registered():
    schema = ingest_checks.random_schema(11)
    source, items = ingest_checks.views(schema)
    nest = synthesize.Nest(source, items, "oid")
    out = synthesize.synthesize(
        schema.targets, "moment", [source], [nest], schema.lag
    )
    assert any(m["kind"] == "nested" for m in out.manifest)


def test_multiplicity_matches_registered_hypotheses_and_mismatch_fails():
    targets = pandas.DataFrame(
        {"cid": numpy.arange(5), "moment": numpy.full(5, 50.0)}
    )
    rng = numpy.random.default_rng(1)
    events = pandas.DataFrame(
        {
            "cid": rng.integers(0, 5, 60),
            "ev": rng.uniform(0, 40, 60),
            "amt": rng.normal(size=60),
        }
    )
    view = asof_view.AsOfView(
        events,
        "cid",
        asof_view.TimeRule(asof_view.EVENT_DELAY, None, 0.0),
        "ev",
        "events",
    )
    out = synthesize.synthesize(
        targets, "moment", [synthesize.Source(view, "cid")]
    )
    ledger = ledger_module.Ledger.create(
        None,
        charter_hash="0" * 64,
        config=online_fdr.RuleConfig("lord++", fractions.Fraction(1, 20)),
        alpha_confirm=fractions.Fraction(1, 20),
        seed=0,
    )
    registered = synthesize.register_family(ledger, out)
    assert len(registered) == out.declared_family_size
    short = synthesize.FeatureSet(
        out.frame, out.manifest, out.declared_family_size + 1
    )
    with pytest.raises(synthesize.MultiplicityError):
        synthesize.check_multiplicity(short.declared_family_size, ledger)


def imported_modules(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            names.add(base)
            names.update(f"{base}.{a.name}" for a in node.names)
    return names


def test_synthesize_has_no_raw_read_path():
    names = imported_modules(ROOT / "scmle/data/ingest/synthesize.py")
    forbidden = (
        "scmle.data.readers",
        "scmle.data.ingest.profile",
        "sqlite3",
    )
    assert not [n for n in names if n.startswith(forbidden)]
    source = (ROOT / "scmle/data/ingest/synthesize.py").read_text()
    assert "read_csv" not in source and "read_sql" not in source
    assert "scmle.data.readers" in "\n".join(
        imported_modules(ROOT / "scmle/data/ingest/profile.py")
    )


def test_today_default_does_not_change_tier_of_unexpired_attestation():
    rule = attested_rule()
    assert rule.tier(datetime.date(2025, 1, 1)) is qualification.Tier.ATTESTED
    assert rule.tier(datetime.date(2100, 1, 1)) is qualification.Tier.UNVERIFIED


def test_evidence_and_expressions_are_stated_exactly():
    view = small_view()
    assert view.evidence() == {
        "table": "events",
        "mode": "transaction",
        "known_column": "rec",
        "delta_rec_days": None,
        "tier": "unverified",
    }
    aggregate = synthesize.FeatureSpec("f", "aggregate", 0, "sum", "val", 7.0)
    assert aggregate.expression() == "sum(val) over 7d"
    free = synthesize.FeatureSpec("f", "aggregate", 0, "count", None, None)
    assert free.expression() == "count(*) over all"
    nested = synthesize.FeatureSpec(
        "f", "nested", 0, "mean", "q", 30.0, nest=0, inner="sum"
    )
    assert nested.expression() == "mean(per-row sum(q) in 30d)"
    assert synthesize.window_label(None) == "all"
    assert synthesize.window_label(90.0) == "90d"


def test_weakest_tier_and_unknown_inputs():
    tiers = [qualification.Tier.CONFIRMED, qualification.Tier.ATTESTED]
    assert asof_view.weakest(tiers) is qualification.Tier.ATTESTED
    assert asof_view.weakest([]) is qualification.Tier.UNVERIFIED
    with pytest.raises(ValueError):
        synthesize.Source(small_view(), "cid", role="sibling")
    with pytest.raises(ValueError):
        synthesize.synthesize(
            pandas.DataFrame({"cid": [1], "moment": [1.0]}),
            "moment",
            [synthesize.Source(small_view(), "cid")],
            lag_days=-1.0,
        )
    with pytest.raises(ValueError):
        synthesize.reduce_rows(small_view().reader(), "median", "val", None, 1)
