"""Reconciliation of an extract against pushed-down source totals."""

import pytest

from benchmarks.qualification import ingest_worlds
from scmle.data.detectors import certificates
from scmle.data.ingest import reconcile


def run(kind, seed=5, gate=certificates.GateStatus.DESCRIPTIVE):
    extract = ingest_worlds.reconcile_world(seed, kind)
    return reconcile.reconcile_extract(
        extract.db.source(),
        extract.assembled,
        "orders",
        ["amount"],
        [
            reconcile.JoinCheck(
                "orders-customers", extract.left_rows, len(extract.assembled)
            )
        ],
        gate=gate,
    )


def test_clean_extract_reconciles_with_orphans_present():
    report = run("clean")
    assert report.passed
    names = [r.item.name for r in report.result.results]
    assert names == [
        "orders.row_count",
        "orders.sum.amount",
        "fanout.orders-customers",
    ]


@pytest.mark.parametrize("kind", ["fanout", "dropped_rows", "torn_read"])
def test_planted_defects_are_unexplained(kind):
    report = run(kind)
    assert not report.passed
    assert not report.result.passed


def test_gate_decides_whether_the_check_blocks():
    descriptive = run("fanout")
    assert not reconcile.blocks(descriptive)
    gating = run("fanout", gate=certificates.GateStatus.GATING)
    assert reconcile.blocks(gating)


def test_zero_source_total_is_listed_not_passed():
    extract = ingest_worlds.reconcile_world(1, "clean")
    extract.db.execute("UPDATE orders SET amount = 0")
    report = reconcile.reconcile_extract(
        extract.db.source(), extract.assembled, "orders", ["amount"]
    )
    assert report.not_reconciled == ("orders.sum.amount",)
    assert not report.passed
    assert report.to_dict()["not_reconciled"] == ["orders.sum.amount"]


def test_feature_total_is_compared_with_a_source_aggregate():
    extract = ingest_worlds.reconcile_world(2, "clean")
    total = float(
        extract.db.frame("SELECT SUM(amount) AS s FROM orders")["s"][0]
    )
    good = reconcile.FeatureTotal(
        "amount_sum", total, "SELECT SUM(amount) FROM orders"
    )
    bad = reconcile.FeatureTotal(
        "amount_sum", total + 5, "SELECT SUM(amount) FROM orders"
    )
    for item, expected in ((good, True), (bad, False)):
        report = reconcile.reconcile_extract(
            extract.db.source(),
            extract.assembled,
            "orders",
            feature_totals=[item],
        )
        assert report.passed is expected
