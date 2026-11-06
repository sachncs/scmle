"""The extract is reconciled to the totals the charter declared (F9)."""

import pandas as pd
import pytest

from scmle.contracts import errors
from scmle.data.charter import ingest
from scmle.data.ingest import declared


def orders():
    return pd.DataFrame({"order_id": range(1, 101), "amount": [50.0] * 100})


def spec(**changes):
    values = {
        "rows": 100,
        "sums": (ingest.ColumnTotal("amount", 5000.0),),
        "checksum_column": "order_id",
        "checksum": declared.key_checksum(orders()["order_id"]),
    }
    values.update(changes)
    return ingest.ReconcileSpec((ingest.TableTotals("orders", **values),))


def test_matching_totals_pass_and_are_reported():
    report = declared.reconcile({"orders": orders()}, spec(), 0.0)
    assert report["passed"] is True
    assert {c["name"] for c in report["checks"]} == {
        "orders.row_count",
        "orders.sum.amount",
        "orders.checksum.order_id",
    }
    declared.require(report)


def test_a_dropped_row_stops_the_run_with_the_quantity_named():
    frame = orders().iloc[:-1]
    report = declared.reconcile({"orders": frame}, spec(), 0.0)
    failed = {c["name"] for c in report["checks"] if not c["passed"]}
    assert failed == {
        "orders.row_count",
        "orders.sum.amount",
        "orders.checksum.order_id",
    }
    with pytest.raises(errors.BlockedError, match=r"orders\.row_count"):
        declared.require(report)


def test_a_sum_within_the_tolerance_passes_and_beyond_it_fails():
    frame = orders()
    frame.loc[0, "amount"] = 60.0
    block = spec(rows=None, checksum_column=None, checksum=None)
    assert not declared.reconcile({"orders": frame}, block, 0.0)["passed"]
    assert not declared.reconcile({"orders": frame}, block, 0.001)["passed"]
    assert declared.reconcile({"orders": frame}, block, 0.01)["passed"]


def test_a_changed_key_fails_the_checksum_only():
    frame = orders()
    frame.loc[0, "order_id"] = 999
    block = spec(rows=100, sums=())
    report = declared.reconcile({"orders": frame}, block, 0.0)
    assert [c["name"] for c in report["checks"] if not c["passed"]] == [
        "orders.checksum.order_id"
    ]


def test_the_checksum_ignores_row_order():
    keys = orders()["order_id"]
    assert declared.key_checksum(keys) == declared.key_checksum(keys[::-1])


def test_an_unknown_view_or_column_is_a_config_error():
    with pytest.raises(errors.ConfigError, match="view"):
        declared.reconcile({}, spec(), 0.0)
    with pytest.raises(errors.ConfigError, match="column"):
        declared.reconcile(
            {"orders": orders().drop(columns=["amount"])}, spec(), 0.0
        )
