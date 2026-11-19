"""The streaming transaction records' refusals and their schema loading."""

import pytest

from scmle.contracts import errors
from scmle.data.readers.streaming import progress
from scmle.data.readers.streaming import transaction

SOURCE = "orders"
BATCH = "batch_1"


def checkpoint(source: str = SOURCE, offset: int = 1) -> progress.Checkpoint:
    return progress.Checkpoint(source, {"snapshot": "a" * 64}, offset)


def voucher(**changes) -> transaction.Voucher:
    values = {
        "source": SOURCE,
        "batch": BATCH,
        "checkpoint": checkpoint(),
        "receipt": "r-1",
    }
    values.update(changes)
    return transaction.Voucher(**values)


def acknowledgement(**changes) -> transaction.Acknowledgement:
    values = {
        "source": SOURCE,
        "consumer": "group",
        "batch": BATCH,
        "checkpoint": checkpoint(),
    }
    values.update(changes)
    return transaction.Acknowledgement(**values)


def commitment() -> transaction.Commitment:
    return transaction.Commitment(SOURCE, BATCH, voucher(), acknowledgement())


def receipt(**changes) -> transaction.Receipt:
    values = {
        "identity": SOURCE,
        "batch": BATCH,
        "accepted": True,
        "duplicate": False,
        "checkpoint": checkpoint(),
    }
    values.update(changes)
    return transaction.Receipt(**values)


def test_a_blank_acknowledgement_source_is_refused() -> None:
    with pytest.raises(errors.InputError, match=r"^source must not be empty$"):
        acknowledgement(source=" ")


def test_a_blank_voucher_source_is_refused() -> None:
    with pytest.raises(errors.InputError, match=r"^source must not be empty$"):
        voucher(source=" ")


def test_a_blank_voucher_receipt_is_refused() -> None:
    with pytest.raises(errors.InputError, match=r"^receipt must not be empty$"):
        voucher(receipt=" ")


def test_a_newer_commitment_names_its_record_kind() -> None:
    record = commitment().to_dict()
    record["schema_version"] = 99
    with pytest.raises(
        errors.InputError, match=r"^Commitment record has schema version 99"
    ):
        transaction.Commitment.from_dict(record)


def test_a_newer_settlement_names_its_record_kind() -> None:
    record = transaction.Settlement(checkpoint()).to_dict()
    record["schema_version"] = 99
    with pytest.raises(
        errors.InputError, match=r"^Settlement record has schema version 99"
    ):
        transaction.Settlement.from_dict(record)


def test_a_settlement_without_commitments_loads_with_none() -> None:
    record = transaction.Settlement(checkpoint(), (BATCH,)).to_dict()
    del record["commitments"]
    loaded = transaction.Settlement.from_dict(record)
    assert loaded.committed == (BATCH,)
    assert loaded.commitments == ()


def test_a_receipt_checkpoint_from_another_source_is_refused() -> None:
    with pytest.raises(
        errors.InputError, match=r"^receipt and checkpoint sources differ$"
    ):
        receipt(checkpoint=checkpoint("other"))


def test_a_sink_receipt_for_another_batch_is_refused() -> None:
    with pytest.raises(
        errors.InputError, match=r"^receipt does not match the commit$"
    ):
        receipt(receipt=voucher(batch="batch_2"))


def test_a_matching_acknowledgement_is_accepted() -> None:
    found = receipt(receipt=voucher(), acknowledgement=acknowledgement())
    assert found.acknowledgement == acknowledgement()


def test_an_acknowledgement_for_another_batch_is_refused() -> None:
    with pytest.raises(
        errors.InputError,
        match=r"^acknowledgement does not match the commit$",
    ):
        receipt(acknowledgement=acknowledgement(batch="batch_2"))
