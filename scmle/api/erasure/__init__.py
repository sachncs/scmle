"""Public API: forget data, retire models, delete runs, with signed receipts.

    from scmle.api import erasure

    subject = erasure.Subject("customers.csv", "customer_id", ["C-1042"])
    receipt = erasure.Forgetting(
        ["runs", "registry"], subject, "receipts", "request 2026-118",
        retrain=True,
    ).execute(key)
    erasure.Retirement("registry", "receipts", "drift", "churn", 3).execute(key)
    erasure.Deletion(["runs/run_001"], "receipts", "expired").execute(key)
    erasure.Receipt.read("receipts/forget-....json").verify(key)

An interrupted erasure resumes when the same request is executed again; its
receipt is written only after every step has finished. A retired model is
refused by `Bundle.load` with `RetiredError` (exit code 2 from the CLI).
"""

from scmle.api.erasure.erasures import Dataset
from scmle.api.erasure.erasures import Deletion
from scmle.api.erasure.erasures import Erasure
from scmle.api.erasure.erasures import Forgetting
from scmle.api.erasure.erasures import Retirement
from scmle.api.erasure.erasures import Subject
from scmle.api.erasure.holdings import CUSTODY
from scmle.api.erasure.receipts import Receipt
from scmle.api.erasure.steps import Step
from scmle.artifacts.bundle.tombstone import RetiredError

__all__ = [
    "CUSTODY",
    "Dataset",
    "Deletion",
    "Erasure",
    "Forgetting",
    "Receipt",
    "RetiredError",
    "Retirement",
    "Step",
    "Subject",
]
