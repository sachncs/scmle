"""The append-only, hash-chained hypothesis ledger.

Every comparison made on the exploration set `X` is a logged hypothesis. The
file ``hypothesis_ledger.jsonl`` holds one JSON object per line; each stores
the hash of the previous line, so editing, removing from the middle or
reordering any line breaks the chain from there on. Rows are:

``open``
    Charter hash, the online rule and its parameters, its guarantee, seeds.
``register``
    A hypothesis specification, written *before* its data is touched.
``result``
    A `Check`, the e-value used, the level assigned, the decision and the
    wealth before and after. Registration precedes evaluation: a result for an
    unregistered or already tested hypothesis is refused in code and, on
    load, rejected as a broken ledger.
``diagnostic``
    A `Check` from a detector run on `X` (a leakage suspicion).
    It spends no wealth and never enters the survivors; it is chained so a
    suspicion that was raised cannot be removed afterwards.
``control``
    The negative-control summary (optional, before the survivors are frozen).
``claim``
    A registered claim that is not a single-feature statistic (the baseline
    comparison, the primary causal hypothesis), written before
    the survivors are frozen with the hash of its registration. It spends no
    wealth, never enters the survivors and is evaluated once, on `C`, inside
    the single opening; the confirmation row records its outcome.
``survivors``
    The rejected hypotheses, frozen with a hash before `C` is unlocked.
``confirmation``
    The outcome of the single confirmation, with the claims file hash, the
    status of every registered claim and the hashes of the report files the
    registered claims produced (``attachments``), so they are bound to the
    ledger head.

Verification checks the chain and then replays every result through the
online rule from the stored evidence, comparing level, decision and wealth
exactly (the rule uses exact fractions, so replay is bit-for-bit identical on
every platform).

Limits. A hash chain shows that nothing was changed, removed from
the middle or reordered since it was written. It cannot show that the *last*
rows were never deleted: a prefix of a valid chain is valid. Pass the head
hash recorded elsewhere (the custody log and the claims file store it) to
`Ledger.load` to detect tail truncation. Someone able to rewrite the whole
file and every copy of the head can forge a ledger; this is tamper-evidence on
one machine, not protection from an adversary.
"""

from scmle.assurance.ledger.ledger.journal import Ledger
from scmle.assurance.ledger.ledger.records import CLAIM
from scmle.assurance.ledger.ledger.records import CONFIRMATION
from scmle.assurance.ledger.ledger.records import CONTROL
from scmle.assurance.ledger.ledger.records import DIAGNOSTIC
from scmle.assurance.ledger.ledger.records import digest
from scmle.assurance.ledger.ledger.records import FILE_NAME
from scmle.assurance.ledger.ledger.records import fingerprint
from scmle.assurance.ledger.ledger.records import GENESIS
from scmle.assurance.ledger.ledger.records import Hypothesis
from scmle.assurance.ledger.ledger.records import LEDGER_VERSION
from scmle.assurance.ledger.ledger.records import LedgerError
from scmle.assurance.ledger.ledger.records import NOT_TESTED
from scmle.assurance.ledger.ledger.records import OPEN
from scmle.assurance.ledger.ledger.records import Outcome
from scmle.assurance.ledger.ledger.records import REGISTER
from scmle.assurance.ledger.ledger.records import REJECT
from scmle.assurance.ledger.ledger.records import RESULT
from scmle.assurance.ledger.ledger.records import RETAIN
from scmle.assurance.ledger.ledger.records import Selection
from scmle.assurance.ledger.ledger.records import SURVIVORS
from scmle.assurance.ledger.ledger.records import Test

__all__ = [
    "CLAIM",
    "CONFIRMATION",
    "CONTROL",
    "DIAGNOSTIC",
    "FILE_NAME",
    "GENESIS",
    "LEDGER_VERSION",
    "NOT_TESTED",
    "OPEN",
    "REGISTER",
    "REJECT",
    "RESULT",
    "RETAIN",
    "SURVIVORS",
    "Hypothesis",
    "Ledger",
    "LedgerError",
    "Outcome",
    "Selection",
    "Test",
    "digest",
    "fingerprint",
]
