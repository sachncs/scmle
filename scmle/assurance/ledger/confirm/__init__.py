"""The single confirmation of the exploration survivors on `C`.

Order of events. The guarantee ``FDR <= alpha_confirm`` comes only
from this stage:

1. The survivors, selected on `X` only, are frozen in the ledger with a hash.
2. If the negative-control gate fails, there is nothing to promote and `C` is
   not opened. With no survivors, `C` is not opened either.
3. The selection is written to the custody log, the release is bound to the
   family's hash and the ledger head (`Locker.commit`), then `C` is opened
   exactly once through the custody API. A set already bound to another
   claim set or ledger state is refused by the vault, so a ledger edited
   after the binding cannot unlock `C`.
4. Each survivor's statistic runs on `C`; the family of survivors is adjusted
   by BH, BY (default) or Holm at ``alpha_confirm``; claims are written.

A second call raises `LeakageError` and is logged. This module takes the
sealed set as an argument; it imports nothing from custody, and no other
module of the ledger package can reach `C` (an import-graph test checks it).

Second stage (optional, pre-registered). Between the negative control and the
opening of `C` the caller may hand over a `Withdrawal`: survivors that a rule
computed from `X` alone (the stability filter) removes from the family that
goes to `C`. The ledger is untouched: it still freezes and hashes the full
survivor set, and the withdrawal is its own accounting, written to the custody
log (``selection_frozen``, with ``family_sha256``) before `C` opens and to the
confirmation row and ``confirmed_claims.json`` after (``stage_two``). Withdrawn
survivors are reported at tier `unverified` and are never tested on `C`.

What is controlled, and under what assumption (`STAGE_TWO_STATEMENT`). Let
``S`` be the frozen survivors and ``F`` the subset not withdrawn. If ``F`` is a
function of `X`, the charter and the seed only, and `C` is independent of `X`
(the existing assumption), then conditionally on ``X`` the p-values on `C` of
the true nulls in ``F`` keep their null distribution, so the rule's guarantee
holds for ``F``: FDR of the confirmed claims at most ``alpha_confirm`` for BY
(any dependence) and for BH under independence or positive dependence on `C`,
FWER at most ``alpha_confirm`` for Holm and Bonferroni. The bound is on the
claims actually made, which are a subset of ``F``; it averages over the
selection, so the same level holds unconditionally. The withdrawal spends no
alpha and adds none. What is not claimed: that the withdrawal raises power
(it can only remove hypotheses, a true effect withdrawn is lost), that a
withdrawn finding is false, or any guarantee if ``F`` was computed from rows
of `C` or `F`. The module cannot see how the caller computed ``F``; it checks
only that the withdrawn ids are survivors and that the family hash is
recorded before `C` opens. `tests/unit/test_ledger_withdrawal.py` simulates
the error rate with an `X`-only withdrawal rule that is correlated with the
signal, and checks the order of events.

Registered claims. A caller may hand over a `Docket`: claims that are not
single-feature statistics (the baseline comparison, the primary causal
hypothesis). Each is registered in the ledger by the hash of its
registration before the survivors are frozen, the sealed set is bound to the
survivor family and the registrations, and the *same* released frame is given
to the runner after the bivariate family was tested. The runner never sees `C`
twice: there is one opening. Their gates differ from the family's: a missing
or failed negative control and an empty family stop the bivariate promotion,
not the registered claims, which carry their own gates. The confirmation row
records the status of each claim and the hash of every report file the runner
wrote, so the files are bound to the ledger head. When `C` does not exist or
the opening fails, each claim is recorded as not evaluated and stays
unverified.

Failure policy. A statistical failure to reject is a result, not a failure.
If the confirmation run *crashes* after `C` was opened, `C` is spent and the
charter's policy applies: ``unconfirmed`` (default) keeps every claim at tier
`unverified` and says so. ``thresholdout`` keeps that rule for the family and
also answers adaptive queries on the same single opening of `C` through the
reusable holdout of Dwork et al. (2015): it needs a reuse session
(`thresholdout.Session`), registered in the ledger like any other claim and
bound to the release before `C` opens (see `opening` and `thresholdout`).
`match` refuses the policy without a session, and a session under another
policy, with a `ConfigError` before anything is frozen or opened.

Assumptions (stated on every claim): the rows of `C` are independent of the
rows of `X` (grouped split by entity, embargo for temporal data); `C` is read
once; the statistics are valid under their null on `C` (the Spearman and
Kruskal tests assume independent rows within `C`, which the user must judge
for grouped or serial data); the family is exactly the frozen survivors.
"""

# `budget` stays reachable as `confirm.budget`, as it was from the module.
from scmle.assurance.ledger import budget
from scmle.assurance.ledger.confirm.evaluation import account
from scmle.assurance.ledger.confirm.evaluation import assumptions
from scmle.assurance.ledger.confirm.evaluation import block
from scmle.assurance.ledger.confirm.evaluation import disclaim
from scmle.assurance.ledger.confirm.evaluation import encode
from scmle.assurance.ledger.confirm.evaluation import statement
from scmle.assurance.ledger.confirm.evaluation import verify
from scmle.assurance.ledger.confirm.evaluation import withdraw
from scmle.assurance.ledger.confirm.opening import confirm
from scmle.assurance.ledger.confirm.opening import deny
from scmle.assurance.ledger.confirm.opening import enforce
from scmle.assurance.ledger.confirm.opening import finish
from scmle.assurance.ledger.confirm.opening import match
from scmle.assurance.ledger.confirm.opening import phrase
from scmle.assurance.ledger.confirm.opening import preregister
from scmle.assurance.ledger.confirm.opening import tip
from scmle.assurance.ledger.confirm.opening import unseal
from scmle.assurance.ledger.confirm.records import ACTOR
from scmle.assurance.ledger.confirm.records import ALL_WITHDRAWN
from scmle.assurance.ledger.confirm.records import CLAIMS_FILE
from scmle.assurance.ledger.confirm.records import Commitment
from scmle.assurance.ledger.confirm.records import Confirmation
from scmle.assurance.ledger.confirm.records import CONFIRMED
from scmle.assurance.ledger.confirm.records import Docket
from scmle.assurance.ledger.confirm.records import FAILED
from scmle.assurance.ledger.confirm.records import FLOOR_NOT_EXCEEDED
from scmle.assurance.ledger.confirm.records import Log
from scmle.assurance.ledger.confirm.records import LOGGER
from scmle.assurance.ledger.confirm.records import merge
from scmle.assurance.ledger.confirm.records import NO_CONFIRMATION_SET
from scmle.assurance.ledger.confirm.records import NO_CONTROL
from scmle.assurance.ledger.confirm.records import NO_SURVIVORS
from scmle.assurance.ledger.confirm.records import NOT_EVALUATED
from scmle.assurance.ledger.confirm.records import Output
from scmle.assurance.ledger.confirm.records import PIPELINE_FAILURE
from scmle.assurance.ledger.confirm.records import POLICIES
from scmle.assurance.ledger.confirm.records import Source
from scmle.assurance.ledger.confirm.records import STAGE_TWO_STATEMENT
from scmle.assurance.ledger.confirm.records import THRESHOLDOUT
from scmle.assurance.ledger.confirm.records import UNCONFIRMED
from scmle.assurance.ledger.confirm.records import Withdrawal

__all__ = [
    "ACTOR",
    "ALL_WITHDRAWN",
    "CLAIMS_FILE",
    "CONFIRMED",
    "FAILED",
    "FLOOR_NOT_EXCEEDED",
    "LOGGER",
    "NOT_EVALUATED",
    "NO_CONFIRMATION_SET",
    "NO_CONTROL",
    "NO_SURVIVORS",
    "PIPELINE_FAILURE",
    "POLICIES",
    "STAGE_TWO_STATEMENT",
    "THRESHOLDOUT",
    "UNCONFIRMED",
    "Commitment",
    "Confirmation",
    "Docket",
    "Log",
    "Output",
    "Source",
    "Withdrawal",
    "account",
    "assumptions",
    "block",
    "budget",
    "confirm",
    "deny",
    "disclaim",
    "encode",
    "enforce",
    "finish",
    "match",
    "merge",
    "phrase",
    "preregister",
    "statement",
    "tip",
    "unseal",
    "verify",
    "withdraw",
]
