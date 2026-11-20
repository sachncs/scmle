"""Continuous evaluation on top of the closed loop.

What this adds to `scmle.governance.loop.Loop`, and what it leaves alone:

* **Feeding.** Delayed labels and observed business outcomes become losses in
  [0, 1] and go to `Loop.mature`, for the model in service and for each
  challenger on the same cases. Raw labels and outcomes are kept.
* **Retraining policy.** `Policy` says when retraining is *recommended*
  (the default) or *triggered*. Either way the question reaches a person
  through `Loop.escalate`, the certified path. The loop's own alarm still does
  not gate retraining (`alarm_gates_retrain` stays False because its
  detection-delay gates missed their floors); nothing here sets it or claims
  otherwise. A trigger only trains a *challenger*: the challenger still has to
  win the loop's e-process, pass the gate and be approved before it serves.
* **History.** Dataset versions (by content digest), training runs, proposals,
  comparisons, replacements and post-promotion checks are appended to a
  hash-chained file, so what a model was trained on can be recovered.
* **Replacement.** `replace` refuses unless the loop recorded a promotion
  proposal for the challenger, the promotion gate admits the build (signed
  bundle, independent approval, scan, approver, staging record for
  production) and the release registry confirms the runtime.
* **After promotion.** `reassess` compares selected operational
  and business metrics before and after with bootstrap intervals and says
  whether the model helped, with the uncertainty, or that it cannot tell.

The package splits the work by concern: `records` (dataset versions and the
training history), `policies` (the retraining policy and its decision),
`promotion` (period and holdback comparisons), `feeding` (labels and outcomes
into the loop), `retraining` (reviews and challengers) and `evaluation`
(`Evaluation` and `Replacement`).
"""

from scmle.operations.continuous.evaluation import Evaluation
from scmle.operations.continuous.evaluation import Replacement
from scmle.operations.continuous.policies import DATA_FAULTS
from scmle.operations.continuous.policies import decide
from scmle.operations.continuous.policies import Decision
from scmle.operations.continuous.policies import DEFAULT_IF_UNANSWERED
from scmle.operations.continuous.policies import MODES
from scmle.operations.continuous.policies import Policy
from scmle.operations.continuous.policies import RECOMMEND
from scmle.operations.continuous.policies import RETRAIN_QUESTION
from scmle.operations.continuous.policies import TRIGGER
from scmle.operations.continuous.promotion import check
from scmle.operations.continuous.promotion import contrast
from scmle.operations.continuous.promotion import Indicator
from scmle.operations.continuous.promotion import reassess
from scmle.operations.continuous.promotion import reserve
from scmle.operations.continuous.records import DATASET_DIRECTORY
from scmle.operations.continuous.records import History
from scmle.operations.continuous.records import HISTORY_FILE
from scmle.operations.continuous.records import LABELS_FILE
from scmle.operations.continuous.records import load
from scmle.operations.continuous.records import serialize
from scmle.operations.continuous.records import stamp

__all__ = [
    "DATASET_DIRECTORY",
    "DATA_FAULTS",
    "DEFAULT_IF_UNANSWERED",
    "HISTORY_FILE",
    "LABELS_FILE",
    "MODES",
    "RECOMMEND",
    "RETRAIN_QUESTION",
    "TRIGGER",
    "Decision",
    "Evaluation",
    "History",
    "Indicator",
    "Policy",
    "Replacement",
    "check",
    "contrast",
    "decide",
    "load",
    "reassess",
    "reserve",
    "serialize",
    "stamp",
]
