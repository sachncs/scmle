"""The errors a run raises: leases, limits, deadlines, budgets, approvals."""

from scmle.contracts import errors
from scmle.core import retry


class BusyError(errors.ScmleError):
    """Another executor holds a valid lease on the run."""

    category = "run_in_flight"


class EvictionError(BusyError):
    """This executor no longer holds the run's lease and must stop writing."""

    category = "lease_lost"


class ConcurrencyLimitError(errors.ScmleError):
    """The workflow already has its maximum number of runs in flight."""

    category = "concurrency_limit"


class DeadlineError(errors.ScmleError):
    """A step passed its timeout."""

    category = "timeout"


class StepBudgetError(errors.ScmleError):
    """A step or the run passed its cost budget."""

    category = "budget"


class PreemptedError(retry.TransientError):
    """The step's compute was taken away; the step is retried."""

    category = "interrupted"


class ApprovalError(errors.ScmleError):
    """An approval was refused (unknown approver, bad signature, requester)."""

    category = "approval"
