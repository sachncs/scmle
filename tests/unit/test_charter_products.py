"""The charter stages return frozen products; the vault closes on every exit.

Each stage of a charter run (exploration, qualification, identification,
impounding) and the planning stage returns a frozen product that
`Product.apply` sets on the context the stages not yet converted read. The
stages run here on fake services: a real context over services with no run
directory, and fakes for the charter, the custody and the reports.
"""

import dataclasses
import types

import pandas
import pytest

from scmle.assurance.charter import contracts
from scmle.assurance.custody import sealed
import scmle.assurance.custody.pipeline
from scmle.contracts import errors
from scmle.contracts import tasks
from scmle.evaluation import metrics
from scmle.evaluation import selection
import scmle.models.registry
from scmle.pipeline import products
from scmle.pipeline import services
import scmle.pipeline.context
from scmle.pipeline.stages import exploration
from scmle.pipeline.stages import identification
from scmle.pipeline.stages import impounding
from scmle.pipeline.stages import planning
from scmle.pipeline.stages import profiling
from scmle.pipeline.stages import qualification
from scmle.runtime import limits


def board(**data):
    """A run context over fake services: no run directory, a charter path."""
    spec = tasks.Config(
        data=tasks.Dataset(uri="x.csv", target="y", **data),
        charter_path="charter.json",
    )
    given = services.Services(
        spec,
        scmle.models.registry.Catalog.stock(),
        metrics.CATALOG,
        selection.Averager(spec.selection),
        limits.Governor(spec.resources, 100.0),
    )
    return scmle.pipeline.context.Context.start(given)


class Vault(sealed.Vault):
    """A vault that never spawns a process and counts its closings."""

    made: list["Vault"] = []

    def __init__(self, log, process=True):
        del process
        super().__init__(log, process=False)
        self.closed = 0
        Vault.made.append(self)

    def close(self):
        self.closed += 1
        super().close()


def manifest():
    return types.SimpleNamespace(
        sha256="m",
        scope="confirmatory",
        sets=types.SimpleNamespace(
            exploration=(1, 2, 3), confirmation=(4,), embargo=()
        ),
    )


def outcome():
    return scmle.assurance.custody.pipeline.Outcome(
        "restricted", "view", manifest(), "feasible"
    )


@pytest.fixture
def custody(monkeypatch):
    """Fakes the charter and the carving; returns the vaults made."""
    Vault.made = []
    charter = types.SimpleNamespace(purpose="predictive")
    monkeypatch.setattr(
        impounding.sealing,
        "read_verified",
        lambda path: types.SimpleNamespace(charter=charter, signed=True),
    )
    monkeypatch.setattr(sealed, "Vault", Vault)
    monkeypatch.setattr(
        scmle.assurance.custody.pipeline,
        "sequester",
        lambda **given: outcome(),
    )
    monkeypatch.setattr(impounding.Impounding, "chronicle", lambda *given: None)
    return Vault.made


FRAME = pandas.DataFrame({"a": [1, 2, 3, 4], "y": [0, 1, 0, 1]})


def impound(context, split):
    """Sequesters as the profiling stage does: applies the custody."""
    made = profiling.ProfileStage().sequester(
        context, tasks.Task.BINARY, FRAME, split
    )
    made.apply(context)
    return made.outcome.split


def test_sequester_returns_the_custody_and_leaves_the_context(custody):
    context = board()
    made = profiling.ProfileStage().sequester(
        context, tasks.Task.BINARY, FRAME, "split"
    )
    assert isinstance(made, impounding.Custody)
    assert made.vault is custody[0]
    assert made.column is None
    assert context.custody is None
    assert context.custody_vault is None
    with pytest.raises(dataclasses.FrozenInstanceError):
        made.column = "g"  # type: ignore[misc]


def test_applied_custody_leaves_its_vault_to_the_context_close(custody):
    context = board()
    split = impound(context, "split")
    assert split == "restricted"
    assert context.custody_vault is custody[0]
    assert context.custody.manifest.sha256 == "m"
    assert custody[0].closed == 0
    context.close()
    assert custody[0].closed == 1


@pytest.mark.parametrize(
    "failure", [errors.InputError("broken"), KeyboardInterrupt()]
)
def test_the_vault_closes_when_the_carving_fails(custody, monkeypatch, failure):
    def sequester(**given):
        given["vault"].seal(
            "C",
            FRAME[["a"]],
            FRAME["y"],
            (0, 1, 2, 3),
        )
        raise failure

    monkeypatch.setattr(
        scmle.assurance.custody.pipeline, "sequester", sequester
    )
    context = board()
    with pytest.raises(type(failure)):
        impound(context, "split")
    assert [vault.closed for vault in custody] == [1]
    assert context.custody_vault is None


def test_the_vault_closes_when_its_records_fail(custody, monkeypatch):
    def chronicle(*given):
        raise OSError("disk full")

    monkeypatch.setattr(impounding.Impounding, "chronicle", chronicle)
    context = board()
    with pytest.raises(OSError, match="disk full"):
        impound(context, "split")
    assert [vault.closed for vault in custody] == [1]
    context.close()
    assert [vault.closed for vault in custody] == [1]


def test_no_vault_is_made_when_the_charter_is_refused(custody, monkeypatch):
    charter = types.SimpleNamespace(purpose=contracts.NEW_ENTITY)
    monkeypatch.setattr(
        impounding.sealing,
        "read_verified",
        lambda path: types.SimpleNamespace(charter=charter, signed=True),
    )
    with pytest.raises(errors.ConfigError, match="group_column"):
        impound(board(), "split")
    assert custody == []


def test_a_later_failure_leaves_the_vault_to_the_context_close(custody):
    context = board()
    impound(context, "x")
    try:
        with pytest.raises(errors.BlockedError):
            raise errors.BlockedError("a later stage stops the run")
    finally:
        context.close()
    assert [vault.closed for vault in custody] == [1]


def test_a_known_entity_charter_names_the_cluster_column(custody, monkeypatch):
    charter = types.SimpleNamespace(purpose=contracts.KNOWN_ENTITY_FUTURE)
    monkeypatch.setattr(
        impounding.sealing,
        "read_verified",
        lambda path: types.SimpleNamespace(charter=charter, signed=True),
    )
    context = board(group_column="g", time_column="t")
    split = types.SimpleNamespace(
        manifest=types.SimpleNamespace(strategy="temporal")
    )
    made = profiling.ProfileStage().sequester(
        context, tasks.Task.BINARY, FRAME, split
    )
    assert made.column == "g"
    assert context.cluster_column is None
    made.apply(context)
    assert context.cluster_column == "g"
    context.close()


def test_the_custody_digest_depends_on_the_manifest_only():
    first = impounding.Custody(outcome(), Vault(None), None)
    second = impounding.Custody(outcome(), Vault(None), None)
    other = impounding.Custody(outcome(), Vault(None), "g")
    path = None
    assert first.save(path) == second.save(path)
    assert first.save(path) != other.save(path)
    assert first.summary()["confirmation"] == 1


def escrow(coverage):
    return exploration.Escrow(
        explored="explored",
        task=tasks.Task.BINARY,
        charter=types.SimpleNamespace(charter_hash="h"),
        view="view",
        structure=(),
        graph=None,
        found="found",
        stage="stage",
        signed=True,
        baseline=None,
        causal=None,
        holds_for_coverage=coverage,
    )


def test_claims_set_the_pending_ledger_and_digest_their_charter():
    held = types.SimpleNamespace(pending_ledger=object())
    pending = escrow(coverage=True)
    exploration.Claims(pending).apply(held)
    assert held.pending_ledger is pending
    exploration.Claims().apply(held)
    assert held.pending_ledger is None
    assert exploration.Claims(pending).summary() == {
        "pending": True,
        "charter": "h",
        "coverage": True,
    }
    assert exploration.Claims().summary() == {"pending": False}


def test_without_custody_the_ledger_leaves_nothing_pending():
    context = board()
    context.pending_ledger = "stale"
    made = profiling.ProfileStage().ledger(context, tasks.Task.BINARY)
    assert made == exploration.Claims()
    made.apply(context)
    assert context.pending_ledger is None


def test_a_task_outside_the_ledger_is_skipped_with_nothing_pending():
    context = board()
    context.custody = "custody"
    context.custody_vault = "vault"
    made = profiling.ProfileStage().ledger(context, tasks.Task.SURVIVAL)
    assert made.pending is None


def test_defer_keeps_the_ledger_off_the_context():
    context = board()
    stage = profiling.ProfileStage()
    assert stage.defer(context, escrow(coverage=True))
    assert not stage.defer(context, escrow(coverage=False))
    assert context.pending_ledger is None


def found():
    return types.SimpleNamespace(
        fragile=["b"],
        stability_gate=types.SimpleNamespace(value="certified"),
        forward=types.SimpleNamespace(status="pass"),
    )


def test_assess_robustness_returns_the_verdicts(monkeypatch):
    report = qualification.scmle.assurance.robustness.report
    monkeypatch.setattr(
        report,
        "stability_report",
        lambda found, note: {"status": "stable", "counts": {}},
    )
    monkeypatch.setattr(
        report,
        "shift_report",
        lambda found: {"status": "unverifiable", "reason": "no reference"},
    )
    monkeypatch.setattr(report, "conform", lambda document: None)
    monkeypatch.setattr(report, "validate_shift", lambda document: None)
    monkeypatch.setattr(
        qualification.html_report, "render", lambda *given: "<html>"
    )
    stage = types.SimpleNamespace(found=found(), note="kept", requested=True)
    context = board()
    made = profiling.ProfileStage().assess_robustness(context, stage)
    assert made == qualification.Robustness(
        "stable", "unverifiable", "certified", "pass", ("b",)
    )
    before = dict(vars(context))
    made.apply(context)
    assert vars(context) == before
    assert isinstance(made, products.Derived)


def test_identify_returns_the_claim_estimated_on_the_opening():
    block = types.SimpleNamespace()
    charter = types.SimpleNamespace(
        estimand=types.SimpleNamespace(causal=block)
    )
    run = types.SimpleNamespace(
        confirmation=types.SimpleNamespace(
            documents={
                identification.claim.CLAIM_FILE: {
                    "status": "supported",
                    "tier": "B",
                }
            }
        )
    )
    context = board()
    made = profiling.ProfileStage().identify(
        context, charter, "view", run, None, "found"
    )
    assert made == identification.Estimand("supported", "B", estimated=True)
    before = dict(vars(context))
    made.apply(context)
    assert vars(context) == before


def test_a_predictive_charter_identifies_nothing():
    charter = types.SimpleNamespace(estimand=types.SimpleNamespace(causal=None))
    made = profiling.ProfileStage().identify(
        board(), charter, "view", "run", None, "found"
    )
    assert made is None


def test_planning_returns_the_blueprint_without_applying_it(monkeypatch):
    plan = types.SimpleNamespace(to_dict=dict, decisions=())
    advice = (types.SimpleNamespace(to_dict=dict),)
    monkeypatch.setattr(planning, "observe", lambda context: None)
    monkeypatch.setattr(planning, "devise", lambda context: plan)
    monkeypatch.setattr(planning, "advise", lambda context: advice)
    context = board()
    made = planning.Planning().execute(context)
    assert made.plan is plan
    assert made.recommendations == advice
    assert made.capability_registry is context.services.capability_registry
    assert context.plan is None
    assert context.recommendations == ()
    returned = planning.Planning().execute(context)
    assert returned.plan is plan
    assert context.plan is None
    returned.apply(context)
    assert context.plan is plan
    assert context.recommendations == advice
