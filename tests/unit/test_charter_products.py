"""The charter stages return frozen products; the vault closes on every exit.

Each stage of a charter run (exploration, qualification, identification,
impounding) and the planning stage returns a frozen product that
`Product.apply` holds on the run's board by type. The stages run here on
fake services: a real board over services with no run directory, and fakes
for the charter, the custody and the reports.
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
from scmle.pipeline import engine
from scmle.pipeline import products
from scmle.pipeline import services
from scmle.pipeline.stages import exploration
from scmle.pipeline.stages import identification
from scmle.pipeline.stages import impounding
from scmle.pipeline.stages import planning
from scmle.pipeline.stages import profiling
from scmle.pipeline.stages import qualification
from scmle.runtime import limits


def start(**data):
    """A run's board over fake services: no run directory, a charter path."""
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
    return engine.Board.start(given)


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


def impound(board, split):
    """Sequesters as the profiling stage does: applies the custody."""
    made = profiling.ProfileStage().sequester(
        board, tasks.Task.BINARY, FRAME, split
    )
    made.apply(board)
    return made.outcome.split


def test_sequester_returns_the_custody_and_leaves_the_context(custody):
    board = start()
    made = profiling.ProfileStage().sequester(
        board, tasks.Task.BINARY, FRAME, "split"
    )
    assert isinstance(made, exploration.Custody)
    assert made.vault is custody[0]
    assert made.column is None
    assert board.find(exploration.Custody) is None
    with pytest.raises(dataclasses.FrozenInstanceError):
        made.column = "g"  # type: ignore[misc]


def test_applied_custody_leaves_its_vault_to_the_context_close(custody):
    board = start()
    split = impound(board, "split")
    assert split == "restricted"
    assert board.get(exploration.Custody).vault is custody[0]
    assert board.get(exploration.Custody).outcome.manifest.sha256 == "m"
    assert custody[0].closed == 0
    board.close()
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
    board = start()
    with pytest.raises(type(failure)):
        impound(board, "split")
    assert [vault.closed for vault in custody] == [1]
    assert board.find(exploration.Custody) is None


def test_the_vault_closes_when_its_records_fail(custody, monkeypatch):
    def chronicle(*given):
        raise OSError("disk full")

    monkeypatch.setattr(impounding.Impounding, "chronicle", chronicle)
    board = start()
    with pytest.raises(OSError, match="disk full"):
        impound(board, "split")
    assert [vault.closed for vault in custody] == [1]
    board.close()
    assert [vault.closed for vault in custody] == [1]


def test_no_vault_is_made_when_the_charter_is_refused(custody, monkeypatch):
    charter = types.SimpleNamespace(purpose=contracts.NEW_ENTITY)
    monkeypatch.setattr(
        impounding.sealing,
        "read_verified",
        lambda path: types.SimpleNamespace(charter=charter, signed=True),
    )
    with pytest.raises(errors.ConfigError, match="group_column"):
        impound(start(), "split")
    assert custody == []


def test_a_later_failure_leaves_the_vault_to_the_context_close(custody):
    board = start()
    impound(board, "x")
    try:
        with pytest.raises(errors.BlockedError):
            raise errors.BlockedError("a later stage stops the run")
    finally:
        board.close()
    assert [vault.closed for vault in custody] == [1]


def test_a_known_entity_charter_names_the_cluster_column(custody, monkeypatch):
    charter = types.SimpleNamespace(purpose=contracts.KNOWN_ENTITY_FUTURE)
    monkeypatch.setattr(
        impounding.sealing,
        "read_verified",
        lambda path: types.SimpleNamespace(charter=charter, signed=True),
    )
    board = start(group_column="g", time_column="t")
    split = types.SimpleNamespace(
        manifest=types.SimpleNamespace(strategy="temporal")
    )
    made = profiling.ProfileStage().sequester(
        board, tasks.Task.BINARY, FRAME, split
    )
    assert made.column == "g"
    assert board.find(exploration.Custody) is None
    made.apply(board)
    assert board.get(exploration.Custody).column == "g"
    board.close()


def test_the_custody_digest_depends_on_the_manifest_only():
    first = exploration.Custody(outcome(), Vault(None), None)
    second = exploration.Custody(outcome(), Vault(None), None)
    other = exploration.Custody(outcome(), Vault(None), "g")
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


def test_claims_hold_the_pending_ledger_and_digest_their_charter():
    held = types.SimpleNamespace(made={})
    pending = escrow(coverage=True)
    exploration.Claims(pending).apply(held)
    assert held.made[exploration.Claims].pending is pending
    exploration.Claims().apply(held)
    assert held.made[exploration.Claims].pending is None
    assert exploration.Claims(pending).summary() == {
        "pending": True,
        "charter": "h",
        "coverage": True,
    }
    assert exploration.Claims().summary() == {"pending": False}


def test_without_custody_the_ledger_leaves_nothing_pending():
    board = start()
    made = profiling.ProfileStage().ledger(board, tasks.Task.BINARY)
    assert made == exploration.Claims()
    made.apply(board)
    assert board.get(exploration.Claims).pending is None


def test_a_task_outside_the_ledger_is_skipped_with_nothing_pending():
    board = start()
    exploration.Custody(outcome(), Vault(None)).apply(board)
    made = profiling.ProfileStage().ledger(board, tasks.Task.SURVIVAL)
    assert made.pending is None
    board.close()


def test_defer_keeps_the_ledger_off_the_context():
    board = start()
    stage = profiling.ProfileStage()
    assert stage.defer(board, escrow(coverage=True))
    assert not stage.defer(board, escrow(coverage=False))
    assert board.find(exploration.Claims) is None


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
    board = start()
    made = profiling.ProfileStage().assess_robustness(board, stage)
    assert made == qualification.Robustness(
        "stable", "unverifiable", "certified", "pass", ("b",)
    )
    made.apply(board)
    assert board.get(qualification.Robustness) is made
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
    board = start()
    made = profiling.ProfileStage().identify(
        board, charter, "view", run, None, "found"
    )
    assert made == identification.Estimand("supported", "B", estimated=True)
    made.apply(board)
    assert board.get(identification.Estimand) is made


def test_a_predictive_charter_identifies_nothing():
    charter = types.SimpleNamespace(estimand=types.SimpleNamespace(causal=None))
    made = profiling.ProfileStage().identify(
        start(), charter, "view", "run", None, "found"
    )
    assert made is None


def test_planning_returns_the_blueprint_without_applying_it(monkeypatch):
    plan = types.SimpleNamespace(to_dict=dict, decisions=())
    advice = (types.SimpleNamespace(to_dict=dict),)
    monkeypatch.setattr(planning, "observe", lambda board: None)
    monkeypatch.setattr(planning, "devise", lambda board: plan)
    monkeypatch.setattr(planning, "advise", lambda board: advice)
    board = start()
    made = planning.Planning().execute(board)
    assert made.plan is plan
    assert made.recommendations == advice
    assert made.capability_registry is board.services.capability_registry
    assert board.find(planning.Blueprint) is None
    returned = planning.Planning().execute(board)
    assert returned.plan is plan
    assert board.find(planning.Blueprint) is None
    returned.apply(board)
    assert board.get(planning.Blueprint) is returned
    assert board.services.capability_registry is returned.capability_registry
