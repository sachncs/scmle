"""What an identification run copies into its claim file and report."""

import pandas
import pytest

from scmle.assurance.charter import causal
from scmle.assurance.ledger import budget
from scmle.causal.identification.run import analysis
from scmle.causal.identification.run import estimator

BUDGET = budget.Alpha.create(0.05, 0.1)


def block(**changes):
    values = {
        "treatment": "t",
        "outcome": "y",
        "estimand": "ATE",
        "assumptions": (
            "unconfoundedness_given:a,b",
            "positivity",
            "no_interference",
        ),
        "strategies": ("backdoor_aipw",),
    }
    values.update(changes)
    return causal.Question(**values)


class Fixed(estimator.Estimator):
    """A fixed estimate on any rows."""

    name = "backdoor_aipw"

    def prepare(self):
        self.design = {"fixed": True}
        self.refusal = None

    def estimate(self, c, level):
        return estimator.Est(1.0, 0.1, 0.8, 1.2, len(c), {"k": 1})

    def controls(self, c, level):
        return []


@pytest.fixture(autouse=True)
def fixed(monkeypatch):
    monkeypatch.setitem(analysis.ADAPTERS, "backdoor_aipw", Fixed)


def test_a_design_only_claim_is_identified():
    out = analysis.diagnose(block(), "X", BUDGET)

    assert out.documents["causal_claims.json"]["status"] == "identified"
    assert out.decision.status == "identified"


def test_an_identification_reports_its_primary_alpha():
    c = pandas.DataFrame({"a": range(8)})

    out = analysis.identify(block(), "X", c, BUDGET, certified=())

    report = out.documents["identification_report.json"]
    assert report["alpha_primary"] == pytest.approx(0.04)


def test_declared_subgroups_reach_the_claim_file():
    spec = block(subgroups=(causal.Subgroup("a", ("0", "1")),))
    c = pandas.DataFrame({"a": [0, 1] * 6})

    out = analysis.identify(spec, "X", c, BUDGET, certified=())

    claims = out.documents["causal_claims.json"]
    report = out.documents["identification_report.json"]
    assert (
        claims["subgroups"]["family_size"] == report["subgroups"]["family_size"]
    )
    assert claims["subgroups"]["rule"] == report["subgroups"]["rule"]
