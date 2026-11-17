"""Online false-discovery-rate rules as exact state machines.

Hypotheses arrive one at a time. A rule turns the history of past decisions
into a test level for the next hypothesis, so each rule is a pure function
``(state, evidence) -> (step, new state)``. All levels, thresholds and wealth
are exact `fractions.Fraction` values and every comparison is exact, so a
replay of the same evidence reproduces every level and decision bit for bit on
any platform and Python version (a float p-value is read as its exact binary
value). Floating point enters only when a p-value is calibrated into an
e-value, and the ledger stores the e-value it used.

Verified statements (read from the primary sources on 2026-11-03).
"Verified" means the statement was read in the cited arXiv paper or in the
onlineFDR manual; it does not mean this code is proved correct. Journal and
conference venues are quoted as the manual or the paper gives them; the
AISTATS 2024 venue of e-LOND is from memory and was not checked (the arXiv
text read is v1, November 2023).

e-LOND (Xu and Ramdas, "Online multiple testing with e-values",
arXiv:2311.06412v1, Section 3.2, Theorem 1; proof in Section 8.1)
    Level ``alpha_t = alpha * gamma_t * (|R_{t-1}| + 1)`` where ``gamma_t``
    is a fixed nonnegative sequence with ``sum gamma_t <= 1``. Hypothesis
    ``t`` is rejected when ``e_t >= 1 / alpha_t``. If every true null has a
    valid e-value, ``E[e_t] <= 1`` (their equation 3), then
    ``FDR <= alpha`` at every time ``t`` under *arbitrary* dependence between
    the e-values. The proof uses only ``E[e_t] <= 1``; it needs neither
    independence nor a conditional version of the e-value condition. The
    hypotheses are indexed in advance (the index set of true nulls is fixed),
    so this module's ledger registers every hypothesis before it is tested and
    never lets an earlier result choose a later hypothesis.

p-to-e calibration (Vovk and Wang, "E-values: calibration and combination",
Annals of Statistics 49(3), 2021, doi:10.1214/20-AOS2020)
    A calibrator ``f: [0, 1] -> [0, inf)`` is nonincreasing, upper
    semicontinuous and has ``integral f <= 1``; ``f(P)`` is then an e-value
    for every super-uniform ``P``. The family used here is
    ``f_kappa(p) = kappa * p ** (kappa - 1)`` for ``kappa`` in (0, 1)
    (``integral = 1``). For a nonincreasing ``f`` and a super-uniform ``P``,
    ``E f(P) <= integral f`` (the layer-cake formula), so the calibrated
    value is a valid e-value; Xu and Ramdas (Corollary 1) use the same device
    to feed p-values to e-LOND. The cost is power: the rejection rule
    ``f(p) >= 1 / alpha_t`` is ``p <= (kappa * alpha_t) ** (1 / (1 - kappa))``.
    The default ``kappa = 1/2`` makes ``e = 1 / (2 * sqrt(p))``, whose only
    floating-point operation (a square root) is exactly rounded and so
    identical on every platform.

LORD++ (Ramdas, Yang, Wainwright and Jordan, "Online control of the false
discovery rate with decaying memory", NeurIPS 2017, arXiv:1710.00499,
equation (5) and Theorem 1)
    ``alpha_t = gamma_t w0 + (alpha - w0) gamma_{t - tau_1}
    + alpha * sum_{j >= 2, tau_j < t} gamma_{t - tau_j}`` with ``tau_j`` the
    time of the j-th rejection, ``0 <= w0 <= alpha`` and a nonincreasing
    positive ``gamma`` summing to one. Hypothesis ``t`` is rejected when
    ``p_t <= alpha_t``. The theorem gives ``FDR <= alpha`` for *independent*
    p-values. The onlineFDR manual (Bioconductor 3.23, version 2.20.0) adds
    that control also holds if only the true-null p-values are mutually
    independent and independent of the non-null p-values. There is no
    guarantee under dependence.

SAFFRON (Ramdas, Zrnic, Wainwright and Jordan, "SAFFRON: an adaptive
algorithm for online control of the false discovery rate", ICML 2018,
arXiv:1802.09098, Section 2.3 and Theorem 1)
    A p-value ``<= lambda`` is a candidate. ``alpha_t = min(lambda,
    (1 - lambda) * (w0 gamma_{t - C0+} + (alpha - w0) gamma_{t - tau_1 - C1+}
    + alpha * sum_{j >= 2} gamma_{t - tau_j - Cj+}))`` where ``Cj+`` counts
    the candidates in ``(tau_j, t - 1]``. FDR control at ``alpha`` holds for
    independent p-values (Theorem 1(d), which also needs the levels to be
    monotone functions of the past, as here); only mFDR control is claimed
    under conditional super-uniformity. The onlineFDR manual gives the same
    independence statement and the defaults ``lambda = 1/2``,
    ``w0 = alpha / 2``.

Not verified, so not claimed. R ``onlineFDR`` 2.20.0 has no e-LOND function,
and no R output could be generated here (R is not installed), so these rules
are checked against hand calculations from the formulas above and against
brute-force re-derivations of the levels from the definitions, not against R.
The package's default discount sequence is ``gamma_j`` proportional to
``j ** -1.6``, an irrational normalisation that cannot be held exactly, so the
default here is the exact ``1 / (j (j + 1))`` (sum one). Defaults for the
initial wealth follow the papers (``w0 = alpha / 2``); the onlineFDR LORD
default is ``alpha / 10``.

Bookkeeping of "wealth". For LORD++ and SAFFRON the wealth after ``t`` tests
is the credit earned (initial wealth plus a reward per rejection) minus the
level spent; the discount sequence uses each ``gamma`` index at most once per
credit source, so it never goes negative, and this module raises if it ever
did. For e-LOND there is no spending: the wealth reported is
``alpha * (1 - sum_{i <= t} gamma_i)``, the share of the level not yet
assigned to any hypothesis.
"""

from scmle.assurance.ledger.online_fdr.rules import Lond
from scmle.assurance.ledger.online_fdr.rules import Lord
from scmle.assurance.ledger.online_fdr.rules import PValueTester
from scmle.assurance.ledger.online_fdr.rules import replay
from scmle.assurance.ledger.online_fdr.rules import Saffron
from scmle.assurance.ledger.online_fdr.rules import Settings
from scmle.assurance.ledger.online_fdr.tester import calibrate
from scmle.assurance.ledger.online_fdr.tester import DEFAULT_KAPPA
from scmle.assurance.ledger.online_fdr.tester import DEFAULT_LAMBDA
from scmle.assurance.ledger.online_fdr.tester import Discount
from scmle.assurance.ledger.online_fdr.tester import ELOND
from scmle.assurance.ledger.online_fdr.tester import Evidence
from scmle.assurance.ledger.online_fdr.tester import Guarantee
from scmle.assurance.ledger.online_fdr.tester import InvariantError
from scmle.assurance.ledger.online_fdr.tester import LORD_PLUS_PLUS
from scmle.assurance.ledger.online_fdr.tester import P_FLOOR
from scmle.assurance.ledger.online_fdr.tester import rational
from scmle.assurance.ledger.online_fdr.tester import RULE_NAMES
from scmle.assurance.ledger.online_fdr.tester import SAFFRON
from scmle.assurance.ledger.online_fdr.tester import State
from scmle.assurance.ledger.online_fdr.tester import Step
from scmle.assurance.ledger.online_fdr.tester import Tester

__all__ = [
    "DEFAULT_KAPPA",
    "DEFAULT_LAMBDA",
    "ELOND",
    "LORD_PLUS_PLUS",
    "P_FLOOR",
    "RULE_NAMES",
    "SAFFRON",
    "Discount",
    "Evidence",
    "Guarantee",
    "InvariantError",
    "Lond",
    "Lord",
    "PValueTester",
    "Saffron",
    "Settings",
    "State",
    "Step",
    "Tester",
    "calibrate",
    "rational",
    "replay",
]
