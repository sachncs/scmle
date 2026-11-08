# Literature check of cited methods

On 2026-11-05 the primary sources of the methods below were
fetched and read (full text, not only abstracts) wherever they could be
retrieved, and the docstrings were corrected where a claim went beyond them.
"Verified" means a statement was read in the source and agrees with the text
of the module; it does not mean the code is correct (that is the tests and
certificates). Sources that could not be retrieved are listed as not verified.

| Method (module) | Source read | Verified | Corrected or not supported |
| --- | --- | --- | --- |
| Betting capital, plug-in bet, cap (`monitoring/sequential_tests.py`) | Waudby-Smith and Ramdas, arXiv:2010.09686, Sections 3 and 4, Theorems 1 to 3, Proposition 2 | Capital process and its predictable-bet martingale property; Ville's inequality; variance and mean estimators `(1/4 + sum)/(t+1)`, `(1/2 + sum)/(t+1)`; cap `c` in (0,1), default 1/2 or 3/4 | The bet formula is the paper's plug-in for the empirical-Bernstein CS (`log(2/alpha)`, two-sided), not for the one-sided capital used here with `log(1/alpha)`. That use is this module's adaptation; its validity rests on predictability only. Docstring now says so. |
| e-LOND, p-to-e calibration (`data/ledger/online_fdr.py`) | Xu and Ramdas, arXiv:2311.06412 (read earlier, recorded in the module) | Level rule and FDR statement under arbitrary dependence | AISTATS 2024 venue still from memory |
| Fixed-sequence threshold (`governance/escalation.py`) | Angelopoulos, Bates, Candes, Jordan, Lei, "Learn then Test", arXiv:2110.01052, Sections 1.1, 2.3.1, 4, Proposition 3 | Fixed-sequence testing stops at the first acceptance and controls FWER; exact binomial tail for selective classification | The paper assumes an i.i.d. calibration set; the docstring said "exchangeable". Corrected. |
| Selective risk and coverage (`governance/escalation.py`) | Geifman and El-Yaniv, arXiv:1705.08500; El-Yaniv and Wiener, JMLR 11 (2010) | Definitions of coverage and selective risk; guarantee with probability `1 - delta` over an i.i.d. sample | None |
| Chow's rule (`governance/escalation.py`) | Chow 1957 and 1970 not retrieved; described in El-Yaniv and Wiener, Section 10 | Only that it is the Bayes-optimal reject rule under 0-1 loss with known distribution and can be suboptimal with estimated posteriors | The reading of the cut as `c_h / c_e` is this module's own, not checked against Chow. Docstring says so. |
| CV+ and weighted conformal (`evaluation/conformal.py`, `cv_plus.py`) | Barber, Candes, Ramdas, Tibshirani, arXiv:1905.02928, Theorem 4, eq. 11; Tibshirani, Barber, Candes, Ramdas, arXiv:1904.06019, Corollary 1 | Interval construction; CV+ guarantee `1 - 2 alpha - min(...) >= 1 - 2 alpha - sqrt(2/n)` for exchangeable data and a symmetric algorithm; weighted quantile with the test point at plus infinity needs the exact likelihood ratio | "Coverage near `1 - alpha`" is an empirical expectation in the paper, not a theorem; the docstring said the statements came from the abstracts. Corrected. |
| MinT (`evaluation/reconcile.py`) | Wickramasuriya, Athanasopoulos, Hyndman, working-paper version of 23 January 2018 (the JASA 2019 text was not read) | `P = (S' W^-1 S)^-1 S' W^-1` minimises the trace among unbiased maps (`S P S = S`); shrinkage toward diagonal with intensity `sum Var(r_ij) / sum r_ij^2` | The unbiasedness constraint was missing from the docstring. Schaefer and Strimmer (2005) not read. |
| Net benefit and threshold (`data/framing/costs.py`) | Vickers and Elkin 2006, author manuscript PMC2577036 | `NB = TP/n - FP/n * pt/(1-pt)`; weight is the harm ratio, so `pt = c_FP/(c_FP + c_FN)` | "Recalled, not re-read" replaced |
| Robustness value, bias and standard-error formulas, benchmarking (`data/identification/sensitivity.py`) | Cinelli and Hazlett, JRSS-B 82(1), 2020, author PDF, eq. 12, 18 to 22, Supplement B | RV, RV with significance, se adjustment, benchmark bound `eta^2` (algebraically identical to the code); Table 1 example reproduced | The paper's rule for `RV_{q,a} <= 1 - 1/f_q^2` is not implemented (documented). "Written from memory" replaced. |
| Fellegi-Sunter weights (`data/ingest/entities.py`) | Fellegi and Sunter 1969 not retrieved; Winkler's EM paper not read | Only secondary descriptions (`log2(m/u)`, upper cutoff, clerical-review band) | Marked as not verified against the primary source |

## Not verified

* Begg and Greenes 1983 (`monitoring/labels.py`): not retrievable; the
  docstring still says "from memory, not read".
* Chow 1957 and 1970, Fellegi and Sunter 1969 (above).
* Uno et al. 2011 (`evaluation/ipcw.py`), Gneiting and Raftery 2007
  (`evaluation/proper_scores.py`), Romano et al. arXiv:1905.03222, Berger 1982,
  M4 and M5 papers: not opened in this pass.
* Every dataset named for replay or corpora (existence, licence, hash).
* No R or Python reference package was run (`sensemakr`, `hts`, `onlineFDR`,
  MAPIE).
