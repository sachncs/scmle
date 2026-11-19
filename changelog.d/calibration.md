### Added

- `benchmarks/evidence/calibration.py` measures C3's clause "every interval
  and test nominal by simulation-based calibration". Each of 30 scenarios
  simulates a world with a known truth. It then calls one product interval or
  test the way a user would: the bootstrap metric interval (three families),
  split-conformal intervals, pooled and class-conditional conformal sets,
  weighted conformal under covariate shift, CV+, the Nadeau-Bengio CV
  interval, the RD effect with its balance, placebo and density tests, the
  DiD ATT with its pre-trend test, the randomized difference in means, the
  Holm subgroup family, IPW and AIPW effects and p-values, linear (HC3),
  logistic and conjugate-Bayesian coefficients, the leakage skill
  permutation test (with and without groups), the missingness test, the
  leakage screen under the global null, the slice coverage flags, the
  frozen-nuisance AIPW ATE and ATT, the IV headline, 2SLS and
  Anderson-Rubin sets under a strong and a weak instrument, Sargan and
  Hansen J under valid instruments, and the synthetic-control placebo
  p-value under no effect.
- The module checks 64 quantities: coverage, size and rank uniformity, each
  against its exact nominal value. Those values include the finite-sample
  conformal rank, the class-size mixture for Mondrian sets, and the exact
  familywise rate of a Holm family. Each check gets a two-sided
  Clopper-Pearson bound at `alpha / 64`, so a registry of nominal procedures
  is flagged with chance at most `alpha`. Each scenario's sample is planned
  by an exact binomial search, so every check sees a 0.025 miscalibration
  with chance `power`.
- A check outside its bound is `liberal`, `conservative` or `nonuniform`. It
  is listed under `miscalibrated` together with the product location that
  produced the number. Run it with
  `python -m benchmarks.evidence.calibration --write | --check`.
- The verdict is `partial`, never `pass`, while `NOT_MEASURED` lists a
  product interval or test no scenario calibrates (18, each by
  `path:name`). A judged field a trial did not report counts against the
  product and is named in the scenario's messages. Bootstrap trials keep
  their scikit-learn settings inside a context, so every other scenario
  runs under the defaults, whatever the worker count.
