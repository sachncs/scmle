### Added

- `scmle.evaluation.calibration` (a package: `verdict`, `labels`, `hazard`,
  `relevance`, `treatment`) tests the calibration of held-out predictions
  for the structured tasks and states the claim each supports
  (`STATEMENTS`): ordinal cumulative probabilities `P(level <= k)` at every
  threshold (`ordinal`), each label's probability (`multilabel`), event-free
  probabilities at horizons against the Kaplan-Meier estimate with
  Greenwood's variance within predicted-risk groups (`survival`; default
  horizons are the event-time `quartiles`), and for ranking both the order
  and the scale of the scores (`ranking`): the mean NDCG gain over a random
  order of the same rows, with a query bootstrap interval that must lie
  above zero (`gain`, `chance`), and the probability of relevance that an
  increasing Platt map learnt on other query folds gives each score
  (`crossfit`, `Platt`); a reversed scorer, whose map decreases, never
  supports the claim. Every test is a grouped Hosmer-Lemeshow chi-square
  with one degree of freedom per group (deciles, merged until each cell
  expects 5 counts, Cochran's rule); a claim holds only when every test ran
  and none rejects at 5% after a Bonferroni split, so an untestable check
  never supports a claim.
- `calibration.uplift` gives the Qini coefficient and the area under the
  uplift curve (`area`) with arm-stratified bootstrap percentile intervals
  (2,000 draws), and tests predicted effects against the treated-minus-
  control difference within groups of at least 30 rows per arm. The uplift
  claim needs the Qini interval above zero and the effect test not
  rejected.
- Measured size: `test_calibrated_predictions_are_rejected_at_most_at_the_level`
  draws 1,000 seeded calibrated simulations per task and asserts that the
  rate at which some check rejects stays within 5% plus a one-sided 99.9%
  binomial margin (2.1 points). The measured rates are 5.1% ordinal, 4.8%
  multilabel, 4.9% survival, 2.3% ranking and 6.1% uplift; uplift over
  8,000 seeds is 5.35%, slightly above the nominal 5% because its group
  test is a two-sample normal approximation with 30 rows per arm.
- A linear family for ranking: `pairwise_linear_ranker`
  (`scmle.models.ranking.estimators.Pairwise`), a logistic RankSVM on
  within-query preference pairs, capped at 50,000 sampled pairs.
- A linear family for uplift: `uplift_linear_t_learner`, a T-learner on
  logistic or ridge outcome models. Every uplift learner takes
  `kind="boosting" | "linear"` (`scmle.models.uplift.estimators.KINDS`); an
  arm whose label has one class gets a prior classifier. A linear
  S-learner's effect is homogeneous: one treatment coefficient for every
  row.

### Changed

- `scmle.models.survival.estimators.kaplan_meier` is vectorised (one sort
  and a cumulative product instead of a pass over the rows per event time);
  its values are unchanged.
