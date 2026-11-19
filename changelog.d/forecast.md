- Forecasting and panels (backlog item 3). Panel data with an entity and a
  time column (`scmle.models.forecast.Panel`) now has three forecast
  families: the seasonal-naive baseline (`Seasonal`), simple exponential
  smoothing with its weight fitted per series (`Smoothing`), and direct
  multi-step gradient boosting on lag and rolling features scaled per
  series, one model shared by every series (`Boosted`).
  `scmle.evaluation.forecasting.backtest` compares them by rolling-origin
  evaluation: origins one horizon apart back from the end, every family
  refitted on the history before each origin, MAE and in-sample-scaled MASE
  at each step ahead, and a one-sided Wilcoxon test against the baseline
  that is left unassessed below the 5 pairs that can reach p < 0.05. A
  history too short for a family to learn from is a `BlockedError`, not a
  `DataError`. `forecasting.run(config, frame)` runs it from a project
  configuration: the panel is read from `data.value_column`, the families
  take the configured lags, windows, period and seed
  (`forecasting.contenders`), and `validation.horizon_rows` is converted
  from table rows to time points per series.
- What the comparison shows: against the weekly seasonal-naive baseline,
  boosting wins (p < 0.05) on seasonal series with a trend, which the
  baseline trails by a season's rise. On a pure season with noise it is not
  shown to beat seasonal naive (p = 0.44 in the test panel).
- Two new split strategies. `grouped_temporal` (`temporal.Succession`)
  keeps each group whole and on one side of the test split and of every
  fold, orders groups by first timestamp, and keeps every training row
  earlier than the rows it is checked against. With a forecast horizon its
  embargo covers the horizon for the groups running at one time, not for
  every group. `spatial_block` (`spatial.Tiling`) keeps square blocks of
  space whole; the block side is derived from the label's semivariogram
  and reported beside its range. The manifest records
  `validation.entities` only for `temporal` and `rolling`;
  `splitter.TIMED` names the three time-ordered strategies.
