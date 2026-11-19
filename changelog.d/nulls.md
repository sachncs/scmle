- C3 null clause: `benchmarks/evidence/nulls.py` measures how often the
  product claims skill on tasks that have none. It runs 577 null tasks end to
  end through `Project` (60 s budget, the C4 benchmark's): nine permutations
  of the target of each of the 35 pinned C4 tables (315 real tasks) and 262
  simulated tables whose target is drawn apart from the features (three
  worlds, binary, multiclass and regression, 300 to 3,000 rows), seeded from
  the registered stream `coverage.null.validation`. A run claims skill when
  its CV or test interval lies wholly better than its `dummy` baseline (the
  cross-validation baseline it reports, and the `dummy` fitted on its own
  development rows and scored on its own test rows), or when its own fit
  verdict is `healthy`; a crash counts as a claim, a refusal does not, and an
  unknown metric direction is refused. The pooled one-sided upper bound at
  `alpha` must be at most `null_skill_max` (0.05). The count is planned from
  `acceptance.toml`: the smallest sample whose exact chance of passing
  reaches `power` when the true rate is `null_skill_max / (1 + size_margin)`
  = 0.0296, and never fewer than `null_tasks_min`.
  `python -m benchmarks.evidence.nulls --write` measures it; `--check`
  checks the stored result and refuses one measured with other code (the
  module, the C4 catalog, the acceptance rules or the product), other pinned
  data, or another bound, alpha, power or margin.
