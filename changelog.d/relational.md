- Related tables (`scmle.data.relational`): declare tables (event time,
  recorded time, recording delay) and links (`many_to_one`, `one_to_many`)
  in code or as a JSON schema (`relational.parse`, which refuses an entry of
  the wrong type with a `ConfigError`). Read them from a SQLite file, a
  Parquet folder or a CSV set through one `Store` interface
  (`stores.locate` recognises the kind). A folder table can be one file or a
  directory of part files, either `orders/` or the Spark/pyarrow layout
  `orders.parquet/`. A table held by two entries is refused.
  `Engine.features` joins each target row to the latest version visible at
  its cutoff (moment less lag). It aggregates `one_to_many` rows with
  `count`, `sum`, `mean`, `min`, `max`, `last` and `trend` over windows
  (default 7, 30 and 90 days and all rows). A row is visible only once its
  event and recorded times are at or before the cutoff, so no feature reads
  the future. A target row with no moment is refused with an `InputError`
  rather than given a count of 0. On the five real relational datasets the
  C5 evidence caches, the engine's aggregates match both the row-by-row
  naive reference (`benchmarks/qualification/ingest_naive.py`) and the
  as-of synthesizer, for every primitive, window, rule and lag. Its
  `many_to_one` join matches the naive lookup on the four of those datasets
  whose rows carry an identifier (baseball's do not).
- The ingest stage (`--ingest-plan`) now runs the relational engine. After
  synthesis it computes every kept feature again with `Engine.features`
  from the same extracts. A feature that differs beyond 1e-8 stops the run
  with a `DataError`. The run event's summary gains an `engine` entry: the
  number of features checked, and each feature the engine cannot compute,
  with the reason. Those are `std`, `nunique` and `time_since_last`,
  depth-2 nests, and views that are `non_versioned` but have an event time.
  The synthesizer is still what produces the run's features. A run-table row
  with no moment now stops the run with an `InputError`. Before this change
  it silently got a count of 0.
- Ingest plans may name a `store` (`{"kind": "sqlite" | "parquet" | "csv",
  "path": ...}`) instead of a `database`, but not both. A folder is mirrored
  once into a SQLite file for the run, and its row counts are checked. A
  view may declare `cardinality`: `one_to_many` sets the `child` role and
  `many_to_one` the `parent` role, and a contradicting role is refused. A
  `many_to_one` table whose rows repeat a (key, event time, recorded time)
  value stops the run with a `SchemaError` before any feature is
  synthesized.
