### Added

- `scmle.data.events` turns an append-only event log of (entity, time,
  type, payload) records into point-in-time features. Events come from
  whole files of any tabular format (`File`), JSON lines on a text stream
  such as standard input (`Lines`), or a JSON-lines file another process
  appends to, read as it grows and resumable from a byte offset (`Tail`); a
  strict source names the bad record, a lenient one skips it and keeps the
  reason. A `Store` takes events one at a time in any arrival order and
  answers `snapshot(entity, cutoff)` (each `kind.field`'s newest value and
  every window's value) and `frame(queries)` from events with
  `time <= cutoff` only. Windows are written `KIND[.FIELD]:STATISTIC:SPAN`
  (`purchase.amount:total:7d`, `*:count:1h`) over `(cutoff - span,
  cutoff]`, with `count`, `total`, `mean`, `min`, `max`, `last`,
  `distinct` and `recency`; a new `Statistic` subclass is found by its name
  with no registry. Events out of order within the declared lateness bound
  are placed where they belong; older ones are dropped, rejected
  (`LatenessError`) or accepted and counted. Under `drop` and `reject` a
  snapshot whose cutoff lies before the watermark is marked `final`; under
  `accept` no snapshot is final, since an event of any age may revise it. `lateness.bound` estimates
  the bound as a distribution-free 95% upper confidence bound on the 0.99
  quantile of observed delays (binomial order statistic; needs 299
  events). Tests check the incremental store against a from-scratch
  recompute over random disordered streams under every policy, and that
  final snapshots never change under every policy.
- `python -m scmle.data.events features|follow|lateness`: features at query
  cutoffs from files or stdin (`-`); `follow` answers each query from a
  growing log the moment the watermark passes its cutoff (under
  `--policy accept`, only once the log goes idle, marked not final);
  `--lateness auto`
  uses the estimated bound. Exit 2 on an input error, with a JSON summary of
  arrivals, bound, watermark and skipped records on standard error.
