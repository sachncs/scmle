### Added

- `benchmarks/evidence/crashes.py` measures C7's kill-and-resume clause at
  every kill point it can find, without sampling. Four subjects run in child
  processes: the pipeline engine (`Project.run`/`Project.resume`), the
  workflow executor (leases and fenced records on a local object store), the
  batch scorer's progress journal and the erasure journal. A fault hook in the
  harness traces each stage record, completion marker, event, trial
  checkpoint, custody entry, lease put/delete, executor record, batch chunk,
  progress-journal write, atomic swap, erasure plan, erasure step and receipt.
  It records the moment before and after each operation. At each traced point,
  a fresh run gets SIGKILL to its whole process group and then resumes in a new
  process. Its final outputs are compared byte for byte with an uninterrupted
  reference, and the module also checks that no record durable at the kill is
  written again. Stage products are unsealed and the leaderboard read into
  canonical JSON, so they compare field by field; each stage record's digests
  are replaced by whether they hold (its own body, its product file, the
  products it needs). Executor records are checked for their fencing token: a
  record the resume wrote must carry one more than the lease the killed run
  left. Three control runs (`CONTROLS`) find what varies between
  uninterrupted runs; only the exact JSON fields that differ in some control
  are masked, and every other field is still compared. Outputs that are not
  JSON and that differ as a whole are compared for presence only. An
  essential output (the model, stage records and products, leaderboard,
  trial records, predictions, receipt) fails the subject if it is volatile or
  has a masked field that is not a timing. Masked fields and presence-only
  outputs are listed in the document. Run it with
  `python -m benchmarks.evidence.crashes --write | --check | --measure OUT`.
