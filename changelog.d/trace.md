### Added

* **A bundle's trace is verified, not only present.** `scmle.trail.origin.trace(bundle)`
  (and `python -m scmle.trail trace BUNDLE`) checks that the four C6 trace
  fields are recorded and that each verifies against records other than the
  manifest: the data file rehashes to the data hash (raw bytes and, for a
  tabular file read whole, the parsed frame); the run's `config.json`
  rehashes to the configuration hash; the commit exists in the repository and
  its `pyproject.toml` declares the scmle version the bundle records; the
  pins agree with the manifest's own scikit-learn, pandas, scmle and library
  versions and with the run's provenance lock. The manifest is trusted only
  when every stored file matches its digest and the signature verifies.
* **Rebuild a bundle from its manifest alone.** `scmle.trail.replica.rebuild(bundle)`
  (and `python -m scmle.trail rebuild BUNDLE`) refuses a bundle that does not
  trace, then creates a fresh `uv` environment at the manifest's Python with
  exactly its pins (`--no-deps`), installs scmle from a clean clone at the
  manifest's commit, retrains the stored configuration in an isolated
  process with the manifest's thread counts, and compares every member byte
  for byte and the locked-test metrics within `replica.TOLERANCE` (1e-6
  relative). Exit codes: 0 traced/reproduced, 1 not, 2 on an error.
* **C6 evidence for the trace and clean-runner clauses.**
  `benchmarks/evidence/traceability.py` trains each task with the real
  command line, traces every bundle, applies seven tamperings (including a
  removed signature) that must each be caught by exactly the field they
  break, and rebuilds each bundle in a temporary environment. It refuses to
  build bundles from a working tree with uncommitted changes, counts an
  unsigned bundle as not intact, records a bundle that cannot be rebuilt
  instead of stopping (so `--rebuild` always writes its record before
  exiting 1), and fails the clean-runner clause when a bundle does not trace
  on that runner. The `reproducibility` workflow gains a `bundle` job and a
  `rebuild` job that rebuilds those bundles on a fresh runner; its record is
  stored with `--merge`.
