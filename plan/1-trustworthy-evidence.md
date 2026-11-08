# Stage 1: Trustworthy evidence

Names follow [`00-canon.md`](00-canon.md). Item classes and reasons:
[`COMPLETION_AUDIT.md`](COMPLETION_AUDIT.md), "Stage 1". Nothing else in this
plan is believed until this stage is done.

## Goal

A pushed head on which every CI job, on every platform, has run and passed, and
on which the published quickstart works as written. A failed, cancelled or
unread job is never a pass.

## Starting point (2026-11-08, `master` at `9356b89`)

* `ci.yml` is green: 23 of 23 jobs in run 42471349728. Four jobs first failed
  with GitHub's own HTTP 504 before running any of our code and passed on
  re-run. The Windows 3.11 and 3.14 jobs, which failed on `f453091` because text
  artifacts were written with CRLF, pass: every writer in `scmle/` now sets
  `newline="\n"` and `tests/unit/test_text_writes_use_lf.py` keeps it so.
* `full-suite.yml` run 42476834640: the test step passed, including
  `test_oracle_table` on Linux 3.14 (the generator's floor is now 1e-14). The
  next step, "Enforce critical mutation score", failed: mutmut instruments the
  sources in its scope, among them the detector and stability modules whose
  hashes the certificates record, so `qualification --check` reports them
  stale inside its tree. This has failed since 4 Nov (run 41805832562), before
  the work above. `test_the_command_line_runs_smoke_check_and_writes_a_sample`
  now skips when the sources are instrumented; **not yet seen passing**.
* Windows jobs take 24 (3.14) and 27 (3.11) minutes; the limit is 60.
* The last full-suite run to succeed was 3 Nov (run 41612013024).

## Work packages

| WP | Items | Class | Scope | Closing criterion |
|---|---|---|---|---|
| 1.1 | 1.4 | B | Read one `full-suite.yml` run on the head step by step. If the mutation step still fails, read why: the baseline test run, or the score below the repo floor | every step of one `full-suite.yml` run green and read |
| 1.2 | 1.3 | A | Re-run the published quickstart (`docs/START_HERE.md`, `website/src/pages/docs/quickstart.astro`) verbatim in a clean environment on macOS, Linux and Windows. Fix the page or the code where they disagree | a transcript per platform, kept with the change |
| 1.3 | 1.5 | B | Watch the Windows duration on three consecutive master runs. If it stays above 14 minutes: split the cross-platform job (unit versus property, leakage and end-to-end) or move slow property tests to the nightly job, keeping a Windows custody and vault test in the bounded set. Do not raise `timeout-minutes`: the limit exists to catch hangs | Windows jobs under 14 minutes on three consecutive master runs |

## Exit gate

```bash
uv run pytest -q tests/unit/test_replay_identity.py tests/benchmarks/test_oracle_table.py
uv run python -m benchmarks.qualification --check all
uv run ruff check . && uv run ruff format --check . && uv run mypy scmle benchmarks
```

Then push and read every job of `ci.yml` and `full-suite.yml` on the final
head.

## Risks

| Risk | Mitigation |
|---|---|
| The oracle drift is a platform numeric difference, not the floor | Report it and tolerate it explicitly in the generator with a dated reason; never edit the document to match one platform |
| Windows stays slow after splitting | Report the measured time; the 14-minute target is a proposal, the 60-minute limit is the real bound |

## Not claimed

That the repository is green on any head other than one whose jobs were read.
