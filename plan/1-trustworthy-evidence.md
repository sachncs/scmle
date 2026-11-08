# Stage 1: Trustworthy evidence

Names follow [`00-canon.md`](00-canon.md). Item classes and reasons:
[`COMPLETION_AUDIT.md`](COMPLETION_AUDIT.md), "Stage 1". Nothing else in this
plan is believed until this stage is done.

## Goal

A pushed head on which every CI job, on every platform, has run and passed, and
on which the published quickstart works as written. A failed, cancelled or
unread job is never a pass.

## Starting point (2026-11-08)

* `ci.yml` run 42282548069 (head `f453091`) was red: Windows 3.11 and 3.14
  failed `tests/unit/test_replay_identity.py` because text artifacts were
  written without `newline="\n"`, so Windows wrote CRLF. All 36 writers in
  `scmle/` now set it and `tests/unit/test_text_writes_use_lf.py` keeps it so.
  The trail records and evidence digests hash raw file bytes, so this was a
  real cross-platform defect, not cosmetic. **Not yet seen green on Windows.**
* `full-suite.yml` run 42356989620 was red on Linux 3.14 at
  `tests/benchmarks/test_oracle_table.py`. The generator's floor was raised from
  1e-16 to 1e-14 because the raw gaps are one to four units in the last place.
  **That cause is inferred, not seen on Linux.**
* The last head with a fully green `ci.yml` run is `9d4bc04` (run 42227276889).
* Windows jobs take 26 to 27 minutes (6475 tests passed, one failed in run
  42282548069); the limit is 60.

## Work packages

| WP | Items | Class | Scope | Closing criterion |
|---|---|---|---|---|
| 1.1 | 1.2 | B | Confirm the oracle-table fix on Linux 3.14 (the next `full-suite.yml` run, or a Linux container). If it still differs, read the diff and fix the generator; do not weaken `--check` or edit the document to match one platform | `pytest tests/benchmarks/test_oracle_table.py` passes on Linux 3.14 and macOS |
| 1.2 | 1.3 | A | Re-run the published quickstart (`docs/START_HERE.md`, `website/src/pages/docs/quickstart.astro`) verbatim in a clean environment on macOS, Linux and Windows. Fix the page or the code where they disagree | a transcript per platform, kept with the change |
| 1.3 | 1.4, 1.5 | B | Push; read every job of one `ci.yml` run on the head, and the next `full-suite.yml` run. Watch the Windows duration on three consecutive master runs. If it stays above 14 minutes: split the cross-platform job (unit versus property, leakage and end-to-end) or move slow property tests to the nightly job, keeping a Windows custody and vault test in the bounded set. Do not raise `timeout-minutes`: the limit exists to catch hangs | a `ci.yml` run on the head with every job green and read; Windows jobs under 14 minutes on three consecutive master runs |

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
