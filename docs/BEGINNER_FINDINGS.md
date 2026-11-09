# Beginner findings

What went wrong, or was confusing, when the author followed the beginner
path on the Titanic table that ships in the repository (`tests/fixtures/
titanic_extract.csv`), using the real command line and the Python API. Each
item says what happened, what was done, and what is left.

How this was gathered: the whole path was run once in the repository's own
environment, then re-run from a fresh clone, a fresh virtual environment and a
fresh working folder, typing only what [`START_HERE.md`](START_HERE.md) says.
The author is not a beginner, so the second pass is a simulation. A real
beginner has not tested these pages.

## Fixed in the tool

| # | What happened | Fix |
| --- | --- | --- |
| 1 | A charter run on the Titanic table crashed with `internal: TypeError('sequence item 2: expected str instance, float found')`. The near-duplicate screen joined text columns with a missing value under pandas 3. | `scmle/data/detectors/duplicates.py` fills missing text before joining; a test covers it. Commit "near-duplicate screen no longer crashes on missing text values under pandas 3". |
| 2 | `--group-column nosuch` (also `--time-column`, `--weight-column`) ended in a raw `KeyError` traceback from deep inside the splitter. | A clear `data error: group column 'nosuch' not found; columns: [...]` before the split. |
| 3 | `scmle resume` on a folder that does not exist printed a Python traceback. | Now `artifact error: run directory not found: ...`, exit 1. |
| 4 | A malformed `--leakage-attestations` file ended in a traceback (`plan`) or `internal error:` (`run`). | Now `config error: invalid leakage attestations: ...`. |
| 5 | A loop policy file containing only the one required key, `reference_risk`, failed with `KeyError: 'high_impact_actions'`. | `high_impact_actions` defaults to none. |

## Documented, not fixed (a decision is needed)

| # | What happened | Where it is explained | Suggested change |
| --- | --- | --- | --- |
| 6 | **Fixed (loudly, not by refusing).** The run on the Titanic table finished with an exam ROC AUC of 0.988. The last line of output says nothing about the leakage suspects. They are in the plan warnings and, at the bottom, in the "Findings" part of `report.html`. | [Reading results](READING_RESULTS.md), [Start here](START_HERE.md) Step 4 | Done: the default stays `warn` because a flag cannot be a verdict ([Leakage](LEAKAGE.md)). The run now logs a warning line at the start, labels the score `PROVISIONAL: N flagged column(s) not attested`, and opens `report.html` with the summary and findings. |
| 7 | A script that calls `MLProject.run()` without `if __name__ == "__main__":` prints a long "bootstrapping phase" error on macOS and Windows (helper processes re-import the script). The run can still finish. | [When things go wrong](WHEN_THINGS_GO_WRONG.md), Start here Step 8 | Catch this case and print a one-line hint. |
| 8 | **Fixed.** The run's last line now ends with `claims: N confirmed` (per tier) for a charter run, and the summary is the first line of `report.html`. Was: Tiers (`confirmed`, `attested`, `unverified`) are not shown in the run's summary line or in `report.html`; `product/README.md` said they were. | Reading results; `product/README.md` corrected | Show the claim counts per tier at the end of a charter run and in the report header. |
| 9 | **Fixed.** `report.html` now opens with the summary and the findings, then the leaderboard; the plan is folded at the bottom. Was: `report.html` begins with the whole plan (about 60 lines of techniques). The leaderboard is below it. | Reading results, "Which file to open first" | Put the result summary first and fold the plan. |
| 10 | With a short time budget the permutation-importance table is empty (`"truncated": true`), and `result.feature_importance()` returns an empty table. Nothing warns you. On the author's second machine the same 120 second command gave an empty table. | Start here Step 7 | Warn when a diagnostic was cut short, and name the budget. |
| 11 | With `--time-budget 60` on a busy computer, every trial failed with `trial exceeded its 6.0s hard time ceiling and was killed`, and the run ended with the bare message `all trials failed`. | When things go wrong | Say in the message that the budget is too small, and suggest a minimum. |
| 12 | The charter template (`scmle charter init`) writes `"intercurrent_events": []` and does not show that an item is `{"event": ..., "strategy": ...}`. The error for a wrong key names the wrong key but not the right ones. | Reading results | Put one worked item in the template. |
| 13 | `charter sign` rewrites the file as a wrapper (`algorithm`, `charter`, `charter_hash`, `signature`). A script that edits the charter after signing finds the fields one level deeper. | Reading results | Document the wrapper in `docs/CLI.md`. |
| 14 | `scmle charter check --rows N` takes development rows (rows left after the exam rows). Passing the full table size gave a more cheerful verdict (327 rows) than the run itself had (273). | Reading results, Decisions | Say so in the help text, or compute the exam rows. |
| 15 | The run's saved bundle is not signed. `scmle predict` on it refuses until you sign it by saving a copy with `signing_key=`. A beginner must write a small script for this. | Start here Step 8 | Sign during the run when `SCMLE_SIGNING_KEY` is set (the author did not test whether it already does), or add `scmle sign`. |
| 16 | The loop commands accept files only by hand-written CSV and JSON, and a reply to an escalation can only be created from Python (`Attestation.create`). Creating a reply is the one step a person must do. | When things go wrong, "How to reply" | Add a `scmle loop attest` command. |
| 17 | `loop mature` with a labels file missing `case_id` gives a raw `KeyError`. | When things go wrong | Check for the column and say which is missing. |
| 18 | The Titanic table has 14 percent exact duplicates. In a plain run they stay on both sides of the split, which can inflate the exam score; only a charter run keeps copies together. SCMLE does say so (`recommendations.json`), but only in a file. | Start here Step 7, Reading results | Print recommendations at the end of the run. |
| 19 | `scmle catalog` prints one long line of JSON. | Start here Step 2 shows what to look for | Pretty-print to the terminal. |
| 20 | The first `scmle run` on a new installation can be slow enough to hit the time ceiling in item 11. | When things go wrong | Warm up, or enforce a minimum. |
| 21 | In a charter run the label `PROVISIONAL: N flagged column(s) not attested` can still list a column that the person did attest, because the development-row screen flagged it, the charter's exploration-row screen did not, and the attestation was reported `unused`. | [Leakage](LEAKAGE.md) | Count a column as attested when the attestation file names it, or run one screen. |
| 22 | A claim's tier does not depend on the provenance of the column it uses: `worst radius` claims were `confirmed` while the column was flagged and not attested. That is how tiers work today (the tier is about the one-time test); changing it needs a design. | [Leakage](LEAKAGE.md) | Decide whether a claim on an unattested flagged column should be capped. |

## Things that worked on the first try

* Install from a fresh clone with `pip install .` (about 20 seconds with
  cached downloads) and the `scmle` command appeared.
* `plan`, the leakage block (exit 2), the real run, `predict`, signed bundle
  refusals, `serve` with its readiness endpoints, `package`, the registry
  approval and the rollback, the monitor and the loop commands all behaved as
  the code and docs say.
* The command-line run and the Python run, with the same seed and data, gave
  identical scores on the author's first machine.

## What I could not verify

* A real beginner. Everything above is from the author's own pass.
* A build of the container from `scmle package` (no Docker was run).
* A release against a live deployment. Only the local steps were run.
* `scmle frame` (the plain-request drafting) and `scmle ingest` (raw tables)
  were not exercised.
* A full charter run with a leakage attestation. The file format is taken
  from the source and the repository's own test.
* Windows. All commands were run on macOS.
