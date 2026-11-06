# Phase 10: Causal identification (F10), what is left

Names follow [`00-canon.md`](00-canon.md). Open items:
[`../product/features/F10-causal-identification.md`](../product/features/F10-causal-identification.md).

The charter block, selector and refusal, the five strategies, the battery,
replication, sensitivity, the tier rule, the stage that writes the report
files, the estimate on `C` inside the single opening and the simulation
certificates (`f10_certificates.json`, pre-registered floors, misses
published) are built; `CHANGELOG.md` has the record. The phase is not closed:
the gold-standard and reference-output evidence needs data and tools that are
not in the repository.

## Work packages still open

**WP-10.6 remainder** (one agent)
- Attestors in the charter so a run can reach `attested`. The estimate on `C`, the file hashes,
  the DAG load and the admissibility coupling are done.

**WP-10.7a: corpora** (one agent, can start at once)
- NSW with CPS and PSID comparison groups, Tennessee STAR, ACIC generators,
  IHDP: manifest entries (hash-verified) in
  `benchmarks/qualification/corpora.json`, all `candidate` until existence,
  licence and hash are checked; record the experimental estimate from the data
  at ingest.

**WP-10.7c: gold standard and references** (needs 10.6 and 10.7a, and R)
- Run the NSW comparison, publish the full table of (estimate, interval, tier,
  refusal) including failures; freeze `rdrobust`, `sensemakr`, `EValue`, `did`,
  `HonestDiD` and `Synth` outputs under `tests/fixtures/oracles/` with
  versions and hashes; build the real-covariate worlds; re-measure
  `coverage_aipw` (its boosted-learner world missed the pre-registered floor);
  verify or delete every literature claim tagged unverified.

## Exit gate still to pass

```bash
uv run python -m benchmarks.qualification --check f10    # built
uv run python -m benchmarks.qualification --f10-gold     # NSW, STAR, ACIC (not built)
```

- Gold standard: on NSW with each declared observational comparison group, no
  `confirmed` tier when the interval excludes the experimental estimate; the
  full table is published including the failures.
- Reference outputs equal the frozen references at the tolerances in F10
  acceptance 9.
- Reports are bit-identical on Python 3.11 and 3.14.
- Outside-party run of the gold-standard benchmark reproduces the table (party
  and date recorded; if it has not happened the phase is not closed).

## Evidence still required

Gold-standard table; frozen oracle manifest with tool versions and hashes;
outside-party report. (Selector tests, coverage and bias tables with Wilson
intervals, overlap curves, battery power curves with `delta_min`, the
sensitivity validity table and `f10_certificates.json` exist.)

## Risks that remain

- **Attested assumptions carry the claim.** An analyst can attest anything.
  The tier text says so; assumptions and attestors are in the claim file;
  `confirmed` needs a negative control, and the blind spot (a confounder the
  control does not share) is reported with its measured rate in the
  `false_confirm` gate, not hidden.
- **Benchmarks are synthetic or old.** NSW is small and heavily studied; ACIC
  generators are not the user's data. Report them as referents for operating
  characteristics, not as proof for any deployment.
- **Overfitting the gate to the benchmarks.** Floors are fixed before the run;
  held-out corpora are to be named before any tuning.
- **Frozen nuisances reduce efficiency.** Wider intervals are accepted.
- **Scope growth.** Each added strategy needs its own certificate (P1).

## Explicitly out of scope

Causal discovery and DAG learning, mediation, dynamic regimes beyond the
existing bounded estimator, interference and spillovers, continuous-treatment
effect curves, heterogeneity as a gate, decision policies built on the
estimate (F11), sequential monitoring (F11), the HTML report.
