"""Measurements and certificate for the framing methods.

Pre-registered before any certification run; a method that misses its floor
stays descriptive and nothing is retuned until it passes.

Intersection-union test (`baseline`). Clustered worlds: `G` clusters of mean
size 10, cluster prevalence Beta with mean 0.3 and intraclass correlation
0.1, the baseline with sensitivity 0.6 and false-positive rate 0.2, the
range `[0.1, 0.4]`. Null worlds: `null_hi` (gain zero at the upper end,
positive at the lower), `null_lo` (zero at the lower end, positive at the
upper) and `null_both` (zero everywhere). Policy: empirical rejection at most
`0.05 + 2 sqrt(0.05 * 0.95 / R)` with `R = 2000` replicates, at 20, 50 and
200 clusters. `MIN_CLUSTERS` is the smallest tested count from which every
tested count meets the bound.

Power. A world with a gain at both ends. The closed form is the standardised
gain over the cluster-robust standard error implied by `s_d` and the design
effect measured on a large sample of the same world; where its power at the
weaker end is at least 0.90, the empirical power of the claim must be at
least `0.80 - 2 sqrt(0.8 * 0.2 / R)`.

Selective labels (`framing-selective-labels`). 1000 rows. Plants: outcomes of
rows with `b = 0` observed with probability drawn from U(0, 0.3), rows with
`b = 1` fully observed. Clean worlds: outcomes missing completely at random
at a rate drawn from {0, U(0, 0.15)}. 500 of each. Policy: recall Wilson
lower bound at least 0.95, false-alarm Wilson upper bound at most 0.05, 95
percent intervals. The plants are strong; the certificate says nothing about
weak selection.

Feasibility. 600 exploration rows; the confirmation size is 0.6 or 1.6 times
the true requirement (away from the boundary). Policy: the verdict equals the
oracle's in at least 95 percent of 500 runs. At ratio 1.0 the share reported
`inconclusive` is recorded only.

Planted omission. 200 worlds where the cost threshold is unknown and the
verdict flips inside the plausible range (question expected), 200 where the
range leaves the verdict unchanged, 200 where the cost model is given.
Policy: question recall Wilson lower bound at least 0.80; false-question
Wilson upper bound at most 0.10 over the 400 worlds that need no question.

Seeds. Development seeds start at `DEV_SEED_BASE`; certification uses seeds
0 up to the replicate count. The ranges never overlap.

    python -m benchmarks.qualification.framing_certify --write
    python -m benchmarks.qualification.framing_certify --check
    python -m benchmarks.qualification.framing_certify --dev
"""

import argparse
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from scmle.contracts import qualification
from scmle.core import randomness
from scmle.data.charter import contracts as charter_contracts
from scmle.data.charter import feasibility as charter_feasibility
from scmle.data.detectors import certificates
from scmle.data.framing import baseline
from scmle.data.framing import costs
from scmle.data.framing import draft
from scmle.data.framing import feasibility

STORE_PATH = Path(__file__).with_name("framing_certificates.json")
MEASUREMENTS_PATH = Path(__file__).with_name("framing_measurements.json")
DEV_SEED_BASE = 10_000
ALPHA = 0.05
RUNS = 2000
CLUSTER_COUNTS = (20, 50, 200)
CLUSTER_SIZE = 10
PREVALENCE = 0.3
RHO = 0.1
RANGE = (0.1, 0.4)
BASE_SENSITIVITY = 0.6
BASE_FALSE_POSITIVE = 0.2
LABEL_ROWS = 1000
LABEL_RUNS = 500
LABEL_POLICY = certificates.CertificationPolicy(0.95, 0.05, 0.95)
FEASIBILITY_RUNS = 500
FEASIBILITY_ROWS = 600
FEASIBILITY_CLUSTERS = 60
FEASIBILITY_RATIOS = (0.6, 1.6)
FEASIBILITY_FLOOR = 0.95
OMISSION_RUNS = 200
OMISSION_RECALL_FLOOR = 0.80
OMISSION_FALSE_CEILING = 0.10
POWER_CLUSTERS = 100
POWER_ATTAINABLE = 0.90
POWER_FLOOR = 0.80
Z_95 = 1.6448536269514722


def size_bound(runs: int = RUNS) -> float:
    """Largest empirical rejection rate accepted for a level-0.05 test.

    Args:
        runs: Number of replicates the rate is observed over.

    Returns:
        ``0.05 + 2 * sqrt(0.05 * 0.95 / runs)``.
    """
    return ALPHA + 2 * math.sqrt(ALPHA * (1 - ALPHA) / runs)


def power_floor(runs: int = RUNS) -> float:
    """Smallest empirical power accepted where the closed form allows it.

    Args:
        runs: Number of replicates the power is observed over.

    Returns:
        ``0.80 - 2 * sqrt(0.8 * 0.2 / runs)``.
    """
    return POWER_FLOOR - 2 * math.sqrt(POWER_FLOOR * (1 - POWER_FLOOR) / runs)


def world(
    seed: int, clusters: int, se_a: float, fp_a: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One clustered world: outcomes, model actions, baseline, clusters.

    Args:
        seed: World seed.
        clusters: Number of clusters.
        se_a: Sensitivity of the model's action.
        fp_a: False-positive rate of the model's action.

    Returns:
        The outcome `y`, the model's action `a`, the baseline's action `b`
        (each 0/1 per row) and the cluster index of each row.
    """
    rng = randomness.component_rng(seed, "framing.world", clusters)
    concentration = (1 - RHO) / RHO
    prevalence = rng.beta(
        PREVALENCE * concentration,
        (1 - PREVALENCE) * concentration,
        size=clusters,
    )
    sizes = rng.integers(CLUSTER_SIZE // 2, CLUSTER_SIZE * 3 // 2 + 1, clusters)
    group = np.repeat(np.arange(clusters), sizes)
    y = (rng.random(group.size) < prevalence[group]).astype(float)
    a = (rng.random(group.size) < np.where(y == 1, se_a, fp_a)).astype(float)
    b = (
        rng.random(group.size)
        < np.where(y == 1, BASE_SENSITIVITY, BASE_FALSE_POSITIVE)
    ).astype(float)
    return y, a, b, group


def null_parameters(kind: str) -> tuple[float, float]:
    """Model sensitivity and false-positive rate for a null world.

    Args:
        kind: `null_both` (no gain anywhere), `null_hi` (no gain at the upper
            end of the range) or `null_lo` (no gain at the lower end).

    Returns:
        The model's sensitivity and false-positive rate.

    Raises:
        ValueError: If `kind` is not one of the three.
    """
    w_lo, w_hi = (costs.odds(t) for t in RANGE)
    if kind == "null_both":
        return BASE_SENSITIVITY, BASE_FALSE_POSITIVE
    if kind == "null_hi":
        d_se = 0.1
        d_fp = PREVALENCE * d_se / ((1 - PREVALENCE) * w_hi)
    elif kind == "null_lo":
        d_se = -0.1
        d_fp = PREVALENCE * d_se / ((1 - PREVALENCE) * w_lo)
    else:
        raise ValueError(f"unknown kind: {kind}")
    return BASE_SENSITIVITY + d_se, BASE_FALSE_POSITIVE + d_fp


def claim_p(seed: int, clusters: int, se_a: float, fp_a: float) -> float:
    """Intersection-union p-value of one world.

    Args:
        seed: World seed.
        clusters: Number of clusters.
        se_a: Sensitivity of the model's action.
        fp_a: False-positive rate of the model's action.

    Returns:
        The larger of the cluster-robust p-values at the two ends of
        `RANGE`.
    """
    y, a, b, group = world(seed, clusters, se_a, fp_a)
    return max(
        baseline.cluster_test(
            baseline.paired_difference(y, a, b, t), group
        ).p_value
        for t in RANGE
    )


def size_rows(offset: int, runs: int) -> dict[str, Any]:
    """Empirical size of the claim in every null world and cluster count.

    Args:
        offset: Added to the world seeds.
        runs: Replicates per world and cluster count.

    Returns:
        For each `kind/clusters` key, the rejections, runs, rate and whether
        the rate meets `size_bound(runs)`.
    """
    rows: dict[str, Any] = {}
    for kind in ("null_hi", "null_lo", "null_both"):
        se_a, fp_a = null_parameters(kind)
        for g in CLUSTER_COUNTS:
            rejected = sum(
                claim_p(offset + s, g, se_a, fp_a) <= ALPHA for s in range(runs)
            )
            rows[f"{kind}/{g}"] = {
                "clusters": g,
                "rejections": rejected,
                "runs": runs,
                "rate": rejected / runs,
                "meets_bound": rejected / runs <= size_bound(runs),
            }
    return rows


def minimum_clusters(rows: dict[str, Any]) -> int | None:
    """Smallest tested count from which every count meets the bound.

    Args:
        rows: Output of `size_rows`.

    Returns:
        The smallest cluster count such that it and all larger tested counts
        meet the bound in every null world; None if even the largest does
        not.
    """
    ok = {
        g: all(r["meets_bound"] for r in rows.values() if r["clusters"] == g)
        for g in CLUSTER_COUNTS
    }
    best = None
    for g in sorted(CLUSTER_COUNTS, reverse=True):
        if not ok[g]:
            break
        best = g
    return best


def closed_form_power(se_a: float, fp_a: float, clusters: int) -> float:
    """Power at the weaker end from `s_d`, the design effect and the gain.

    Args:
        se_a: Sensitivity of the model's action.
        fp_a: False-positive rate of the model's action.
        clusters: Number of clusters of the studied sample.

    Returns:
        The smaller normal-approximation power over the two ends of
        `RANGE` at one-sided level 0.05.
    """
    y, a, b, group = world(DEV_SEED_BASE, 4000, se_a, fp_a)
    worst = 1.0
    for t in RANGE:
        d = baseline.paired_difference(y, a, b, t)
        rho, m = feasibility.intraclass(d, group)
        deff = 1 + (m - 1) * rho
        n = clusters * CLUSTER_SIZE
        z = d.mean() * math.sqrt(n / deff) / d.std(ddof=1)
        worst = min(worst, float(stats.norm.cdf(z - Z_95)))
    return worst


def power_rows(offset: int, runs: int) -> dict[str, Any]:
    """Empirical power of the claim in a world with a gain at both ends.

    Args:
        offset: Added to the world seeds.
        runs: Number of replicates.

    Returns:
        The cluster count, closed-form and empirical power, the floor,
        whether the closed form is attainable and the floor is met.
    """
    se_a, fp_a = 0.72, 0.14
    rejected = sum(
        claim_p(offset + s, POWER_CLUSTERS, se_a, fp_a) <= ALPHA
        for s in range(runs)
    )
    closed = closed_form_power(se_a, fp_a, POWER_CLUSTERS)
    return {
        "clusters": POWER_CLUSTERS,
        "closed_form_power": closed,
        "attainable": closed >= POWER_ATTAINABLE,
        "empirical_power": rejected / runs,
        "floor": power_floor(runs),
        "meets_floor": rejected / runs >= power_floor(runs),
        "runs": runs,
    }


def label_world(planted: bool, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """(observed, b) of one world.

    Args:
        planted: Whether outcomes of rows with `b = 0` are selectively
            observed.
        seed: World seed.

    Returns:
        A boolean array of observed outcomes and the 0/1 array `b`.
    """
    rng = randomness.component_rng(seed, "framing.labels", int(planted))
    b = (rng.random(LABEL_ROWS) < 0.3).astype(float)
    if planted:
        keep = rng.uniform(0.0, 0.3)
        observed = (b == 1) | (rng.random(LABEL_ROWS) < keep)
    else:
        rate = 0.0 if rng.random() < 0.5 else rng.uniform(0.0, 0.15)
        observed = rng.random(LABEL_ROWS) >= rate
    return observed, b


def label_certificate(
    offset: int,
) -> tuple[certificates.DetectorCertificate, dict[str, Any]]:
    """Measures the selective-label detector.

    Args:
        offset: Added to the world seeds.

    Returns:
        The certificate and the measurements.
    """
    detected = sum(
        baseline.selective_labels(*label_world(True, offset + s)).flagged
        for s in range(LABEL_RUNS)
    )
    alarms = sum(
        baseline.selective_labels(*label_world(False, offset + s)).flagged
        for s in range(LABEL_RUNS)
    )
    result = qualification.Certificate(
        detector_id=baseline.DETECTOR_ID,
        defect_type=baseline.DEFECT_TYPE,
        planted=LABEL_RUNS,
        detected=detected,
        clean_rows=LABEL_RUNS,
        false_alarms=alarms,
        seed_path=randomness.seed_path(offset, "framing.certify.labels", 0),
    )
    cert = certificates.certify(
        baseline.DETECTOR_ID,
        code_digest(),
        certificates.corpus_hash(
            [
                certificates.hash_source(
                    json.dumps(
                        {
                            "rows": LABEL_ROWS,
                            "tolerance": baseline.LABEL_TOLERANCE,
                            "alpha": baseline.LABEL_ALPHA,
                            "min_arm_rows": baseline.MIN_ARM_ROWS,
                            "keep": [0.0, 0.3],
                            "clean_rate": [0.0, 0.15],
                        },
                        sort_keys=True,
                    )
                )
            ]
        ),
        [offset],
        [result],
        LABEL_POLICY,
    )
    return cert, {
        "planted": LABEL_RUNS,
        "detected": detected,
        "clean": LABEL_RUNS,
        "false_alarms": alarms,
        "tolerance": baseline.LABEL_TOLERANCE,
        "alpha": baseline.LABEL_ALPHA,
        "seed_offset": offset,
    }


def code_digest() -> str:
    """Hash of the selective-label detector, the certificate key.

    Returns:
        The code hash of `baseline.selective_labels`.
    """
    return certificates.code_hash(baseline.selective_labels)


def feasibility_run(seed: int, ratio: float) -> tuple[str, str]:
    """(estimated verdict, oracle verdict) of one run.

    Args:
        seed: World seed.
        ratio: Confirmation sample size as a multiple of the true
            requirement.

    Returns:
        The verdict from the estimated requirement and the oracle verdict
        (`feasible` for a ratio of at least 1, otherwise `downgrade`).
    """
    se_a, fp_a = 0.85, 0.12
    big = world(DEV_SEED_BASE, 4000, se_a, fp_a)
    d_big = baseline.paired_difference(*big[:3], RANGE[0])
    rho, m = feasibility.intraclass(d_big, big[3])
    deff = 1 + (m - 1) * rho
    s_true = float(d_big.std(ddof=1))
    delta_min = 0.04
    required = charter_feasibility.n_required_mean(
        delta_min / s_true, alpha=ALPHA, k_max=1, power=0.8
    )
    n_conf = round(ratio * required * deff)
    y, a, b, group = world(seed, FEASIBILITY_CLUSTERS, se_a, fp_a)
    d = baseline.paired_difference(y, a, b, RANGE[0])
    got = feasibility.assess(
        d,
        group,
        delta_min=delta_min,
        n_confirmation=n_conf,
        alpha=ALPHA,
        k_max=1,
        power=0.8,
    )
    oracle = "feasible" if ratio >= 1.0 else "downgrade"
    return got.verdict, oracle


def feasibility_rows(offset: int, runs: int) -> dict[str, Any]:
    """Agreement with the oracle away from the boundary, and at it.

    Args:
        offset: Added to the world seeds.
        runs: Runs per ratio.

    Returns:
        Agreements per ratio in `FEASIBILITY_RATIOS`, the overall agreement,
        whether it meets the floor and the share of `inconclusive` verdicts
        at ratio 1.
    """
    out: dict[str, Any] = {}
    agree = total = 0
    for ratio in FEASIBILITY_RATIOS:
        hits = sum(
            (lambda r: r[0] == r[1])(feasibility_run(offset + s, ratio))
            for s in range(runs)
        )
        out[f"ratio_{ratio}"] = {"agree": hits, "runs": runs}
        agree += hits
        total += runs
    inconclusive = sum(
        feasibility_run(offset + s, 1.0)[0] == feasibility.INCONCLUSIVE
        for s in range(runs)
    )
    out["agreement"] = agree / total
    out["meets_floor"] = agree / total >= FEASIBILITY_FLOOR
    out["boundary_inconclusive_share"] = inconclusive / runs
    return out


def omission_frame(seed: int, flips: bool) -> tuple[pd.DataFrame, np.ndarray]:
    """Exploration rows and pilot actions; the gain may flip in the range.

    Args:
        seed: World seed.
        flips: Whether the verdict flips inside the plausible cost range.

    Returns:
        The 600-row exploration table and the pilot's 0/1 actions.
    """
    rng = randomness.component_rng(seed, "framing.omission", int(flips))
    n = 600
    y = (rng.random(n) < 0.3).astype(int)
    if flips:
        pilot_rate = np.where(y == 1, 0.8, 0.45)
        base_rate = np.where(y == 1, 0.55, 0.15)
    else:
        pilot_rate = np.where(y == 1, 0.9, 0.05)
        base_rate = np.where(y == 1, 0.5, 0.3)
    pilot = (rng.random(n) < pilot_rate).astype(int)
    baseline_column = (rng.random(n) < base_rate).astype(int)
    frame = pd.DataFrame(
        {
            "customer_id": np.arange(n),
            "as_of": pd.date_range("2024-01-01", periods=n, freq="h"),
            "churn": y,
            "current_rule": baseline_column,
        }
    )
    return frame, pilot


def omission_asks(seed: int, flips: bool, given: bool) -> bool:
    """Whether the drafter raises the cost question for one request.

    Args:
        seed: World seed.
        flips: Whether the verdict flips inside the plausible cost range.
        given: Whether a cost model is supplied.

    Returns:
        True if the draft contains a question about the cost threshold.
    """
    frame, pilot = omission_frame(seed, flips)
    source = draft.SourceDescription(
        "warehouse://crm/accounts",
        tuple(frame.columns),
        primary_key=("customer_id",),
    )
    request = draft.FramingRequest("Predict churn for each customer.", source)
    model = (
        costs.DecisionCostModel(
            0.0,
            5.0,
            1.0,
            0.0,
            0.05,
            0.5,
            (costs.CostSource.HUMAN,) * 4,
            costs.CostSource.HUMAN,
        )
        if given
        else None
    )
    result = draft.frame(
        request,
        frame,
        cost_model=model,
        baseline_column="current_rule",
        pilot_actions=pilot.tolist(),
    )
    return any(q.slot == "cost_threshold" for q in result.questions)


def omission_rows(offset: int, runs: int) -> dict[str, Any]:
    """Question recall and false-question rate.

    Args:
        offset: Added to the world seeds.
        runs: Worlds per case.

    Returns:
        The counts of questions asked in each case, the Wilson lower bound of
        recall, the Wilson upper bound of false questions and whether both
        floors are met.
    """
    asked = sum(omission_asks(offset + s, True, False) for s in range(runs))
    robust = sum(omission_asks(offset + s, False, False) for s in range(runs))
    given = sum(omission_asks(offset + s, True, True) for s in range(runs))
    low = certificates.wilson_interval(asked, runs)[0]
    high = certificates.wilson_interval(robust + given, 2 * runs)[1]
    return {
        "runs_each": runs,
        "asked_when_critical": asked,
        "asked_when_robust": robust,
        "asked_when_given": given,
        "recall_wilson_lower": low,
        "false_question_wilson_upper": high,
        "meets_floor": low >= OMISSION_RECALL_FLOOR
        and high <= OMISSION_FALSE_CEILING,
    }


def fuzz_rows(count: int = 1000) -> dict[str, Any]:
    """Malformed charter records that `from_dict` wrongly accepts.

    Args:
        count: Number of malformed records to try.

    Returns:
        The number tried and the number accepted.
    """
    base = charter_contracts.template()
    base["question"] = "q"
    base["estimand"].update(
        population="p",
        target="y",
        prediction_moment="m",
        summary_measure="s",
        intercurrent_events=[{"event": "none", "strategy": "n/a"}],
    )
    base["unit"] = "u"
    base["deployment_population"] = "d"
    rng = np.random.default_rng(DEV_SEED_BASE)
    accepted = 0
    for number in range(count):
        record = json.loads(json.dumps(base))
        record["seed"] = number
        kind = rng.integers(0, 6)
        if kind == 0:
            del record[str(rng.choice(sorted(record)))]
        elif kind == 1:
            record[f"extra{rng.integers(1000)}"] = 1
        elif kind == 2:
            record[str(rng.choice(["question", "unit"]))] = ""
        elif kind == 3:
            record["power"] = float(rng.choice([0.0, 1.0, -1.0, 2.0]))
        elif kind == 4:
            record["estimand"]["kind"] = "other"
        else:
            record["k_max"] = float(rng.choice([0, -3]))
        try:
            charter_contracts.AnalysisCharter.from_dict(record)
            accepted += 1
        except (ValueError, KeyError, TypeError):
            pass
    return {"malformed": count, "accepted": accepted}


def certify_all(
    offset: int = 0, runs: int = RUNS
) -> tuple[certificates.CertificateStore, dict[str, Any]]:
    """Measures every framing method; `offset` shifts the seed range.

    Args:
        offset: Added to the world seeds.
        runs: Replicates per size and power cell.

    Returns:
        The certificate store and the measurements document.
    """
    started = time.monotonic()
    store = certificates.CertificateStore()
    cert, labels = label_certificate(offset)
    store.add(cert)
    sizes = size_rows(offset, runs)
    document = {
        "size": sizes,
        "size_bound": size_bound(runs),
        "min_clusters_measured": minimum_clusters(sizes),
        "min_clusters_in_code": baseline.MIN_CLUSTERS,
        "power": power_rows(offset, runs),
        "selective_labels": labels,
        "feasibility": feasibility_rows(offset, FEASIBILITY_RUNS),
        "planted_omission": omission_rows(offset, OMISSION_RUNS),
        "schema_fuzz": fuzz_rows(),
        "development_seed_base": DEV_SEED_BASE,
        "seconds": round(time.monotonic() - started, 1),
    }
    return store, document


def problems(
    store: certificates.CertificateStore, document: dict[str, Any] | None
) -> list[str]:
    """Why the files are unusable: stale, missing or inconsistent.

    Args:
        store: The certificate store.
        document: The stored measurements, or None when the file is absent.

    Returns:
        Messages for a missing or stale certificate, a missing measurements
        file, a `MIN_CLUSTERS` that differs from the measured minimum and an
        accepted malformed charter.
    """
    found = []
    report = certificates.audit(
        store, {baseline.DETECTOR_ID: (code_digest(), baseline.DEFECT_TYPE)}
    )
    found += [f"{d}: missing" for d in report.missing]
    found += [f"{d}: stale" for d in report.stale]
    if document is None:
        return [*found, "framing_measurements.json missing"]
    if document["min_clusters_measured"] != baseline.MIN_CLUSTERS:
        found.append(
            "baseline.MIN_CLUSTERS differs from the measured minimum "
            f"({document['min_clusters_measured']})"
        )
    if document["schema_fuzz"]["accepted"] != 0:
        found.append("a malformed charter record was accepted")
    return found


def descriptive_only(
    store: certificates.CertificateStore, document: dict[str, Any] | None
) -> list[str]:
    """Methods measured and not meeting their floor (not an error).

    Args:
        store: The certificate store.
        document: The stored measurements, or None.

    Returns:
        The names of methods that missed their floor.
    """
    names = []
    report = certificates.audit(
        store, {baseline.DETECTOR_ID: (code_digest(), baseline.DEFECT_TYPE)}
    )
    names += [f"{d}/{baseline.DEFECT_TYPE}" for d in report.uncertified]
    if document:
        if document["min_clusters_measured"] is None:
            names.append("baseline claim (size bound never met)")
        power = document["power"]
        if power["attainable"] and not power["meets_floor"]:
            names.append("baseline claim power")
        if not document["feasibility"]["meets_floor"]:
            names.append("framing feasibility verdict")
        if not document["planted_omission"]["meets_floor"]:
            names.append("framing questions")
    return names


def load_document() -> dict[str, Any] | None:
    """The committed measurements, or None.

    Returns:
        The parsed measurements file, or None when it does not exist.
    """
    if not MEASUREMENTS_PATH.exists():
        return None
    loaded: dict[str, Any] = json.loads(
        MEASUREMENTS_PATH.read_text(encoding="utf-8")
    )
    return loaded


def check() -> dict[str, Any]:
    """The `--check framing` report.

    Returns:
        `ok`, the problems, the descriptive-only methods and a note that no
        gold corpus exists.
    """
    store = certificates.CertificateStore.load(STORE_PATH)
    document = load_document()
    found = problems(store, document)
    return {
        "ok": not found,
        "problems": found,
        "descriptive_only": descriptive_only(store, document),
        "gold_corpus": "absent: no LLM-assisted field may gate",
    }


def main(arguments: list[str] | None = None) -> int:
    """Measures (`--write`), verifies (`--check`) or looks (`--dev`).

    Args:
        arguments: Command-line arguments; `None` reads `sys.argv`.

    Returns:
        With `--check`, 0 when the files are consistent, otherwise 1; with
        `--write` and `--dev`, always 0.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true")
    action.add_argument("--check", action="store_true")
    action.add_argument("--dev", action="store_true")
    options = parser.parse_args(arguments)
    if options.dev:
        document = certify_all(DEV_SEED_BASE, 300)[1]
        print(json.dumps(document, indent=2, sort_keys=True))
        return 0
    if options.check:
        report = check()
        print(json.dumps(report, indent=2))
        return 0 if report["ok"] else 1
    store, document = certify_all()
    store.save(STORE_PATH)
    MEASUREMENTS_PATH.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(document, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
