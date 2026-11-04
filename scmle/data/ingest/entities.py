"""Entity resolution: Fellegi-Sunter with blocking, checked by sampled labels.

Records are compared on a field vector. For comparison level `k` of a field,
`m_k = Pr(level | match)` and `u_k = Pr(level | non-match)`; the pair weight is
`sum log2(m_k / u_k)` under conditional independence (Fellegi and Sunter 1969),
parameters by EM on the blocked pairs (Winkler's approach). Blocking limits
comparisons to pairs sharing a key in at least one of several passes and
reports the reduction ratio and, where truth is known, pair completeness.

The independence assumption is false in practice, so the model's own
probabilities are not trusted. Precision and recall are estimated from a
stratified clerical sample of pairs by score band, labelled by the benchmark
(validation) or a named reviewer (use), with Clopper-Pearson band intervals
carried through by simulation. Without labels the result is `unverified`
and the links serve only to prevent leakage: link-then-group joins every pair
above the lower threshold into one group, which errs toward keeping more rows
together, and never merges attribute values.

What matters to the instrument is cross-split link recall: the share of
true-match pairs that end in one group, because F2 splits by group.
"""

from collections.abc import Callable
from collections.abc import Sequence
import dataclasses
import itertools
import math
import re
from typing import Any

import numpy as np
import pandas as pd

from scmle.contracts import qualification
from scmle.core import randomness
from scmle.data.detectors import certificates

LEVELS = 3
MISSING = -1
DISAGREE = 0
SIMILAR = 1
AGREE = 2
SIMILAR_AT = 0.88
MAX_BLOCK = 200
BANDS = 10
CLEAN = re.compile(r"[^a-z0-9]+")
SOUNDEX_CODES = {
    **dict.fromkeys("bfpv", "1"),
    **dict.fromkeys("cgjkqsxz", "2"),
    **dict.fromkeys("dt", "3"),
    "l": "4",
    **dict.fromkeys("mn", "5"),
    "r": "6",
}
Pair = tuple[int, int]
Labeler = Callable[[int, int], bool]


def normalize(value: Any) -> str:
    """Lower-case alphanumerics of a value; empty for a missing one."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return CLEAN.sub("", str(value).lower())


def jaro(a: str, b: str) -> float:
    """Jaro similarity of two strings."""
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    reach = max(max(len(a), len(b)) // 2 - 1, 0)
    taken = [False] * len(b)
    left = []
    for i, ch in enumerate(a):
        for j in range(max(0, i - reach), min(len(b), i + reach + 1)):
            if not taken[j] and b[j] == ch:
                taken[j] = True
                left.append(ch)
                break
    if not left:
        return 0.0
    right = [b[j] for j in range(len(b)) if taken[j]]
    swaps = sum(x != y for x, y in zip(left, right, strict=True)) / 2
    m = len(left)
    return (m / len(a) + m / len(b) + (m - swaps) / m) / 3.0


def jaro_winkler(a: str, b: str) -> float:
    """Jaro-Winkler similarity with the usual 0.1 prefix scale."""
    base = jaro(a, b)
    prefix = 0
    for x, y in zip(a[:4], b[:4], strict=False):
        if x != y:
            break
        prefix += 1
    return base + prefix * 0.1 * (1.0 - base)


def soundex(text: str) -> str:
    """American Soundex of the letters in `text`; empty for no letters."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return ""
    out = letters[0].upper()
    last = SOUNDEX_CODES.get(letters[0], "")
    for ch in letters[1:]:
        code = SOUNDEX_CODES.get(ch, "")
        if code and code != last:
            out += code
        if ch not in "hw":
            last = code
    return (out + "000")[:4]


@dataclasses.dataclass(frozen=True, slots=True)
class Field:
    """A field to compare.

    Attributes:
        name: Column name.
        fuzzy: Use a three-level string comparison (disagree, similar,
            agree) instead of exact agreement.
    """

    name: str
    fuzzy: bool = True


@dataclasses.dataclass(frozen=True, slots=True)
class Pass:
    """One blocking pass.

    Attributes:
        field: Column the key is built from.
        kind: `exact`, `prefix` (first `size` characters) or `soundex`.
        size: Prefix length.
    """

    field: str
    kind: str = "exact"
    size: int = 3

    def key(self, value: Any) -> str:
        """Block key of one value; empty means the record is not blocked."""
        text = normalize(value)
        if not text:
            return ""
        if self.kind == "exact":
            return text
        if self.kind == "prefix":
            return text[: self.size]
        if self.kind == "soundex":
            return soundex(text)
        raise ValueError(f"unknown blocking kind: {self.kind}")


def candidate_pairs(
    records: pd.DataFrame, passes: Sequence[Pass], max_block: int = MAX_BLOCK
) -> tuple[list[Pair], int]:
    """Pairs sharing a block key in any pass, and blocks skipped as too big.

    A block larger than `max_block` is skipped whole and counted, because its
    pair count would dominate the work; the loss shows up in pair
    completeness, never silently.
    """
    pairs: set[Pair] = set()
    skipped = 0
    for rule in passes:
        blocks: dict[str, list[int]] = {}
        for index, value in enumerate(records[rule.field].tolist()):
            key = rule.key(value)
            if key:
                blocks.setdefault(key, []).append(index)
        for members in blocks.values():
            if len(members) > max_block:
                skipped += 1
                continue
            pairs.update(itertools.combinations(members, 2))
    return sorted(pairs), skipped


def compare(
    records: pd.DataFrame, fields: Sequence[Field], pairs: Sequence[Pair]
) -> np.ndarray:
    """Comparison level of every pair on every field, `MISSING` if empty."""
    columns = {f.name: [normalize(v) for v in records[f.name]] for f in fields}
    out = np.full((len(pairs), len(fields)), MISSING, dtype=np.int8)
    for row, (i, j) in enumerate(pairs):
        for col, field in enumerate(fields):
            a, b = columns[field.name][i], columns[field.name][j]
            if not a or not b:
                continue
            if a == b:
                out[row, col] = AGREE
            elif field.fuzzy and jaro_winkler(a, b) >= SIMILAR_AT:
                out[row, col] = SIMILAR
            else:
                out[row, col] = DISAGREE
    return out


@dataclasses.dataclass(frozen=True, slots=True)
class Model:
    """Fitted Fellegi-Sunter parameters.

    Attributes:
        prior: Estimated share of match pairs among compared pairs.
        m: `m[field][level]` for levels disagree, similar, agree.
        u: `u[field][level]`.
        iterations: EM iterations run.
    """

    prior: float
    m: np.ndarray
    u: np.ndarray
    iterations: int

    def weights(self, levels: np.ndarray) -> np.ndarray:
        """Log2 likelihood-ratio weight of each pair; missing fields add 0."""
        total = np.zeros(len(levels))
        for col in range(levels.shape[1]):
            seen = levels[:, col] != MISSING
            level = np.where(seen, levels[:, col], 0)
            ratio = np.log2(self.m[col][level] / self.u[col][level])
            total += np.where(seen, ratio, 0.0)
        return total


def fit_em(
    levels: np.ndarray, iterations: int = 100, tol: float = 1e-8
) -> Model:
    """EM for the two-class conditional-independence model.

    Starting values are fixed (m favours agreement, u favours disagreement),
    so the fit is deterministic. Parameters are floored away from 0 and 1.
    """
    pairs, fields = levels.shape
    m = np.tile([0.05, 0.15, 0.80], (fields, 1))
    u = np.tile([0.90, 0.07, 0.03], (fields, 1))
    prior = 0.05
    floor = 1e-6
    done = 0
    for step in range(1, iterations + 1):
        done = step
        log_m = np.full(pairs, math.log(prior))
        log_u = np.full(pairs, math.log(1.0 - prior))
        for col in range(fields):
            seen = levels[:, col] != MISSING
            level = np.where(seen, levels[:, col], 0)
            log_m += np.where(seen, np.log(m[col][level]), 0.0)
            log_u += np.where(seen, np.log(u[col][level]), 0.0)
        top = np.maximum(log_m, log_u)
        gamma = np.exp(log_m - top) / (
            np.exp(log_m - top) + np.exp(log_u - top)
        )
        new_prior = float(np.clip(gamma.mean(), floor, 1 - floor))
        new_m, new_u = m.copy(), u.copy()
        for col in range(fields):
            seen = levels[:, col] != MISSING
            for lv in range(LEVELS):
                hit = seen & (levels[:, col] == lv)
                gm, gu = gamma[seen].sum(), (1 - gamma[seen]).sum()
                new_m[col][lv] = gamma[hit].sum() / gm if gm > 0 else m[col][lv]
                new_u[col][lv] = (
                    (1 - gamma[hit]).sum() / gu if gu > 0 else u[col][lv]
                )
        new_m = np.clip(new_m, floor, 1.0)
        new_u = np.clip(new_u, floor, 1.0)
        shift = max(
            abs(new_prior - prior),
            float(np.abs(new_m - m).max()),
            float(np.abs(new_u - u).max()),
        )
        m, u, prior = new_m, new_u, new_prior
        if shift < tol:
            break
    return Model(prior, m, u, done)


def components(count: int, links: Sequence[Pair]) -> list[int]:
    """Group id of every record: connected components of the links."""
    parent = list(range(count))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in links:
        a, b = find(i), find(j)
        if a != b:
            parent[max(a, b)] = min(a, b)
    return [find(i) for i in range(count)]


@dataclasses.dataclass(frozen=True, slots=True)
class BandEstimate:
    """Clerical-sample result for one score band."""

    low: float
    high: float
    pairs: int
    sampled: int
    matches: int

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready record."""
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class Estimate:
    """Precision and recall of the link rule, with intervals.

    Attributes:
        precision: Point estimate and 95 percent interval of the share of
            linked pairs that are true matches.
        recall: Share of true matches among the compared pairs that are
            linked. Matches the blocking never compared are not in it.
        bands: The per-band sample.
        reviewer: Who labelled the sample.
    """

    precision: tuple[float, float, float]
    recall: tuple[float, float, float]
    bands: tuple[BandEstimate, ...]
    reviewer: str

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready record."""
        return {
            "precision": list(self.precision),
            "recall": list(self.recall),
            "bands": [b.to_dict() for b in self.bands],
            "reviewer": self.reviewer,
        }


def geometric_bands(order: np.ndarray, bands: int) -> list[np.ndarray]:
    """Splits ascending-ordered pairs into bands that thin out at the top.

    The top band holds the highest `2^-(bands-1)` share of the pairs, the next
    one twice that, and so on; the bottom band takes the rest. Matches are
    rare and sit at the top, so equal-count bands would hide them in one band.
    """
    total = len(order)
    edges = [0]
    for k in range(bands - 1, 0, -1):
        edges.append(total - total // 2**k if k > 0 else total)
    edges.append(total)
    edges = sorted(set(edges))
    return [order[a:b] for a, b in itertools.pairwise(edges) if b > a]


def stratified_estimate(
    weights: np.ndarray,
    pairs: Sequence[Pair],
    threshold: float,
    labeler: Labeler,
    reviewer: str,
    per_band: int,
    seed: int,
    bands: int = BANDS,
    draws: int = 2000,
) -> Estimate:
    """Precision and recall of `weight >= threshold` from a stratified sample.

    Pairs are cut into `bands` equal-count score bands; `per_band` pairs are
    drawn in each (all, if fewer) and labelled by `labeler`. Each band's match
    rate gets a Beta posterior with the Jeffreys prior and the intervals come
    from 2000 seeded joint draws, which is an approximation, not exact.
    """
    rng = randomness.component_rng(seed, "ingest.entities.clerical", 0)
    order = np.argsort(weights, kind="stable")
    cuts = geometric_bands(order, bands)
    sizes, hits, taken, above = [], [], [], []
    summary = []
    for members in cuts:
        if len(members) == 0:
            continue
        pick = rng.choice(
            len(members), min(per_band, len(members)), replace=False
        )
        labelled = [labeler(*pairs[int(members[k])]) for k in pick]
        sizes.append(len(members))
        hits.append(sum(labelled))
        taken.append(len(labelled))
        above.append(bool(weights[members].min() >= threshold))
        summary.append(
            BandEstimate(
                float(weights[members].min()),
                float(weights[members].max()),
                len(members),
                len(labelled),
                sum(labelled),
            )
        )
    n = np.array(sizes, dtype=float)
    k = np.array(hits, dtype=float)
    t = np.array(taken, dtype=float)
    mask = np.array(above)
    rates = rng.beta(k + 0.5, t - k + 0.5, size=(draws, len(n)))
    matches = rates * n
    linked_true = matches[:, mask].sum(axis=1)
    linked = n[mask].sum()
    total_true = matches.sum(axis=1)
    p = linked_true / linked if linked else np.zeros(draws)
    r = np.divide(
        linked_true, total_true, out=np.zeros(draws), where=total_true > 0
    )
    point_rate = (k + 0.5) / (t + 1.0)
    point_true = point_rate * n
    point_p = point_true[mask].sum() / linked if linked else 0.0
    point_r = (
        point_true[mask].sum() / point_true.sum() if point_true.sum() else 0.0
    )
    lo, hi = np.percentile(p, [2.5, 97.5])
    rlo, rhi = np.percentile(r, [2.5, 97.5])
    return Estimate(
        (float(point_p), float(lo), float(hi)),
        (float(point_r), float(rlo), float(rhi)),
        tuple(summary),
        reviewer,
    )


@dataclasses.dataclass(frozen=True, slots=True)
class Resolution:
    """Result of resolving a record set.

    Attributes:
        groups: Group id per record (link-then-group); use as F2's group
            column.
        links: Pairs at or above the grouping threshold.
        t_group: Weight threshold used for grouping.
        t_match: Weight above which a pair is called a match.
        model: Fitted parameters.
        compared: Pairs compared after blocking.
        reduction_ratio: Share of all `N(N-1)/2` pairs avoided.
        skipped_blocks: Blocks too large to compare.
        estimate: Clerical-sample estimate, or None when unlabelled.
        tier: `attested` with a named reviewer's sample, else `unverified`.
    """

    groups: list[int]
    links: tuple[Pair, ...]
    t_group: float
    t_match: float
    model: Model
    compared: int
    reduction_ratio: float
    skipped_blocks: int
    estimate: Estimate | None
    tier: qualification.Tier

    def to_dict(self) -> dict[str, Any]:
        """`entity_resolution.json`."""
        return {
            "tier": self.tier.value,
            "t_group": self.t_group,
            "t_match": self.t_match,
            "compared_pairs": self.compared,
            "reduction_ratio": self.reduction_ratio,
            "skipped_blocks": self.skipped_blocks,
            "links": len(self.links),
            "groups": len(set(self.groups)),
            "records": len(self.groups),
            "model": {
                "prior": self.model.prior,
                "m": self.model.m.tolist(),
                "u": self.model.u.tolist(),
                "iterations": self.model.iterations,
            },
            "estimate": None
            if self.estimate is None
            else self.estimate.to_dict(),
        }


def resolve(
    records: pd.DataFrame,
    fields: Sequence[Field],
    passes: Sequence[Pass],
    labeler: Labeler | None = None,
    reviewer: str = "",
    per_band: int = 60,
    min_precision: float = 0.95,
    seed: int = 0,
) -> Resolution:
    """Resolves entities within one record set.

    Args:
        records: One row per record (stack two sources before calling).
        fields: Fields compared.
        passes: Blocking passes.
        labeler: Gives the true match status of a record pair; with it the
            thresholds come from the clerical sample, without it they come
            from the model's weights and the result is `unverified`.
        reviewer: Who labels; required with a labeler.
        per_band: Pairs sampled per score band.
        min_precision: Target precision of the match call (`1 - mu`).
        seed: Seed of the sample.

    Returns:
        The resolution. Grouping uses the lower threshold `t_group`, the
        smallest weight at which sampled precision is still above half, so
        possible matches are grouped too.
    """
    if labeler is not None and not reviewer.strip():
        raise ValueError("a labeler needs a named reviewer")
    count = len(records)
    pairs, skipped = candidate_pairs(records, passes)
    total = count * (count - 1) / 2
    reduction = 1.0 - len(pairs) / total if total else 1.0
    if not pairs:
        model = Model(
            0.0,
            np.ones((len(fields), LEVELS)),
            np.ones((len(fields), LEVELS)),
            0,
        )
        return Resolution(
            list(range(count)),
            (),
            math.inf,
            math.inf,
            model,
            0,
            reduction,
            skipped,
            None,
            qualification.Tier.UNVERIFIED,
        )
    levels = compare(records, fields, pairs)
    model = fit_em(levels)
    weights = model.weights(levels)
    estimate = None
    tier = qualification.Tier.UNVERIFIED
    if labeler is not None:
        estimate_at = stratified_estimate(
            weights, pairs, -math.inf, labeler, reviewer, per_band, seed
        )
        t_match, t_group = thresholds_from_bands(
            estimate_at.bands, min_precision
        )
        estimate = stratified_estimate(
            weights, pairs, t_match, labeler, reviewer, per_band, seed
        )
        tier = qualification.Tier.ATTESTED
    else:
        posterior_odds = (
            2.0 ** np.clip(weights, -60.0, 60.0)
            * model.prior
            / (1.0 - model.prior)
        )
        posterior = posterior_odds / (1.0 + posterior_odds)
        t_match = lowest_weight(weights, posterior >= min_precision)
        t_group = lowest_weight(weights, posterior >= 0.5)
    t_group = min(t_group, t_match)
    links = tuple(
        p for p, w in zip(pairs, weights, strict=True) if w >= t_group
    )
    return Resolution(
        components(count, links),
        links,
        t_group,
        t_match,
        model,
        len(pairs),
        reduction,
        skipped,
        estimate,
        tier,
    )


def lowest_weight(weights: np.ndarray, keep: np.ndarray) -> float:
    """Smallest weight among selected pairs; infinity when none."""
    chosen = weights[keep]
    return float(chosen.min()) if len(chosen) else math.inf


def thresholds_from_bands(
    bands: Sequence[BandEstimate], min_precision: float
) -> tuple[float, float]:
    """`(t_match, t_group)` from the labelled bands.

    Bands are scanned from the top. `t_match` is the lowest band edge such
    that the cumulative sampled precision from the top stays at or above
    `min_precision`; `t_group` the lowest edge whose own sampled match rate
    is at least one half.
    """
    t_match = math.inf
    t_group = math.inf
    linked = true = 0.0
    for band in reversed(bands):
        rate = band.matches / band.sampled if band.sampled else 0.0
        linked += band.pairs
        true += rate * band.pairs
        if linked and true / linked >= min_precision:
            t_match = band.low
        if rate >= 0.5:
            t_group = band.low
    return t_match, t_group


@dataclasses.dataclass(frozen=True, slots=True)
class Quality:
    """Measures against known truth (benchmarks only).

    Attributes:
        pair_completeness: Share of true-match pairs the blocking compared.
        link_recall: Share of true-match pairs placed in one group.
        link_recall_interval: Wilson interval of `link_recall`.
        true_pairs: True-match pairs.
        grouped_pairs: True-match pairs found in one group.
        false_group_pairs: Non-match pairs placed in one group.
    """

    pair_completeness: float
    link_recall: float
    link_recall_interval: tuple[float, float]
    true_pairs: int
    grouped_pairs: int
    false_group_pairs: int


def measure_quality(
    entity: Sequence[Any], resolution: Resolution, pairs: Sequence[Pair]
) -> Quality:
    """Compares groups with the true entity of each record.

    `pairs` are the compared pairs, to give pair completeness.
    """
    by_entity: dict[Any, list[int]] = {}
    for index, key in enumerate(entity):
        by_entity.setdefault(key, []).append(index)
    truth = {
        pair
        for members in by_entity.values()
        for pair in itertools.combinations(members, 2)
    }
    grouped = sum(
        resolution.groups[i] == resolution.groups[j] for i, j in truth
    )
    compared = set(pairs)
    completeness = len(truth & compared) / len(truth) if truth else 1.0
    by_group: dict[int, list[int]] = {}
    for index, group in enumerate(resolution.groups):
        by_group.setdefault(group, []).append(index)
    false_pairs = sum(
        entity[i] != entity[j]
        for members in by_group.values()
        for i, j in itertools.combinations(members, 2)
    )
    recall = grouped / len(truth) if truth else 1.0
    interval = (
        certificates.wilson_interval(grouped, len(truth))
        if truth
        else (0.0, 1.0)
    )
    return Quality(
        completeness, recall, interval, len(truth), grouped, false_pairs
    )
