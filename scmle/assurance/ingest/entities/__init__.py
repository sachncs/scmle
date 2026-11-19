"""Entity resolution: Fellegi-Sunter with blocking, checked by sampled labels.

Records are compared on a field vector. For comparison level `k` of a field,
`m_k = Pr(level | match)` and `u_k = Pr(level | non-match)`; the pair weight is
`sum log2(m_k / u_k)` under conditional independence (Fellegi and Sunter 1969),
parameters by EM on the blocked pairs (Winkler's approach). Not verified
against the primary sources: Fellegi and Sunter (JASA 64, 1969) could not be
retrieved and Winkler's EM paper could not be read. A search on 2026-11-05
found secondary descriptions (Winkler's census working papers, an ABS note)
that write the field weight as `log2(m / u)` summed over fields and classify
with an upper cutoff and a clerical-review band; the conditional-independence
assumption and the EM details are from memory. Blocking limits
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
true-match pairs that end in one group, because the evaluation splits the
data by group.
"""

from collections.abc import Sequence

import numpy
import pandas

from scmle.assurance.ingest.entities.blocking import Field
from scmle.assurance.ingest.entities.blocking import MAX_BLOCK
from scmle.assurance.ingest.entities.blocking import Pair
from scmle.assurance.ingest.entities.blocking import pairing
from scmle.assurance.ingest.entities.blocking import Pass
from scmle.assurance.ingest.entities.fitting import AGREE
from scmle.assurance.ingest.entities.fitting import components
from scmle.assurance.ingest.entities.fitting import DISAGREE
from scmle.assurance.ingest.entities.fitting import em
from scmle.assurance.ingest.entities.fitting import LEVELS
from scmle.assurance.ingest.entities.fitting import MISSING
from scmle.assurance.ingest.entities.fitting import Model
from scmle.assurance.ingest.entities.fitting import SIMILAR
from scmle.assurance.ingest.entities.fitting import SIMILAR_AT
from scmle.assurance.ingest.entities.resolution import grade
from scmle.assurance.ingest.entities.resolution import isolate
from scmle.assurance.ingest.entities.resolution import Quality
from scmle.assurance.ingest.entities.resolution import Resolution
from scmle.assurance.ingest.entities.sampling import appraise
from scmle.assurance.ingest.entities.sampling import Band
from scmle.assurance.ingest.entities.sampling import BANDS
from scmle.assurance.ingest.entities.sampling import calibrate
from scmle.assurance.ingest.entities.sampling import cutoffs
from scmle.assurance.ingest.entities.sampling import Estimate
from scmle.assurance.ingest.entities.sampling import GROUP_PROBABILITY
from scmle.assurance.ingest.entities.sampling import infer
from scmle.assurance.ingest.entities.sampling import Labeler
from scmle.assurance.ingest.entities.sampling import stratify
from scmle.assurance.ingest.entities.sampling import weakest
from scmle.assurance.ingest.entities.similarity import CLEAN
from scmle.assurance.ingest.entities.similarity import jaro
from scmle.assurance.ingest.entities.similarity import jarowinkler
from scmle.assurance.ingest.entities.similarity import normalize
from scmle.assurance.ingest.entities.similarity import soundex
from scmle.assurance.ingest.entities.similarity import SOUNDEX_CODES
from scmle.contracts import errors

__all__ = [
    "AGREE",
    "BANDS",
    "CLEAN",
    "DISAGREE",
    "GROUP_PROBABILITY",
    "LEVELS",
    "MAX_BLOCK",
    "MISSING",
    "SIMILAR",
    "SIMILAR_AT",
    "SOUNDEX_CODES",
    "Band",
    "Estimate",
    "Field",
    "Labeler",
    "Model",
    "Pair",
    "Pass",
    "Quality",
    "Resolution",
    "appraise",
    "calibrate",
    "components",
    "cutoffs",
    "em",
    "grade",
    "infer",
    "isolate",
    "jaro",
    "jarowinkler",
    "normalize",
    "pairing",
    "soundex",
    "stratify",
    "weakest",
]


def compare(
    records: pandas.DataFrame, fields: Sequence[Field], pairs: Sequence[Pair]
) -> numpy.ndarray:
    """Compares every pair on every field.

    Args:
        records: One row per record.
        fields: Fields compared.
        pairs: Record-index pairs.

    Returns:
        An int8 array of shape (pairs, fields) holding `AGREE`, `SIMILAR`
        or `DISAGREE`, or `MISSING` where either value is empty.
    """
    columns = {f.name: [normalize(v) for v in records[f.name]] for f in fields}
    out = numpy.full((len(pairs), len(fields)), MISSING, dtype=numpy.int8)
    for row, (i, j) in enumerate(pairs):
        for col, field in enumerate(fields):
            a, b = columns[field.name][i], columns[field.name][j]
            if not a or not b:
                continue
            if a == b:
                out[row, col] = AGREE
            elif field.fuzzy and jarowinkler(a, b) >= SIMILAR_AT:
                out[row, col] = SIMILAR
            else:
                out[row, col] = DISAGREE
    return out


def resolve(
    records: pandas.DataFrame,
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

    Raises:
        ValueError: If a labeler is given without a non-blank reviewer.
    """
    if labeler is not None and not reviewer.strip():
        raise errors.InputError("a labeler needs a named reviewer")
    count = len(records)
    pairs, skipped = pairing(records, passes)
    total = count * (count - 1) / 2
    reduction = 1.0 - len(pairs) / total if total else 1.0
    if not pairs:
        return isolate(count, len(fields), reduction, skipped)
    levels = compare(records, fields, pairs)
    model = em(levels)
    weights = model.weights(levels)
    t_match, t_group, estimate, tier = calibrate(
        weights,
        pairs,
        model.prior,
        labeler,
        reviewer,
        per_band,
        min_precision,
        seed,
    )
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
