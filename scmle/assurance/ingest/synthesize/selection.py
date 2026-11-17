"""Selects the kept features and registers them in the ledger.

Label-free gates (cap, as-of rule, strictness, missing rate, constant and
duplicate) decide what is kept; the kept set fixes the family size the
ledger corrects for.
"""

from collections.abc import Sequence
import dataclasses
import datetime
import hashlib
from typing import Any

import numpy
import pandas

from scmle.assurance.admissibility import provenance
from scmle.assurance.ingest import asof_view
from scmle.assurance.ingest.synthesize.evaluation import evaluate
from scmle.assurance.ingest.synthesize.grammar import Feature
from scmle.assurance.ingest.synthesize.grammar import Nest
from scmle.assurance.ingest.synthesize.grammar import propose
from scmle.assurance.ingest.synthesize.grammar import Source
from scmle.assurance.ingest.synthesize.grammar import WINDOWS_DAYS
import scmle.assurance.ledger.ledger
from scmle.contracts import errors
from scmle.contracts import qualification

FAMILY = "synthesized"


STATISTIC = "spearman"


class MultiplicityError(ValueError):
    """The declared family size differs from the registered hypotheses."""


@dataclasses.dataclass(frozen=True, slots=True)
class Selection:
    """Kept features, the manifest, and the declared family size.

    Attributes:
        frame: One column per kept feature, indexed like the targets.
        manifest: One record per proposed feature, kept or not, with the
            reason it was dropped.
        declared_family_size: Number of kept features. Fixed here, before any
            feature meets the target.
    """

    frame: pandas.DataFrame
    manifest: tuple[dict[str, Any], ...]
    declared_family_size: int

    def kept(self) -> list[dict[str, Any]]:
        """Selects the manifest records of the kept features.

        Returns:
            The records whose `kept` flag is true, in manifest order.
        """
        return [m for m in self.manifest if m["kept"]]

    def to_dict(self) -> dict[str, Any]:
        """Serializes the set, for `feature_manifest.json`.

        Returns:
            A JSON-ready mapping with the declared family size, the number
            of proposals and the manifest.
        """
        return {
            "declared_family_size": self.declared_family_size,
            "proposed": len(self.manifest),
            "features": list(self.manifest),
        }


def fingerprint(column: pandas.Series) -> str:
    """Digests a feature column.

    Args:
        column: The feature values.

    Returns:
        A SHA-256 hex digest, equal for identical values (NaN equals NaN).
    """
    values = column.to_numpy()
    if values.dtype == object:
        text = repr(list(values)).encode("utf-8")
    else:
        text = numpy.nan_to_num(values.astype(float), nan=-1.2345e300).tobytes()
    return hashlib.sha256(text).hexdigest()


def describe(
    spec: Feature,
    sources: Sequence[Source],
    nests: Sequence[Nest],
    lag_days: float,
) -> tuple[dict[str, Any], list[asof_view.View], qualification.Tier]:
    """Opens the manifest record of one proposal, before any gate.

    Args:
        spec: The proposed feature.
        sources: Related tables as views.
        nests: Depth-2 routes.
        lag_days: Gap between the cutoff and the moment.

    Returns:
        The record (not kept, no reason yet), the views the feature reads
        and the weakest tier among them.
    """
    views = [sources[spec.source].view]
    if spec.nest is not None:
        views.append(nests[spec.nest].leaf)
    tier = asof_view.weakest([v.tier for v in views])
    return (
        {
            "name": spec.name,
            "kind": spec.kind,
            "expression": spec.expression(),
            "tables": [v.name for v in views],
            "cutoff_rule": f"known_time <= moment - {lag_days:g}d",
            "as_of": [v.evidence() for v in views],
            "tier": tier.value,
            "kept": False,
            "dropped": "",
        },
        views,
        tier,
    )


def synthesize(
    targets: pandas.DataFrame,
    moment: str,
    sources: Sequence[Source],
    nests: Sequence[Nest] = (),
    lag_days: float = 0.0,
    windows: Sequence[float | None] = WINDOWS_DAYS,
    max_features: int = 200,
    max_null_rate: float = 0.95,
    strict: bool = False,
) -> Selection:
    """Builds the feature family for `targets` through the as-of views.

    Args:
        targets: One row per prediction case.
        moment: Column holding each case's prediction moment.
        sources: Related tables as views; the keys are columns of `targets`.
        nests: Depth-2 routes; each `mid` must be one of `sources`.
        lag_days: Gap between the cutoff and the moment (`t_i - lag`).
        windows: Declared windows in days; None means all visible rows.
        max_features: Cap on proposals. Proposals beyond it are dropped in
            grammar order and listed.
        max_null_rate: Features missing more often are dropped.
        strict: Drop features whose weakest source tier is `unverified`.

    Returns:
        The feature set. Every dropped proposal is in the manifest with its
        reason: `cap`, `no as-of rule: <tables>`, `unverified (strict)`,
        `mostly missing`, `constant` or `duplicate of <name>`.

    Raises:
        ValueError: If `lag_days` is negative.
    """
    if lag_days < 0:
        raise errors.InputError("lag_days must be non-negative")
    specs = propose(sources, nests, windows)
    columns: dict[str, pandas.Series] = {}
    manifest: list[dict[str, Any]] = []
    seen: dict[str, str] = {}
    for position, spec in enumerate(specs):
        record, views, tier = describe(spec, sources, nests, lag_days)
        manifest.append(record)
        if position >= max_features:
            record["dropped"] = "cap"
            continue
        blocked = [
            v.name
            for v in views
            if v.rule.mode == asof_view.NON_VERSIONED and not v.permissive
        ]
        if blocked:
            record["dropped"] = f"no as-of rule: {blocked}"
            continue
        if strict and tier is qualification.Tier.UNVERIFIED:
            record["dropped"] = "unverified (strict)"
            continue
        values = evaluate(spec, targets, moment, sources, nests, lag_days)
        values.index = targets.index
        values.name = spec.name
        if values.isna().mean() > max_null_rate:
            record["dropped"] = "mostly missing"
            continue
        if values.nunique(dropna=False) <= 1:  # noqa: PD101 -- counts NaN as a value and accepts an empty series, which an equality check cannot reproduce
            record["dropped"] = "constant"
            continue
        digest = fingerprint(values)
        if digest in seen:
            record["dropped"] = f"duplicate of {seen[digest]}"
            continue
        seen[digest] = spec.name
        record["kept"] = True
        columns[spec.name] = values
    frame = pandas.DataFrame(columns, index=targets.index)
    return Selection(frame, tuple(manifest), len(columns))


def lineage(
    features: Selection,
    lag_days: float,
    views: Sequence[asof_view.View],
    today: datetime.date | None = None,
) -> provenance.Dictionary:
    """Builds the provenance table of the kept features.

    Each feature inherits the weakest source among the tables it reads. A
    feature whose weakest source is unverified gets an unknown as-of rule, so
    admissibility tiers it `unverified` and strict mode excludes it.

    Args:
        features: The feature set.
        lag_days: Lag used in synthesis.
        views: Every view that was used, to look up each table's rule.
        today: Day on which attestations must be valid.

    Returns:
        One record per kept feature, in manifest order.
    """
    by_name = {v.name: v for v in views}
    records = []
    for entry in features.kept():
        sources = [by_name[t] for t in entry["tables"]]
        weakest_view = min(
            sources, key=lambda v: asof_view.TIER_ORDER[v.rule.tier(today)]
        )
        rule = weakest_view.rule
        tier = rule.tier(today)
        history = (f"{entry['expression']}; {entry['cutoff_rule']}",)
        attestation = rule.attestation
        if tier is qualification.Tier.CONFIRMED:
            as_of = provenance.Lead(
                provenance.Cutoff.TIMESTAMP, lag_days * 86_400.0
            )
            source = provenance.Provenance.LOG
        elif tier is qualification.Tier.ATTESTED:
            as_of = provenance.Lead(provenance.Cutoff.LAG, lag_days * 86_400.0)
            source = provenance.Provenance.ATTESTED
        else:
            as_of = provenance.Lead.unknown()
            source = provenance.Provenance.NONE
            attestation = None
        records.append(
            provenance.Pedigree(
                name=entry["name"],
                origin=provenance.Origin.DERIVED,
                writer="scmle.assurance.ingest.synthesize",
                written_at=provenance.Stamp.UNKNOWN,
                overwritten=provenance.Overwritten.NEVER,
                definition_history=history,
                as_of=as_of,
                as_of_source=source,
                available_at_serving=provenance.Availability.UNKNOWN,
                attestation=attestation,
                log_check=rule.log_check
                if tier is qualification.Tier.CONFIRMED
                else None,
            )
        )
    return provenance.Dictionary(tuple(records))


def enlist(
    ledger: scmle.assurance.ledger.ledger.Ledger, features: Selection
) -> list[scmle.assurance.ledger.ledger.Hypothesis]:
    """Registers every kept feature as a `synthesized` hypothesis.

    If the number registered then differs from the declared size,
    `plurality` raises `MultiplicityError`, which propagates.

    Args:
        ledger: The hypothesis ledger.
        features: The feature set whose kept features are registered.

    Returns:
        The registered hypotheses, in manifest order.
    """
    registered = [
        ledger.register(
            scmle.assurance.ledger.ledger.Test.create(
                name=f"{entry['name']}~target",
                family=FAMILY,
                statistic=STATISTIC,
                null=(
                    f"{entry['name']} is not monotonically associated "
                    "with the target"
                ),
                column=entry["name"],
            )
        )
        for entry in features.kept()
    ]
    plurality(features.declared_family_size, ledger)
    return registered


def plurality(
    declared: int, ledger: scmle.assurance.ledger.ledger.Ledger
) -> int:
    """Requires the registered `synthesized` count to equal the declared one.

    Args:
        declared: The declared family size.
        ledger: The hypothesis ledger.

    Returns:
        The number of `synthesized` hypotheses registered.

    Raises:
        MultiplicityError: On a mismatch; the run must fail.
    """
    registered = sum(
        h.spec.family == FAMILY for h in ledger.hypotheses.values()
    )
    if registered != declared:
        raise MultiplicityError(
            f"declared family size {declared} but {registered} "
            f"{FAMILY} hypotheses are registered"
        )
    return registered
