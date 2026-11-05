"""Lists what a ``destroy`` left behind, from listings the operator saves.

After ``terraform destroy`` an environment should leave nothing: not in the
state, and not in the account. This module reads two saved listings and
reports the leftovers; it calls no cloud API and runs no Terraform.

Listings it reads (JSON, as the provider tools print them):

* AWS: ``aws resourcegroupstaggingapi get-resources`` (a
  ``ResourceTagMappingList`` of entries with ``ResourceARN`` and ``Tags``).
  Pass ``--tag-filters`` for the environment, or list the whole account; the
  checker filters again.
* GCP: ``gcloud asset search-all-resources --format=json`` (a list of entries
  with ``name``, ``assetType``, ``labels`` and ``state``).
* The state: ``terraform state list`` output, one address per line.

A resource belongs to the environment when its ``scmle-environment`` tag or
label equals it, or, when untagged, its name contains ``<prefix>-<env>``
(reported as a suspect). A resource the provider keeps on purpose after a
destroy (a KMS key in its deletion window, a secret scheduled for deletion, a
GCP resource in ``DELETE_REQUESTED``) is reported as pending, not as an
orphan; pass those states in ``pending_states``.
"""

import argparse
from collections.abc import Iterable
from collections.abc import Mapping
import dataclasses
import json
import pathlib
import sys
from typing import Any

from scmle.contracts import errors

TAG = "scmle-environment"
PENDING_STATES = frozenset(
    {"PendingDeletion", "DELETE_REQUESTED", "DELETED", "DESTROYED"}
)
DEFAULT_PREFIX = "scmle"


@dataclasses.dataclass(frozen=True, slots=True)
class Leftover:
    """One resource that is still there."""

    identifier: str
    kind: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        """Returns a JSON-ready record."""
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class OrphanReport:
    """What remained after a destroy.

    Attributes:
        provider: ``aws`` or ``gcp``.
        environment: The destroyed environment.
        orphans: Resources that should be gone and are not.
        pending: Resources the provider removes later, by design.
        state: Addresses still in the Terraform state.
    """

    provider: str
    environment: str
    orphans: tuple[Leftover, ...]
    pending: tuple[Leftover, ...]
    state: tuple[str, ...]

    @property
    def clean(self) -> bool:
        """Whether nothing is left in the account or in the state."""
        return not self.orphans and not self.state

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-ready record."""
        return {
            "provider": self.provider,
            "environment": self.environment,
            "clean": self.clean,
            "orphans": [item.to_dict() for item in self.orphans],
            "pending": [item.to_dict() for item in self.pending],
            "state": list(self.state),
        }


def normalise_aws(listing: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Turns a tagging-API listing into the common record shape."""
    entries = listing.get("ResourceTagMappingList")
    if not isinstance(entries, list):
        raise errors.ArtifactError(
            "AWS listing has no ResourceTagMappingList list"
        )
    out = []
    for entry in entries:
        arn = str(entry["ResourceARN"])
        parts = arn.split(":")
        kind = parts[2] if len(parts) > 2 else "unknown"
        tags = {t["Key"]: t["Value"] for t in entry.get("Tags", [])}
        out.append(
            {
                "id": arn,
                "kind": kind,
                "tags": tags,
                "state": str(entry.get("State", "")),
            }
        )
    return out


def normalise_gcp(listing: Any) -> list[dict[str, Any]]:
    """Turns a ``search-all-resources`` listing into the common shape."""
    if not isinstance(listing, list):
        raise errors.ArtifactError("GCP listing must be a list")
    return [
        {
            "id": str(entry["name"]),
            "kind": str(entry.get("assetType", "unknown")),
            "tags": dict(entry.get("labels") or {}),
            "state": str(entry.get("state", "")),
        }
        for entry in listing
    ]


def find_orphans(
    provider: str,
    environment: str,
    listing: Any,
    state_lines: Iterable[str] = (),
    prefix: str = DEFAULT_PREFIX,
    pending_states: Iterable[str] = PENDING_STATES,
) -> OrphanReport:
    """Reports what is left of ``environment``.

    Args:
        provider: ``aws`` or ``gcp``.
        environment: Environment name.
        listing: Parsed provider listing (see the module docstring).
        state_lines: ``terraform state list`` lines after the destroy.
        prefix: Resource-name prefix, for untagged resources.
        pending_states: States that mean the provider will remove it.

    Raises:
        ArtifactError: For an unknown provider or an unreadable listing.
    """
    if provider == "aws":
        entries = normalise_aws(listing)
    elif provider == "gcp":
        entries = normalise_gcp(listing)
    else:
        raise errors.ArtifactError(f"unknown provider {provider!r}")
    waiting = set(pending_states)
    needle = f"{prefix}-{environment}"
    orphans: list[Leftover] = []
    pending: list[Leftover] = []
    for entry in entries:
        tagged = entry["tags"].get(TAG)
        if tagged is not None and tagged != environment:
            continue
        if tagged is None and needle not in entry["id"]:
            continue
        reason = (
            f"tagged {TAG}={environment}"
            if tagged is not None
            else f"untagged, name contains {needle}"
        )
        item = Leftover(entry["id"], entry["kind"], reason)
        if entry["state"] in waiting:
            pending.append(
                Leftover(
                    item.identifier,
                    item.kind,
                    f"{item.reason}; provider state {entry['state']}",
                )
            )
        else:
            orphans.append(item)
    state = tuple(line.strip() for line in state_lines if line.strip())
    return OrphanReport(
        provider, environment, tuple(orphans), tuple(pending), state
    )


def main(arguments: list[str] | None = None) -> int:
    """Command line: exit 0 only when the environment left nothing."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--provider", choices=["aws", "gcp"], required=True)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--listing", type=pathlib.Path, required=True)
    parser.add_argument("--state", type=pathlib.Path)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    options = parser.parse_args(arguments)
    try:
        listing = json.loads(options.listing.read_text(encoding="utf-8"))
        lines = (
            options.state.read_text(encoding="utf-8").splitlines()
            if options.state
            else []
        )
        report = find_orphans(
            options.provider,
            options.environment,
            listing,
            lines,
            options.prefix,
        )
    except (OSError, ValueError, KeyError, errors.ArtifactError) as error:
        print(f"orphan check failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0 if report.clean else 1
