"""The `records` command: moves stored records to the current schema.

`scmle records upgrade FOLDER` rewrites every record it knows in the folder
(and its sub-folders) at the current schema version, through each record
kind's own migrations: the charter, the split and dataset manifests, the
feasibility, error budget and cost model records and the claims file. Every
hash and signature that names one of these is anchored at schema version 1,
so it still verifies after the rewrite. A record whose exact bytes are bound
by a digest elsewhere is left as stored (it still loads through its
migrations): the hash-chained logs (`*.jsonl`), the negative control (its
digest is in the ledger) and every file of a sealed trail (a folder holding
`trail.json`).

The stored bytes are verified before anything is written and again after:
each record must load (which checks its seal), the charter's hash and
signature must verify, and a charter run's ledger chain, custody log and
claims must pass `claims verify`. Each rewritten file keeps its old bytes
beside it as `NAME.bak`; if the second verification fails, every file is
restored from its `.bak` and the command exits 1.
"""

import argparse
from collections.abc import Mapping
import json
import logging
import pathlib
from typing import Any

from scmle import durability
from scmle.artifacts import claims_audit
from scmle.assurance.charter import contracts
from scmle.assurance.charter import feasibility
from scmle.assurance.charter import sealing
from scmle.assurance.custody import manifest
from scmle.assurance.framing import costs
from scmle.assurance.ledger import budget
from scmle.cli import command
from scmle.contracts import errors
from scmle.contracts import qualification
from scmle.contracts.data import provenance

LOGGER = logging.getLogger(__name__)

CHARTER = "analysis_charter.json"
CLAIMS = "confirmed_claims.json"
LEDGER = "hypothesis_ledger.jsonl"
TRAIL = "trail.json"
BACKUP = ".bak"
KINDS: Mapping[str, Any] = {
    "dataset_manifest.json": provenance.DatasetManifest,
    "decision_cost_model.json": costs.Tariff,
    "error_budget.json": budget.ErrorBudget,
    "feasibility.json": feasibility.Feasibility,
    "split_manifest.json": manifest.Manifest,
}
NAMES = frozenset({CHARTER, CLAIMS, *KINDS})


def read(path: pathlib.Path) -> Any:
    """Decodes a stored JSON record.

    Args:
        path: The file.

    Returns:
        The decoded value.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def rewrite(name: str, stored: Any) -> Any:
    """Returns a stored record laid out at its kind's current version.

    A signed charter keeps its wrapper (format, hash, algorithm and
    signature); only the charter inside it moves, since the hash and the
    signature cover the version-1 layout.

    Args:
        name: The file name, which names the record kind.
        stored: The decoded record.

    Returns:
        The record at the current version.
    """
    if name == CHARTER:
        if "charter" not in stored:
            return contracts.AnalysisCharter.from_dict(stored).to_dict()
        inner = contracts.AnalysisCharter.from_dict(stored["charter"])
        return {**stored, "charter": inner.to_dict()}
    if name == CLAIMS:
        return {
            **stored,
            "claims": [
                qualification.Claim.from_dict(row).to_dict()
                for row in stored["claims"]
            ],
            "checks": [
                qualification.Check.from_dict(row).to_dict()
                for row in stored["checks"]
            ],
        }
    return KINDS[name].from_dict(stored).to_dict()


def inventory(folder: pathlib.Path) -> list[pathlib.Path]:
    """Lists the records this command knows under a folder.

    Args:
        folder: The folder to search, with its sub-folders.

    Returns:
        The record files, sorted, without those of a sealed trail.
    """
    sealed = [p.parent for p in folder.rglob(TRAIL)]
    return sorted(
        p
        for p in folder.rglob("*.json")
        if p.name in NAMES and not any(p.is_relative_to(t) for t in sealed)
    )


def verify(folder: pathlib.Path, key: str | None) -> list[str]:
    """Verifies the stored bytes of every record under a folder.

    Args:
        folder: The folder.
        key: The signing key, or None to use `SCMLE_SIGNING_KEY`.

    Returns:
        One line per problem; empty if everything verified.
    """
    problems: list[str] = []
    for path in inventory(folder):
        try:
            stored = read(path)
            if path.name == CHARTER:
                sealing.read_verified(path, key)
            if rewrite(path.name, rewrite(path.name, stored)) != rewrite(
                path.name, stored
            ):
                problems.append(f"{path}: the upgrade is not stable")
        except (OSError, ValueError, KeyError, TypeError) as error:
            # handled: reported as a problem with the error text
            problems.append(f"{path}: {error}")
    for ledger in sorted(folder.rglob(LEDGER)):
        result = claims_audit.verify(ledger.parent, key)
        problems.extend(
            f"{ledger.parent}: {found.name}: {found.detail}"
            for found in result.findings
            if found.status == claims_audit.FAIL
        )
    return problems


def restore(paths: list[pathlib.Path]) -> None:
    """Puts back the stored bytes of each file from its `.bak` copy.

    Args:
        paths: The rewritten files; each `.bak` copy is removed.
    """
    for path in paths:
        backup = path.with_name(path.name + BACKUP)
        durability.write_atomically(path, backup.read_bytes())
        backup.unlink()


class Records(command.Command):
    """Upgrades the stored records of a folder to the current schema."""

    name = "records"
    help = "upgrade stored records to the current schema version"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        """Adds the `upgrade` action.

        Args:
            parser: The argument parser of this command.
        """
        actions = parser.add_subparsers(dest="records_action", required=True)
        upgrade = actions.add_parser(
            "upgrade",
            help="rewrite every known record at the current schema version, "
            "keeping a .bak of each file",
        )
        upgrade.add_argument("folder")
        upgrade.add_argument(
            "--dry-run",
            action="store_true",
            help="list the files that would change and write nothing",
        )
        upgrade.add_argument(
            "--key-file",
            help="file holding the signing key; default: SCMLE_SIGNING_KEY",
        )

    def execute(self, args: argparse.Namespace) -> int:
        """Verifies, rewrites and verifies again, restoring on a failure.

        Args:
            args: The parsed command-line arguments.

        Returns:
            EXIT_OK (0) when every record verified before and after.

        Raises:
            ArtifactError: If the folder is missing, or the records do not
                verify before the upgrade (nothing is written) or after it
                (every file is restored from its `.bak`).
        """
        folder = pathlib.Path(args.folder)
        if not folder.is_dir():
            raise errors.ArtifactError(f"no such folder: {folder}")
        key = (
            pathlib.Path(args.key_file).read_text(encoding="utf-8").strip()
            if args.key_file
            else None
        )
        self.check(verify(folder, key), "before the upgrade; nothing changed")
        changes = {}
        for path in inventory(folder):
            stored = read(path)
            current = rewrite(path.name, stored)
            if current != stored:
                changes[path] = current
        verb = "would upgrade" if args.dry_run else "upgraded"
        for path in changes:
            print(f"{verb} {path.relative_to(folder)}")
        print(f"{len(changes)} of {len(inventory(folder))} record(s) {verb}")
        if args.dry_run:
            return command.EXIT_OK
        for path, current in changes.items():
            backup = path.with_name(path.name + BACKUP)
            durability.write_atomically(backup, path.read_bytes())
            text = json.dumps(current, indent=2, sort_keys=True) + "\n"
            durability.write_atomically(path, text.encode("utf-8"))
        problems = verify(folder, key)
        if problems:
            restore(list(changes))
        self.check(problems, "after the upgrade; every file was restored")
        return command.EXIT_OK

    @staticmethod
    def check(problems: list[str], when: str) -> None:
        """Logs each problem and fails if there is any.

        Args:
            problems: The verification problems.
            when: Which verification it was and what happened to the files.

        Raises:
            ArtifactError: If `problems` is not empty.
        """
        for problem in problems:
            LOGGER.error("%s", problem)
        if problems:
            raise errors.ArtifactError(f"stored records do not verify {when}")
