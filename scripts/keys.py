"""Writes the stored-record key map: docs/record_keys.json and RECORD_KEYS.md.

Stage B renamed the multi-word keys of every stored record (for example
`data_hash -> digests.data`, `run_id -> run`). The approved vocabularies,
`plan/stage_b/phase5_vocabulary.md` (records read back) and
`plan/stage_b/phase5_vocabulary_rest.md` (write-only records), hold one
table per record class. This script reads the tables whose renames have
landed into `docs/record_keys.json` (old stored key -> new dotted key, per
record kind) and renders `docs/RECORD_KEYS.md` from that JSON, so the two
always agree. A row of the second file counts only when the key it names is
written to disk.

    python scripts/keys.py                 # rebuild both files
    python scripts/keys.py --check         # fail if RECORD_KEYS.md is stale

The vocabulary files are planning files and may be absent from a checkout;
then the JSON is kept as it is and only the Markdown is rendered from it.
"""

import argparse
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
VOCABULARY = ROOT / "plan" / "stage_b" / "phase5_vocabulary.md"
REST = ROOT / "plan" / "stage_b" / "phase5_vocabulary_rest.md"
TABLE = ROOT / "docs" / "record_keys.json"
DOC = ROOT / "docs" / "RECORD_KEYS.md"
HEADING = re.compile(r"^### `([^`]+)` `([^`]+)`")
ROW = re.compile(r"^\| `([^`]+)` \| (keep )?`?([^`|]+?)`? \|")
FAMILY = re.compile(r"^## (F\d+) ")
LANDED = frozenset({"F1", "F2", "F3", "F8"})
"""Families of the rest vocabulary whose renames are in the code."""
PROPERTIES = frozenset(
    {
        "contracts/data/profiles.py Curation",
        "contracts/modalities.py Declaration",
    }
)
"""Kinds of the stored-property family (F10) that a landed family writes."""
UNWRITTEN = frozenset(
    {
        ("assurance/admissibility/screen/report.py Screen", "unchecked_reason"),
        ("contracts/data/profiles.py Portrait", "column_statistics"),
        ("contracts/data/profiles.py Portrait", "target_summary"),
        *(
            ("contracts/data/profiles.py Blueprint", key)
            for key in (
                "group_column",
                "time_column",
                "weight_column",
                "event_column",
                "ordinal_levels",
                "query_column",
                "target_columns",
                "value_column",
                "treatment_column",
                "binary_outcome",
                "text_encoding",
                "embedding_model",
                "informative_missing",
            )
        ),
    }
)
"""Rest rows whose old key never reached a file, or reached it unchanged."""
STORED = {
    ("causal/identification/select.py Plan", "chartered_order"): (
        "order_chartered"
    ),
    ("contracts/data/profiles.py Portrait", "dataset_statistics"): "dataset",
}
"""Rest rows whose field was written under another old key."""

PREFACE = """\
# Stored record keys

Stage B renamed the multi-word keys of the records scmle writes to disk
(charters, custody manifests, ledgers, contracts, catalogs, detector
certificates, governance, deployment and monitoring records, run manifests
and evaluation members). Each name became one word or a nested object:
`data_hash` is now `digests.data`, `row_count` is now `rows`, `run_id` is
`identifier` on the record that is the run and `run` on a record that
refers to a run. Every renamed record kind has a higher `schema_version`,
except the few that carry none (a trial checkpoint, the stage records of a
lifecycle report): scmle tells their two layouts apart by the keys. The
write-only records (detector certificates, framing and ingest reports,
robustness and causal reports, decision reports, capability, modality,
observability and performance contracts, release policies and explanation
evidence) were renamed in a second pass and are listed here too.

**scmle reads old files itself.** Each record kind has a migration, so a
folder written by an older scmle loads without any change. Hashes and
signatures that were anchored at version 1 (the charter hash, the custody
seal, the ledger chains, HMAC-signed records) are checked on the stored
bytes before the migration, so old seals keep verifying.

**Update your own scripts** if they read these files directly
(with `json.load`, `jq`, a dashboard query or a notebook). For each key you
read, look up its record below and read the new key; a dotted name such as
`digests.charter` is a nested object (`record["digests"]["charter"]`). To
read both old and new files, try the new key first and fall back to the old
one, as `scripts/trace_model.py` does in its `field` helper.

**To rewrite a folder at the current version**, run `scmle records upgrade
FOLDER` (add `--dry-run` to list what would change, `--key-file` for the
signing key; default `SCMLE_SIGNING_KEY`). It rewrites the charter, the
split and dataset manifests, the feasibility, error budget and cost model
records and the claims file through their migrations, keeps each old file
as `NAME.bak`, and verifies the seals, signatures and chains on the stored
bytes before and after; if the check after the rewrite fails, every file is
restored from its `.bak` and the command exits 1. Hash-chained logs, the
negative control (its digest is in the ledger) and sealed trails are left
as stored; they still load through their migrations.

Records are listed by class. The files most scripts read hold these:
`analysis_charter.json` holds an `AnalysisCharter` (with its `Question`,
`AnalysisEstimand`, `SetSizes`, `Trigger`, `Procedure`, `Uncertainty`,
`ErrorBudget` and `Tariff` parts), `split_manifest.json` the custody
`Manifest`, `feasibility.json` a `Feasibility`, `decision_cost_model.json`
a `Tariff`, a `--leakage-attestations` file `Affidavit` rows (with
`Attestation` fields), and a loop `policy.json` a `Preregistration`.
[`ARTIFACTS.md`](ARTIFACTS.md) lists every file a run writes.

The same map is machine-readable in
[`record_keys.json`](record_keys.json): `{"kinds": {"<module> <Class>":
{"<old key>": "<new dotted key>"}}}`. Module paths are the ones the
vocabulary used when the renames were approved; class names did not change.
Keys that kept their name (`schema_version`, `p_value`,
`adjusted_p_value`, `key_id`, `running_variable`, `c_00` to `c_11`) are
not listed. This file is generated by `scripts/keys.py`; do not edit it by
hand.
"""


def parse(text: str) -> dict[str, dict[str, str]]:
    """Reads the vocabulary tables into old -> new maps per record kind.

    Args:
        text: The vocabulary Markdown.

    Returns:
        `"<module> <Class>"` -> {old key: new dotted key}, renames only; a
        kept name is left out. Kinds appear in the vocabulary's order.
    """
    kinds: dict[str, dict[str, str]] = {}
    kind = ""
    for line in text.splitlines():
        heading = HEADING.match(line)
        if heading:
            kind = f"{heading.group(1)} {heading.group(2)}"
            continue
        row = ROW.match(line)
        if row and kind and not row.group(2):
            kinds.setdefault(kind, {})[row.group(1)] = row.group(3)
    return kinds


def landed(text: str) -> dict[str, dict[str, str]]:
    """Reads the landed, written rows of the rest vocabulary.

    Args:
        text: The rest vocabulary Markdown.

    Returns:
        `"<module> <Class>"` -> {old stored key: new dotted key} for the
        families in `LANDED` and the kinds in `PROPERTIES`, without the rows
        in `UNWRITTEN` and with the old keys in `STORED` substituted.
    """
    kinds: dict[str, dict[str, str]] = {}
    family = kind = ""
    for line in text.splitlines():
        found = FAMILY.match(line)
        if found:
            family = found.group(1)
            continue
        heading = HEADING.match(line)
        if heading:
            kind = f"{heading.group(1)} {heading.group(2)}"
            continue
        row = ROW.match(line)
        wanted = family in LANDED or (family == "F10" and kind in PROPERTIES)
        if not row or not kind or row.group(2) or not wanted:
            continue
        old = row.group(1)
        if (kind, old) in UNWRITTEN:
            continue
        old = STORED.get((kind, old), old)
        kinds.setdefault(kind, {})[old] = row.group(3)
    return kinds


def render(kinds: dict[str, dict[str, str]]) -> str:
    """Renders the Markdown table of every record kind.

    Args:
        kinds: The map that `parse` returns.

    Returns:
        The content of `docs/RECORD_KEYS.md`.
    """
    parts = [PREFACE]
    total = sum(len(keys) for keys in kinds.values())
    parts.append(
        f"\n## The map\n\n{total} renamed keys on {len(kinds)} record kinds.\n"
    )
    for kind, keys in kinds.items():
        module, record = kind.split(" ")
        parts.append(
            f"\n### `{record}` (`{module}`)\n\n| Old | New |\n|---|---|\n"
        )
        parts.extend(f"| `{old}` | `{new}` |\n" for old, new in keys.items())
    return "".join(parts)


def main(arguments: list[str]) -> int:
    """Entry point.

    Args:
        arguments: Command-line arguments.

    Returns:
        0 on success; 1 when `--check` finds the Markdown stale.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if RECORD_KEYS.md differs from the JSON map",
    )
    options = parser.parse_args(arguments)
    if options.check:
        kinds = json.loads(TABLE.read_text(encoding="utf-8"))["kinds"]
        if DOC.read_text(encoding="utf-8") != render(kinds):
            print("docs/RECORD_KEYS.md is stale: run scripts/keys.py")
            return 1
        return 0
    if VOCABULARY.is_file():
        kinds = parse(VOCABULARY.read_text(encoding="utf-8"))
        sources = ["plan/stage_b/phase5_vocabulary.md"]
        if REST.is_file():
            for kind, keys in landed(REST.read_text(encoding="utf-8")).items():
                kinds.setdefault(kind, {}).update(keys)
            sources.append("plan/stage_b/phase5_vocabulary_rest.md")
        document: dict[str, object] = {"source": ", ".join(sources)}
        document["kinds"] = kinds
        TABLE.write_text(
            json.dumps(document, indent=2) + "\n", encoding="utf-8"
        )
    kinds = json.loads(TABLE.read_text(encoding="utf-8"))["kinds"]
    DOC.write_text(render(kinds), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
