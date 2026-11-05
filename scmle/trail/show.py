"""A readable account of a trail, from the business question to outcomes."""

from pathlib import Path
from typing import Any

from scmle.trail import records

WIDTH = 78


def clip(text: str) -> str:
    """Shortens a line to the display width."""
    return text if len(text) <= WIDTH else text[: WIDTH - 3] + "..."


def facts_lines(facts: dict[str, Any]) -> list[str]:
    """One line per fact; long values are cut."""
    lines = []
    for key in sorted(facts):
        value = facts[key]
        if (
            isinstance(value, list)
            and value
            and all(isinstance(i, dict) and "passed" in i for i in value)
        ):
            for item in value:
                mark = "ok  " if item.get("passed") else "FAIL"
                lines.append(
                    f"    [{mark}] {item.get('id')}: {item.get('path')} "
                    f"{item.get('op')} {item.get('value')} "
                    f"(observed {item.get('observed')})"
                )
            continue
        lines.append(clip(f"    {key}: {value}"))
    return lines


def render(directory: str | Path) -> str:
    """Text of the trail in ``directory``.

    Raises:
        FileNotFoundError: If there is no ``trail.json``.
    """
    document = records.read_json(Path(directory) / records.TRAIL_FILE)
    lines = [
        f"Trail for {document['model']} v{document['version']}",
        f"charter {document['charter_hash']}",
        f"head    {document['head']}",
        "",
    ]
    for index, step in enumerate(document["steps"], start=1):
        lines.append(f"{index}. {step['title']}  [{step['kind']}]")
        lines.extend(facts_lines(step["facts"]))
        for relative, held in step["records"].items():
            lines.append(f"    record {relative}  {held[:12]}")
        lines.append("")
    return "\n".join(lines)
