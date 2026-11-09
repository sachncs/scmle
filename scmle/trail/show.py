"""A readable account of a trail, from the business question to outcomes."""

from pathlib import Path
from typing import Any

from scmle.trail import records

WIDTH = 78


def clip(text: str) -> str:
    """Shortens a line to the display width.

    Args:
        text: The line.

    Returns:
        The line, or its first `WIDTH - 3` characters followed by `...` if it is
        longer than `WIDTH`.
    """
    return text if len(text) <= WIDTH else text[: WIDTH - 3] + "..."


def facts_lines(facts: dict[str, Any]) -> list[str]:
    """Formats the facts of a step.

    Args:
        facts: The step's facts.

    Returns:
        One line per fact, sorted by name; long values are cut. A list of
        criteria with `passed` flags gets one line per criterion.
    """
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
    """Renders the trail in ``directory`` as text.

    Args:
        directory: The trail folder.

    Returns:
        The model, version, charter hash and head, then each step with its facts
        and records. A folder without ``trail.json`` raises `FileNotFoundError`.
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
