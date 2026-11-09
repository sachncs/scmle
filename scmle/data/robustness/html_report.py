"""A static HTML page written from the stability and shift reports.

`render` turns the two JSON documents of a charter run
(`stability_report.json`, `shift_report.json`) into one page with no script,
no stylesheet link and no network reference, so it opens from a local file.
Each finding gets its status, its headline numbers and a small
specification curve (the effect of every specification, ordered, with its
interval). Every string taken from the documents is escaped. The page
describes; it adds no number that is not already in the two documents.
"""

import html
from typing import Any

WIDTH = 360
HEIGHT = 90
PAD = 6
STYLE = (
    "body{font:14px/1.4 sans-serif;margin:2rem auto;max-width:56rem;"
    "padding:0 1rem}table{border-collapse:collapse}"
    "td,th{border:1px solid #888;padding:.25rem .5rem;text-align:left}"
    "svg{background:#f6f6f6}"
)


def esc(value: Any) -> str:
    """Escapes a value for HTML.

    Args:
        value: Any value; it is converted with `str` first.

    Returns:
        The text with `&`, `<`, `>` and both quote characters escaped.
    """
    return html.escape(str(value), quote=True)


def curve(specs: list[dict[str, Any]]) -> str:
    """Draws the specification curve of one finding.

    Args:
        specs: One record per specification, with the keys `effect`, `low`
            and `high`. Records where any of the three is missing or None
            are left out.

    Returns:
        An inline SVG element with the specifications ordered by effect, each
        drawn as its interval and point estimate against a dashed zero
        line; or a paragraph saying that no specification produced an
        estimate when none is left.
    """
    points = [
        s
        for s in specs
        if s.get("effect") is not None
        and s.get("low") is not None
        and s.get("high") is not None
    ]
    if not points:
        return "<p>no specification produced an estimate</p>"
    points = sorted(points, key=lambda s: s["effect"])
    low = min(min(s["low"] for s in points), 0.0)
    high = max(max(s["high"] for s in points), 0.0)
    span = (high - low) or 1.0

    def y(value: float) -> float:
        """Maps an effect value to a vertical pixel position.

        Args:
            value: Effect value between `low` and `high`.

        Returns:
            The y coordinate, with `high` at the top margin and `low` at
            the bottom margin.
        """
        return PAD + (HEIGHT - 2 * PAD) * (1.0 - (value - low) / span)

    step = (WIDTH - 2 * PAD) / max(len(points) - 1, 1)
    parts = [
        f'<svg width="{WIDTH}" height="{HEIGHT}" role="img" '
        f'aria-label="specification curve of {len(points)} specifications">',
        f'<line x1="{PAD}" x2="{WIDTH - PAD}" y1="{y(0.0):.1f}" '
        f'y2="{y(0.0):.1f}" stroke="#888" stroke-dasharray="3"/>',
    ]
    for index, spec in enumerate(points):
        x = PAD + index * step
        parts.append(
            f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{y(spec["low"]):.1f}" '
            f'y2="{y(spec["high"]):.1f}" stroke="#468"/>'
        )
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y(spec["effect"]):.1f}" r="2" '
            'fill="#124"/>'
        )
    parts.append("</svg>")
    return "".join(parts)


def stability_section(document: dict[str, Any]) -> str:
    """Renders the stability part of the page.

    Args:
        document: The stability report, with the keys `status`, `reason`,
            `counts`, `findings` (each with `name` and `status`, and
            optionally `reason`, `headline` and `specs`) and `guarantee`.

    Returns:
        The HTML fragment, with every taken string escaped.
    """
    rows = [
        "<h2>Stability</h2>",
        f"<p>status: <b>{esc(document['status'])}</b>. "
        f"{esc(document['reason'])}</p>",
        f"<p>counts: {esc(document['counts'])}</p>",
    ]
    findings = document["findings"]
    if not findings:
        rows.append("<p>no findings were evaluated.</p>")
    for item in findings:
        rows.append(
            f"<h3>{esc(item['name'])}: {esc(item['status'])}</h3>"
            f"<p>{esc(item.get('reason', ''))}</p>"
            f"<p>{esc(item.get('headline', ''))}</p>"
            f"{curve(item.get('specs', []))}"
        )
    rows.append(f"<p><i>{esc(document['guarantee'])}</i></p>")
    return "".join(rows)


def shift_section(document: dict[str, Any]) -> str:
    """Renders the shift part of the page.

    Args:
        document: The shift report, with the keys `status`, `reason`,
            `periods` and `forward` (each with `status` and optionally
            `reason`) and `assumptions` (a list of strings).

    Returns:
        The HTML fragment, with every taken string escaped.
    """
    forward = document["forward"]
    periods = document["periods"]
    items = "".join(f"<li>{esc(a)}</li>" for a in document["assumptions"])
    return (
        "<h2>Shift</h2>"
        f"<p>status: <b>{esc(document['status'])}</b>. "
        f"{esc(document['reason'])}</p>"
        "<table><tr><th>check</th><th>status</th><th>reason</th></tr>"
        f"<tr><td>per-period drift</td><td>{esc(periods['status'])}</td>"
        f"<td>{esc(periods.get('reason', ''))}</td></tr>"
        f"<tr><td>forward check</td><td>{esc(forward['status'])}</td>"
        f"<td>{esc(forward.get('reason', ''))}</td></tr></table>"
        f"<h3>Assumptions</h3><ul>{items}</ul>"
    )


def render(stability: dict[str, Any], shift: dict[str, Any]) -> str:
    """Renders one self-contained HTML page for the two reports.

    Args:
        stability: The stability report document.
        shift: The shift report document.

    Returns:
        The complete HTML page as a string.
    """
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        "<title>SCMLE stability and shift report</title>"
        f"<style>{STYLE}</style></head><body>"
        "<h1>Stability and shift</h1>"
        f"{stability_section(stability)}{shift_section(shift)}"
        "</body></html>"
    )
