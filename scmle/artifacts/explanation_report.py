"""Human-readable rendering for portable explanation evidence."""

import html
import json
from typing import Any
from typing import cast

from scmle.contracts import explanations


class ExplanationHtmlReport:
    """Renders one explanation record as a self-contained HTML artifact."""

    def render(
        self,
        evidence: explanations.ExplanationEvidence,
        title: str | None = None,
    ) -> str:
        """Builds an escaped, privacy-aware explanation report.

        Args:
            evidence: Validated portable explanation evidence.
            title: Optional document title. The method name is used by default.

        Returns:
            A self-contained HTML document with a summary, output, links, and
            the canonical evidence record.

        The renderer does not make evidence links active. Artifact URI schemes
        are useful to a local viewer but are not safe to turn into arbitrary
        browser navigations without a caller-owned trust policy.
        """
        record = evidence.to_dict()
        privacy = cast(dict[str, Any], record["privacy_decision"])
        output: Any = record["output"]
        if privacy["decision"] == "denied":
            output = {
                "status": "withheld",
                "reason": privacy["reason"],
            }
        rendered_record = {**record, "output": output}
        document_title = title or evidence.method.name
        return self.document(
            document_title,
            evidence.method.name,
            evidence.method.parameters,
            output,
            cast(list[str], record["limitations"]),
            privacy,
            cast(list[dict[str, Any]], record["evidence_links"]),
            cast(dict[str, Any], rendered_record),
        )

    @staticmethod
    def document(
        title: str,
        method_name: str,
        parameters: dict[str, Any],
        output: Any,
        limitations: list[str],
        privacy: dict[str, Any],
        evidence_links: list[dict[str, Any]],
        record: dict[str, Any],
    ) -> str:
        """Builds the deterministic HTML document from JSON-native fields."""
        escape = html.escape

        def pretty(value: Any) -> str:
            return escape(
                json.dumps(value, allow_nan=False, indent=2, sort_keys=True)
            )

        links = "".join(
            "<li><b>"
            + escape(str(link["relation"]))
            + "</b> <code>"
            + escape(str(link["uri"]))
            + "</code></li>"
            for link in evidence_links
        )
        limitations_html = "".join(
            f"<li>{escape(str(limitation))}</li>" for limitation in limitations
        )
        privacy_status = escape(str(privacy["decision"]))
        privacy_reason = escape(str(privacy["reason"]))
        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width">
  <title>{escape(title)} — explanation evidence</title>
  <style>
    :root {{ color-scheme: light; font-family: system-ui, sans-serif; }}
    body {{ background: #f6f7f3; color: #18231f; line-height: 1.5; margin: 0; }}
    main {{ margin: 0 auto; max-width: 960px; padding: 48px 24px 80px; }}
    h1, h2 {{ letter-spacing: -.03em; }}
    h1 {{ font-size: clamp(2rem, 5vw, 3.5rem); line-height: 1; }}
    h2 {{ border-top: 1px solid #cdd4c7; margin-top: 40px; padding-top: 16px; }}
    .meta {{
      color: #53635a;
      font-family: ui-monospace, monospace;
      font-size: .8rem;
    }}
    .status {{
      background: #e8edc7;
      border-left: 4px solid #9eaf2e;
      padding: 16px;
    }}
    pre {{
      background: #18231f;
      color: #e9efe5;
      overflow-x: auto;
      padding: 16px;
    }}
    code {{ font-family: ui-monospace, monospace; }}
    li + li {{ margin-top: 8px; }}
  </style>
</head>
<body>
  <main>
    <p class="meta">PORTABLE EXPLANATION EVIDENCE</p>
    <h1>{escape(title)}</h1>
    <p class="meta">method: {escape(method_name)}</p>
    <div class="status">
      <strong>Privacy decision: {privacy_status}</strong><br>
      {privacy_reason}
    </div>
    <h2>Method parameters</h2>
    <pre><code>{pretty(parameters)}</code></pre>
    <h2>Explanation output</h2>
    <pre><code>{pretty(output)}</code></pre>
    <h2>Limitations</h2>
    <ul>{limitations_html}</ul>
    <h2>Evidence links</h2>
    <ul>{links}</ul>
    <details>
      <summary>Canonical evidence record</summary>
      <pre><code>{pretty(record)}</code></pre>
    </details>
  </main>
</body>
</html>
"""
