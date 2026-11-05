"""One traceable report from a business question to observed outcomes.

`build` copies the retained records of a deployed model into a folder and
seals them as a hash-chained, signed trail; `verify` checks that folder alone;
`render` prints it. See ``docs/TRAIL.md``.
"""

from scmle.trail.builder import build
from scmle.trail.chain import Report
from scmle.trail.chain import seal
from scmle.trail.chain import verify
from scmle.trail.show import render
from scmle.trail.trace import Finding

__all__ = ["Finding", "Report", "build", "render", "seal", "verify"]
