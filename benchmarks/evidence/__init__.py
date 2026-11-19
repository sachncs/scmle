"""Measured evidence for the scorecard claims (plan section 5).

Each module measures one claim, writes its raw results next to it as
`<module>.json`, and checks that the stored results are for the current
code. `benchmarks.scorecard` reads the stored results; nothing here is
typed by hand.
"""

import hashlib
from pathlib import Path


def source(names: tuple[str, ...], package: Path) -> str:
    """Hash of every Python file in some subpackages of the product.

    A stored measurement fingerprinted with it goes stale when any file the
    measured code may run changes, not only the functions it names.

    Args:
        names: Subpackages of `package`, such as `models`.
        package: The `scmle` package directory.

    Returns:
        SHA-256 over the path and contents of every Python file in `names`,
        in path order.
    """
    digest = hashlib.sha256()
    for name in names:
        for path in sorted((package / name).rglob("*.py")):
            digest.update(path.relative_to(package).as_posix().encode())
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()
