"""The retirement mark of a bundle: a retired bundle is never loaded.

Retiring a model writes `retired.json` into its bundle directory. The file is
not one of the bundle's checksummed members, so the bundle's content address
does not change, and `Bundle.load` refuses the directory before reading
anything else. Every path that serves or scores a bundle loads it first, so a
retired model is refused by `scmle predict`, `batch`, `serve`, `package`,
`export-onnx` and the registry alike.
"""

import json
import pathlib

from scmle import durability
from scmle.contracts import errors
from scmle.contracts import records

FILE = "retired.json"


class RetiredError(errors.BlockedError):
    """A retired bundle was asked to load, score or serve."""


def mark(directory: str | pathlib.Path, record: records.Layout) -> None:
    """Writes the retirement mark into a bundle directory.

    Args:
        directory: The bundle directory.
        record: What the mark says: at least `reason` and `retired`.

    Raises:
        ArtifactError: If the directory does not exist.
    """
    root = pathlib.Path(directory)
    if not root.is_dir():
        raise errors.ArtifactError(f"no such bundle directory: {root}")
    text = json.dumps(dict(record), indent=2, sort_keys=True) + "\n"
    durability.write_atomically(root / FILE, text.encode("utf-8"))


def read(directory: str | pathlib.Path) -> records.Layout | None:
    """Returns the retirement mark of a bundle directory, if it has one.

    Args:
        directory: The bundle directory.

    Returns:
        The mark, or None when the bundle is not retired. A mark that cannot
        be parsed still retires the bundle and reads as `{"reason":
        "unreadable retirement mark"}`.
    """
    path = pathlib.Path(directory) / FILE
    if not path.is_file():
        return None
    try:
        found = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        # handled: a damaged mark still retires the bundle
        return {"reason": "unreadable retirement mark"}
    return found if isinstance(found, dict) else {"reason": str(found)}


def enforce(directory: str | pathlib.Path) -> None:
    """Refuses a retired bundle.

    Args:
        directory: The bundle directory.

    Raises:
        RetiredError: If the directory carries a retirement mark.
    """
    found = read(directory)
    if found is not None:
        raise RetiredError(
            f"the model at {directory} was retired"
            f" ({found.get('retired', 'date unknown')}): "
            f"{found.get('reason', 'no reason recorded')}",
            {"retirement": found},
        )
