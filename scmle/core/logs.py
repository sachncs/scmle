"""The package logging convention (task 2.1, ERR-24, ERR-22).

- One logger per module: `LOGGER = logging.getLogger(__name__)`.
- Every record carries the run id and the request id in scope. The run
  binds its id for the length of the run and the server binds each request's
  id while it handles it (`bind`); `Stamp` copies both onto every record a
  handler sees, so they appear in the text and the JSON formats alike.
- A WARNING at every swallow, fallback and gate downgrade, so the change is
  visible in the log as well as in the returned notes.
- A degraded optional path (an optional dependency missing, a cheaper method
  substituted) also raises `Degraded` through `warnings.warn` (`degrade`),
  which callers and tests can filter or turn into errors.
- An `except` that neither re-raises nor logs carries a
  `# handled: <where the error goes>` marker (tests/unit/test_handlers.py).

The library never installs handlers on import; the CLI (or an embedding
application) calls `configure` once.
"""

from collections.abc import Iterator
import contextlib
import contextvars
import json
import logging
from typing import IO
import warnings

RUN: contextvars.ContextVar[str] = contextvars.ContextVar("run", default="")
REQUEST: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request", default=""
)
PACKAGE = "scmle"
TEXT = (
    "%(asctime)s %(levelname)s %(name)s "
    "[run=%(run)s request=%(request)s]: %(message)s"
)
# `degrade` is called from a module function on behalf of that function's
# caller: 1 is `degrade`, 2 the module function, 3 the code that chose the
# degraded path.
CALLER = 3


class Degraded(UserWarning):
    """An optional path ran in a reduced form."""


class Stamp(logging.Filter):
    """Copies the run id and request id in scope onto every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Adds `run` and `request` attributes to `record`.

        Args:
            record: The record being handled.

        Returns:
            True: the filter never drops a record.
        """
        record.run = RUN.get()
        record.request = REQUEST.get()
        return True


class Lines(logging.Formatter):
    """Formats a record as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        """Returns `record` as a JSON line.

        Args:
            record: The record to format.

        Returns:
            One line of JSON with the time, level, logger, message, run id,
            request id and, when an exception is attached, its traceback.
        """
        document = {
            "time": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "run": getattr(record, "run", RUN.get()),
            "request": getattr(record, "request", REQUEST.get()),
        }
        if record.exc_info:
            document["traceback"] = self.formatException(record.exc_info)
        return json.dumps(document, default=str)


@contextlib.contextmanager
def bind(run: str | None = None, request: str | None = None) -> Iterator[None]:
    """Puts a run id and/or request id in scope for the enclosed block.

    Context variables follow the thread and the asyncio task, so concurrent
    requests do not see each other's ids. An argument left as None keeps
    the id already in scope.

    Args:
        run: The run id.
        request: The request id.

    Yields:
        None, with the ids in scope until the block exits.
    """
    tokens = [
        (variable, variable.set(value))
        for variable, value in ((RUN, run), (REQUEST, request))
        if value is not None
    ]
    try:
        yield
    finally:
        for variable, token in reversed(tokens):
            variable.reset(token)


def handler(
    structured: bool = False, stream: IO[str] | None = None
) -> logging.Handler:
    """Returns a stream handler that stamps the ids and formats records.

    Args:
        structured: JSON lines when True, the `TEXT` format otherwise.
        stream: Where records go; standard error by default.

    Returns:
        The handler, with a `Stamp` filter and the chosen formatter.
    """
    made = logging.StreamHandler(stream)
    made.addFilter(Stamp())
    made.setFormatter(Lines() if structured else logging.Formatter(TEXT))
    return made


def configure(level: int | str, structured: bool = False) -> logging.Handler:
    """Installs the convention's handler on the package logger.

    Calling it again replaces the handler it installed before, so repeated
    calls (tests, an embedding application) never duplicate records.

    Args:
        level: The package logger's level.
        structured: JSON lines when True, text otherwise.

    Returns:
        The installed handler.
    """
    package = logging.getLogger(PACKAGE)
    for old in [h for h in package.handlers if getattr(h, "convention", 0)]:
        package.removeHandler(old)
    installed = handler(structured)
    installed.convention = True  # type: ignore[attr-defined]
    package.addHandler(installed)
    package.setLevel(level)
    return installed


def degrade(logger: logging.Logger, message: str, *args: object) -> None:
    """Reports a degraded optional path: a WARNING and a `Degraded` warning.

    Args:
        logger: The calling module's logger.
        message: A %-style message naming what was lost and what replaced it.
        *args: The message's arguments.
    """
    logger.warning(message, *args)
    warnings.warn(message % args if args else message, Degraded, CALLER)
