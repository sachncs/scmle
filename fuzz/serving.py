"""Atheris fuzz target: the prediction service's request parser.

`scmle.contracts.serving.parse` is the first code that sees an untrusted
request body. The first input byte sets the record limit (1 to 64); the rest
is decoded as JSON and parsed. A body the JSON decoder rejects is skipped,
since the decoder is not under test. The parser may only refuse with the
exceptions it documents (TypeError, ValueError, OverflowError,
MismatchError); any other exception, or an accepted request that breaks the
request schema, is a crash.

    python fuzz/serving.py fuzz/seeds/serving -max_total_time=300
"""

import json
import sys

import atheris

with atheris.instrument_imports():
    from scmle.contracts import serving

# Records a request may hold, at most; the first input byte picks 1 to this.
LIMIT = 64


def probe(data: bytes) -> None:
    """Parses one fuzzed request and checks the result.

    Args:
        data: The fuzzed input: a limit byte, then the request body.

    Raises:
        RuntimeError: If an accepted request breaks the request schema.
    """
    if not data:
        return
    limit = data[0] % LIMIT + 1
    try:
        payload = json.loads(data[1:])
    except (ValueError, RecursionError):
        return
    try:
        request = serving.parse(payload, limit)
    except (TypeError, ValueError, OverflowError, serving.MismatchError):
        return
    if not 1 <= len(request.records) <= limit:
        raise RuntimeError(f"accepted {len(request.records)} records")
    if not all(isinstance(record, dict) for record in request.records):
        raise RuntimeError("accepted a record that is not an object")
    if request.schema_version not in serving.SUPPORTED_VERSIONS:
        raise RuntimeError(f"accepted version {request.schema_version!r}")
    digest = request.expected_schema_digest
    if digest is not None and not isinstance(digest, str):
        raise RuntimeError("accepted a digest that is not text")


def main() -> None:
    """Runs the fuzzer with libFuzzer's command-line arguments."""
    atheris.Setup(sys.argv, probe)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
