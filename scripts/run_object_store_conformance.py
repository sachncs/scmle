#!/usr/bin/env python3
r"""Run the object-store conformance suite against a backend you point it at.

Local (no account)::

    python scripts/run_object_store_conformance.py local --root /tmp/store

MinIO or any S3-compatible endpoint (needs boto3 and credentials in the
environment)::

    python scripts/run_object_store_conformance.py s3 --bucket test \
        --endpoint-url http://127.0.0.1:9000

The GCS emulator or a real bucket (needs google-cloud-storage; the emulator
is selected by STORAGE_EMULATOR_HOST)::

    python scripts/run_object_store_conformance.py gcs --bucket test

The suite writes only under a unique ``conformance-<id>/`` prefix and removes
it. Exit status is 0 only if every check passed. These runs against real
services are the cloud half of Phase 15; the unit tests run the same suite
against in-memory fakes.
"""

import argparse
import json
import sys

from scmle.data.store import cloud
from scmle.data.store import conformance
from scmle.data.store import objectstore


def build(options: argparse.Namespace) -> objectstore.ObjectStore:
    """The store the arguments name."""
    if options.backend == "local":
        return objectstore.LocalObjectStore(options.root)
    if options.backend == "s3":
        arguments = {}
        if options.endpoint_url:
            arguments["endpoint_url"] = options.endpoint_url
        return cloud.s3_store(options.bucket, options.prefix, **arguments)
    return cloud.gcs_store(options.bucket, options.prefix)


def main(arguments: list[str] | None = None) -> int:
    """Run the suite and print the report as JSON."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("backend", choices=("local", "s3", "gcs"))
    parser.add_argument("--root", help="directory, for the local backend")
    parser.add_argument("--bucket")
    parser.add_argument("--prefix", default="")
    parser.add_argument("--endpoint-url")
    parser.add_argument("--out", help="write the report here as well")
    options = parser.parse_args(arguments)
    if options.backend == "local" and not options.root:
        parser.error("local needs --root")
    if options.backend != "local" and not options.bucket:
        parser.error(f"{options.backend} needs --bucket")
    report = conformance.run_conformance(build(options)).to_dict()
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if options.out:
        with open(options.out, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text + "\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
