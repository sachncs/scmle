#!/usr/bin/env python3
"""Verify an independent release review against an exact source commit."""

import argparse
import json
import os
from pathlib import Path
import sys

from scmle.governance import review

DEFAULT_REQUIRED_KINDS = [kind.value for kind in review.ReviewClaimKind]


def parse_arguments(arguments: list[str] | None = None) -> argparse.Namespace:
    """Parses the review verification command line.

    Args:
        arguments: Command-line arguments; `None` reads `sys.argv`.

    Returns:
        The namespace with the review path, source commit, required claim
        kinds and the name of the signing-key environment variable.
    """
    parser = argparse.ArgumentParser(
        description="Verify signed independent release review evidence."
    )
    parser.add_argument("review_path", type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument(
        "--required-kind",
        action="append",
        dest="required_kinds",
        choices=[kind.value for kind in review.ReviewClaimKind],
        default=DEFAULT_REQUIRED_KINDS,
    )
    parser.add_argument(
        "--signing-key-env",
        default="SCMLE_REVIEW_SIGNING_KEY",
        help="Environment variable containing the review verification key.",
    )
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> int:
    """Returns zero only when the signed review satisfies every gate.

    Args:
        arguments: Command-line arguments; `None` reads `sys.argv`.

    Returns:
        0 when the review is accepted; 1 when the acceptance gate is not
        satisfied; 2 when the signing key is missing or the review cannot be
        read or verified.
    """
    options = parse_arguments(arguments)
    signing_key = os.environ.get(options.signing_key_env)
    if not signing_key:
        print(
            f"missing review signing key in ${options.signing_key_env}",
            file=sys.stderr,
        )
        return 2
    try:
        artifact = review.IndependentReleaseReview.read(options.review_path)
        accepted = artifact.accepted(
            options.source_commit,
            tuple(
                review.ReviewClaimKind(value)
                for value in options.required_kinds
            ),
            signing_key,
        )
    except (OSError, TypeError, ValueError, KeyError) as error:
        print(f"review verification failed: {error}", file=sys.stderr)
        return 2
    if not accepted:
        print(
            "review verification failed: acceptance gate is not satisfied",
            file=sys.stderr,
        )
        return 1
    print(
        json.dumps(
            {
                "accepted": True,
                "release_id": artifact.release_id,
                "source_commit": artifact.source_commit,
                "reviewer_id": artifact.reviewer_id,
                "claim_kinds": sorted(
                    {claim.kind.value for claim in artifact.claims}
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
