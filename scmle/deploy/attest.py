"""Signed provenance for a service folder and for the image built from it.

Two statements, in the shape of an in-toto statement (subject, predicate):

* `provenance.json`, written when the folder is packaged and shipped inside
  the image. Its subject is the model bundle (by manifest digest); its
  materials are the files that define the build. The service verifies it
  before it deserializes anything.
* `image-attestation.json`, written after the image is built, outside the
  image. Its subject is the image digest; it binds that digest to the
  provenance file and to the pinned base image.

Both are signed with HMAC-SHA256 under the signing key. That proves the
statement was made by a holder of the key and was not altered; it is not a
public-key signature, a transparency-log entry or a SLSA level, and it does
not prove that the build host was honest. Federated, keyless signing is not
provided.
"""

import argparse
from collections.abc import Mapping
import hmac
import json
import pathlib
import sys
from typing import Any

from scmle.contracts import errors
from scmle.contracts import signing
from scmle.deploy import provenance

STATEMENT_TYPE = "https://in-toto.io/Statement/v1"
SERVICE_PREDICATE = "https://scmle.dev/attestation/service-provenance/v1"
IMAGE_PREDICATE = "https://scmle.dev/attestation/image-binding/v1"
PROVENANCE_FILE = "provenance.json"
IMAGE_ATTESTATION_FILE = "image-attestation.json"
MATERIAL_FILES = (
    "Dockerfile",
    "serve.py",
    "requirements.txt",
    "requirements.lock",
    "environment.lock",
    "sbom.cdx.json",
)


def material(root: pathlib.Path, name: str) -> dict[str, Any]:
    """Describes one file of the folder by SHA-256.

    Args:
        root: Service folder.
        name: File name relative to `root`; the file must exist.

    Returns:
        A dict with `name` and `digest`, the latter a dict whose `sha256` is the
        hex digest without the `sha256:` prefix.
    """
    digest = provenance.file_digest(root / name)
    return {"name": name, "digest": {"sha256": digest.removeprefix("sha256:")}}


def statement(
    subject_name: str,
    subject_digest: str,
    predicate_type: str,
    predicate: Mapping[str, Any],
) -> dict[str, Any]:
    """Builds an unsigned statement about one subject.

    Args:
        subject_name: Name of the subject.
        subject_digest: SHA-256 digest of the subject, with or without the
            `sha256:` prefix.
        predicate_type: URI of the predicate type.
        predicate: Predicate content; copied.

    Returns:
        An in-toto style statement with one subject.
    """
    return {
        "_type": STATEMENT_TYPE,
        "subject": [
            {
                "name": subject_name,
                "digest": {"sha256": subject_digest.removeprefix("sha256:")},
            }
        ],
        "predicateType": predicate_type,
        "predicate": dict(predicate),
    }


def sign_statement(
    body: Mapping[str, Any], signing_key: str | bytes | None
) -> dict[str, Any]:
    """Wraps a statement with its signature; unsigned without a key.

    Args:
        body: The statement to sign.
        signing_key: Signing key, or None to resolve it from the environment.

    Returns:
        A dict with the `statement`, the signature `algorithm` and the
        `signature`, which is None when no key could be resolved.
    """
    key = signing.resolve_key(signing_key)
    return {
        "statement": dict(body),
        "algorithm": signing.SIGNATURE_ALGORITHM,
        "signature": None if key is None else signing.sign_payload(body, key),
    }


def verify_document(
    document: Mapping[str, Any], signing_key: str | bytes | None
) -> tuple[dict[str, Any], bool]:
    """Checks a signed statement.

    Args:
        document: A document produced by `sign_statement`.
        signing_key: Signing key, or None to resolve it from the environment.

    Returns:
        The statement and whether its signature was verified. A statement
        without a signature, or checked without a key, is returned as
        unverified; the caller decides whether that is acceptable.

    Raises:
        ArtifactError: If the document is malformed or its signature is present
            and wrong.
    """
    body = document.get("statement")
    if (
        not isinstance(body, dict)
        or body.get("_type") != STATEMENT_TYPE
        or not body.get("subject")
    ):
        raise errors.ArtifactError("attestation is not a valid statement")
    if document.get("algorithm") != signing.SIGNATURE_ALGORITHM:
        raise errors.ArtifactError("unsupported attestation algorithm")
    recorded = document.get("signature")
    key = signing.resolve_key(signing_key)
    if recorded is None or key is None:
        return body, False
    expected = signing.sign_payload(body, key)
    if not isinstance(recorded, str) or not hmac.compare_digest(
        recorded, expected
    ):
        raise errors.ArtifactError(
            "attestation signature is invalid: the statement was modified "
            "or signed with a different key"
        )
    return body, True


def read_document(path: pathlib.Path) -> dict[str, Any]:
    """Reads an attestation file.

    Args:
        path: Path of the JSON file.

    Returns:
        The parsed JSON object.

    Raises:
        ArtifactError: If the file cannot be read or parsed, or does not hold a
            JSON object.
    """
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise errors.ArtifactError(f"cannot read {path.name}") from error
    if not isinstance(document, dict):
        raise errors.ArtifactError(f"{path.name} is not an object")
    return document


def write_document(path: pathlib.Path, document: Mapping[str, Any]) -> None:
    """Writes an attestation file deterministically.

    Keys are sorted and the file ends with a newline.

    Args:
        path: Destination file.
        document: The document to write.
    """
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_provenance(
    folder: pathlib.Path,
    bundle_digest: str,
    builder: str,
    base_image: str,
    signing_key: str | bytes | None,
) -> dict[str, Any]:
    """Writes `provenance.json` for a packaged service folder.

    The materials are those files of `MATERIAL_FILES` that exist in the folder.

    Args:
        folder: Service folder.
        bundle_digest: Manifest digest of the model bundle (the subject).
        builder: Identifier of the builder, recorded in the predicate.
        base_image: Pinned base image, recorded in the predicate.
        signing_key: Signing key, or None to resolve it from the environment.

    Returns:
        The signed (or unsigned, without a key) document that was written.
    """
    materials = [
        material(folder, name)
        for name in MATERIAL_FILES
        if (folder / name).exists()
    ]
    body = statement(
        "model",
        bundle_digest,
        SERVICE_PREDICATE,
        {
            "builder": builder,
            "base_image": base_image,
            "materials": materials,
        },
    )
    document = sign_statement(body, signing_key)
    write_document(folder / PROVENANCE_FILE, document)
    return document


def material_problems(
    folder: pathlib.Path, body: Mapping[str, Any]
) -> list[str]:
    """Lists materials of a service statement that differ from the folder.

    Args:
        folder: Service folder to check.
        body: The verified statement, whose predicate lists the materials.

    Returns:
        One message per material that is missing or whose digest has changed;
        empty if all match.
    """
    problems = []
    for item in body.get("predicate", {}).get("materials", []):
        path = folder / str(item.get("name", ""))
        if not path.is_file():
            problems.append(f"{item.get('name')} is missing")
        elif provenance.file_digest(path) != "sha256:" + str(
            item.get("digest", {}).get("sha256")
        ):
            problems.append(f"{item.get('name')} was modified")
    return problems


def write_image_attestation(
    folder: pathlib.Path,
    image_digest: str,
    base_image_lock: Mapping[str, str],
    signing_key: str | bytes | None,
) -> dict[str, Any]:
    """Writes `image-attestation.json`, binding an image to its provenance.

    Args:
        folder: Service folder holding `provenance.json`, which must exist.
        image_digest: Digest of the built image (the subject).
        base_image_lock: The base image lock record (image and digest).
        signing_key: Signing key, or None to resolve it from the environment.

    Returns:
        The signed (or unsigned, without a key) document that was written.
    """
    provenance_digest = provenance.file_digest(folder / PROVENANCE_FILE)
    body = statement(
        "image",
        image_digest,
        IMAGE_PREDICATE,
        {
            "provenance_sha256": provenance_digest.removeprefix("sha256:"),
            "base_image": dict(base_image_lock),
        },
    )
    document = sign_statement(body, signing_key)
    write_document(folder / IMAGE_ATTESTATION_FILE, document)
    return document


def verify_image(
    folder: pathlib.Path,
    image_digest: str,
    signing_key: str | bytes | None,
    require_signature: bool = True,
) -> dict[str, Any]:
    """Verifies that `image_digest` was attested for this folder.

    Args:
        folder: Service folder holding the attestation and provenance files.
        image_digest: Digest the image is expected to have, as `sha256:<hex>`.
        signing_key: Signing key, or None to resolve it from the environment.
        require_signature: Whether an unsigned or unverifiable attestation is
            refused.

    Returns:
        The verified attestation statement.

    Raises:
        ArtifactError: If the attestation file cannot be read or is not a
            valid statement, the signature is wrong or (when required)
            missing, the image digest differs, or the provenance file is not
            the one attested.
    """
    body, signed = verify_document(
        read_document(folder / IMAGE_ATTESTATION_FILE), signing_key
    )
    if require_signature and not signed:
        raise errors.ArtifactError("image attestation is not verified")
    recorded = body["subject"][0]["digest"].get("sha256")
    if "sha256:" + str(recorded) != image_digest:
        raise errors.ArtifactError(
            "image attestation is for a different image digest"
        )
    attested = body["predicate"].get("provenance_sha256")
    if "sha256:" + str(attested) != provenance.file_digest(
        folder / PROVENANCE_FILE
    ):
        raise errors.ArtifactError("provenance file differs from the attested")
    return body


def main(arguments: list[str] | None = None) -> int:
    """Runs the command line used by the generated `build.sh`.

    Args:
        arguments: Command-line arguments; None reads `sys.argv`.

    Returns:
        0 after the image attestation is written, 1 if an input cannot be read
        or the attestation cannot be made (the error goes to stderr).
    """
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0]  # pyright: ignore[reportOptionalMemberAccess]  # the module docstring is always present
    )
    actions = parser.add_subparsers(dest="action", required=True)
    image = actions.add_parser("image", help="attest a built image")
    image.add_argument("--folder", type=pathlib.Path, required=True)
    image.add_argument("--metadata", type=pathlib.Path, required=True)
    image.add_argument("--lock", type=pathlib.Path, required=True)
    options = parser.parse_args(arguments)
    try:
        metadata = json.loads(options.metadata.read_text(encoding="utf-8"))
        digest = metadata["containerimage.digest"]
        lock = json.loads(options.lock.read_text(encoding="utf-8"))
        document = write_image_attestation(
            options.folder, digest, lock, signing.resolve_key(None)
        )
    except (OSError, KeyError, ValueError, errors.ScmleError) as error:
        print(f"error: {error}", file=sys.stderr)  # noqa: T201 -- command-line output
        return 1
    signed = document["signature"] is not None
    print(f"attested {digest} (signed={signed})")  # noqa: T201 -- command-line output
    return 0


if __name__ == "__main__":
    sys.exit(main())
