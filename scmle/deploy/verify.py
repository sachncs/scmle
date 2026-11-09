"""Everything a service checks before it deserializes the model.

Bundles are pickle-based and loading one runs code, so the service decides
whether to trust a folder first, using only hashing and JSON: the bundle's
signature, the checksum of every file the manifest lists, the provenance
statement (signature, bundle digest, build materials) and the environment
rules. Only then is the bundle loaded.
"""

from collections.abc import Mapping
import dataclasses
import json
import pathlib

from scmle.artifacts import bundle
from scmle.contracts import errors
from scmle.contracts import signing
from scmle.deploy import attest
from scmle.deploy import contract
from scmle.deploy import provenance

TRUSTED = "SCMLE_TRUSTED"
ENVIRONMENT = "SCMLE_ENV"
SIGNING_KEY = "SCMLE_SIGNING_KEY"


@dataclasses.dataclass(frozen=True, slots=True)
class Verified:
    """What was established about a service folder before loading.

    Attributes:
        bundle_digest: Digest of the bundle manifest.
        bundle_signed: Whether the bundle's signature verified.
        provenance_verified: Whether a provenance statement was present and its
            signature verified.
        trusted: Whether the unsigned bundle is loaded on the strength of
            `SCMLE_TRUSTED=1` (never true for a signed bundle).
        production: Whether the environment declares itself production.
    """

    bundle_digest: str
    bundle_signed: bool
    provenance_verified: bool
    trusted: bool
    production: bool


PUBLISHED_KEYS = frozenset({"scmle-local-demo-signing-key-published-fixture"})


ALLOW_PUBLISHED_KEY = "SCMLE_ALLOW_PUBLISHED_KEY"


def refuse_published_key(
    key: str | None, production: bool, allowed: bool = False
) -> None:
    """Raises when production is given a key that is published in the repo.

    `allowed` is the explicit, named opt-out the local demo sets; nothing else
    should.

    Args:
        key: The signing key in use, or None.
        production: Whether the environment is production.
        allowed: Whether the opt-out was set.

    Raises:
        ArtifactError: If `production` is true, `key` is one of `PUBLISHED_KEYS`
            and `allowed` is false.
    """
    if production and key in PUBLISHED_KEYS and not allowed:
        raise errors.ArtifactError(
            "the signing key is the published demo fixture; configure a "
            f"managed key with ${SIGNING_KEY} or "
            f"${signing.SIGNING_KEY_FILE_ENV}"
        )


def is_production(environ: Mapping[str, str]) -> bool:
    """Reports whether the environment declares itself production.

    Args:
        environ: The environment; `SCMLE_ENV` is read.

    Returns:
        True if `SCMLE_ENV` is `production`, ignoring case and surrounding
        blanks.
    """
    return environ.get(ENVIRONMENT, "").strip().lower() == contract.PRODUCTION


def refuse_trusted_in_production(environ: Mapping[str, str]) -> None:
    """Raises if `SCMLE_TRUSTED=1` is set in a production environment.

    Args:
        environ: The environment.

    Raises:
        ArtifactError: If `SCMLE_TRUSTED` is `1` and `SCMLE_ENV` is
            `production`.
    """
    if environ.get(TRUSTED) == "1" and is_production(environ):
        raise errors.ArtifactError(
            f"{TRUSTED}=1 is refused when {ENVIRONMENT}=production: sign the "
            "bundle and supply the signing key instead"
        )


def check_listed_files(model_directory: pathlib.Path) -> None:
    """Checks the checksum of every file the manifest lists, unparsed.

    Args:
        model_directory: Bundle directory with its manifest.

    Raises:
        ArtifactError: If the manifest cannot be read or lists no files, a
            listed file is missing, or a checksum differs.
    """
    try:
        manifest = json.loads(
            (model_directory / bundle.MANIFEST_FILE).read_text(encoding="utf-8")
        )
        listed = manifest["files"]
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError) as error:
        raise errors.ArtifactError("invalid model bundle metadata") from error
    for name, recorded in sorted(listed.items()):
        path = model_directory / name
        if not path.is_file():
            raise errors.ArtifactError(f"missing {name}")
        if bundle.file_sha256(path) != recorded:
            raise errors.ArtifactError(
                f"checksum mismatch: {name} was modified or corrupted"
            )


def verify_service(
    folder: str | pathlib.Path, environ: Mapping[str, str]
) -> Verified:
    """Verifies a service folder without deserializing anything.

    In production the bundle must be signed, the provenance statement must be
    present and verify, and `SCMLE_TRUSTED=1` is refused. Elsewhere an unsigned
    bundle is accepted only with `SCMLE_TRUSTED=1`.

    Args:
        folder: Service folder with a `model` directory and, optionally, the
            provenance statement.
        environ: The environment (`SCMLE_ENV`, `SCMLE_TRUSTED`, the signing key
            settings).

    Returns:
        What was established about the folder.

    Raises:
        ArtifactError: On any failed check; nothing has been loaded.
    """
    refuse_trusted_in_production(environ)
    root = pathlib.Path(folder)
    model_directory = root / "model"
    production = is_production(environ)
    key = signing.key_from_environment(environ)
    refuse_published_key(
        key, production, environ.get(ALLOW_PUBLISHED_KEY) == "1"
    )
    digest = provenance.bundle_digest(model_directory)
    signed = bundle.verify_signature(model_directory, key)
    trusted = environ.get(TRUSTED) == "1"
    if not signed and (production or not trusted):
        raise errors.ArtifactError(
            "refusing to load an unauthenticated bundle: it is pickle-based "
            f"and loading executes code. Sign the bundle and set "
            f"${SIGNING_KEY}"
        )
    check_listed_files(model_directory)
    verified = False
    statement_path = root / attest.PROVENANCE_FILE
    if statement_path.exists():
        body, verified = attest.verify_document(
            attest.read_document(statement_path), key
        )
        subject = "sha256:" + str(body["subject"][0]["digest"].get("sha256"))
        if subject != digest:
            raise errors.ArtifactError(
                "provenance names a different bundle digest than the one "
                "in the folder"
            )
        problems = attest.material_problems(root, body)
        if problems:
            raise errors.ArtifactError(
                "build materials changed since packaging: "
                + "; ".join(problems)
            )
    if production and not verified:
        raise errors.ArtifactError(
            "production requires a signed provenance statement that verifies"
        )
    return Verified(
        digest, signed, verified, trusted and not signed, production
    )
