"""Signed curation evidence and the curator and preparer interfaces."""

from __future__ import annotations

import abc
from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
import functools
import hmac
import json
from pathlib import Path
from typing import Any
from typing import cast
from typing import ClassVar
from typing import Self
from typing import TYPE_CHECKING

from scmle.contracts import errors
from scmle.contracts import modalities
from scmle.contracts import records
from scmle.contracts import signing
from scmle.data.modalities import curation
from scmle.data.modalities import document
from scmle.evaluation import envelope
from scmle.governance import attestation
from scmle.pipeline.modality_lifecycle import constants
from scmle.pipeline.modality_lifecycle import containers

if TYPE_CHECKING:
    import numpy
    from sklearn.linear_model import LogisticRegression

    from scmle.explainability import quality


DATASET = {"digests": {"dataset_hash": "dataset"}}
MIGRATION = records.Migration(
    renames={"sample_count": "samples", "dataset_card": "card"},
    change=functools.partial(envelope.fold, groups=DATASET),
    revert=functools.partial(envelope.unfold, groups=DATASET),
)
VERSIONS = (records.ORIGIN, constants.CURATION_SCHEMA_VERSION)


def unseal(signature: object) -> str:
    """Restores the stored signature value of a curation record.

    Args:
        signature: The record's `signature` entry.

    Returns:
        The signature value.

    Raises:
        ValueError: If the entry is missing, names another algorithm or
            holds a value that is not a SHA-256 digest.
    """
    if not isinstance(signature, Mapping):
        raise errors.InputError("modality curation signature is missing")
    if signature.get("algorithm") != signing.SIGNATURE_ALGORITHM:
        raise errors.InputError("unsupported modality curation signature")
    value = signature.get("value")
    if not isinstance(value, str) or not containers.hashlike(value):
        raise errors.InputError("modality curation signature is invalid")
    return value


@dataclasses.dataclass(frozen=True, slots=True)
class Curation:
    """Signed curation evidence for a dataset without a model claim.

    ``digest`` is the SHA-256 digest of the canonical unsigned
    evidence. It is deliberately independent of the signing key so that
    consumers can compare curation content across attestations, and it is
    anchored at schema version 1: it is computed over the evidence as
    version 1 laid it out, whatever version the record is stored at. The
    signature covers the layout of the version the record is stored at, so
    a version-1 record still verifies after it is loaded.

    Attributes:
        modality: Modality value of the dataset.
        digests: The hash identifying the dataset, under `dataset`.
        samples: Number of samples curated; non-negative.
        curation: Curation report evidence as a JSON-native mapping.
        card: Dataset card evidence as a JSON-native mapping.
        path: Where the evidence file is written.
        signature: SHA-256 signature of the unsigned evidence.
        schema_version: Version of the layout the record is stored and
            signed in.
    """

    modality: str
    digests: Mapping[str, str]
    samples: int
    curation: Mapping[str, object]
    card: Mapping[str, object]
    path: str
    signature: str
    schema_version: int = constants.CURATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validates curation evidence identity before it is published.

        Raises:
            ValueError: If an identity field is blank, the digests are not
                exactly the dataset's, the sample count is not a
                non-negative integer, the schema version is unknown, the
                signature is not a SHA-256 digest, or the curation or
                dataset card disagrees with the modality, dataset hash or
                sample count.
        """
        if set(self.digests) != {"dataset"}:
            raise errors.InputError("digests must hold exactly the dataset")
        identity: tuple[tuple[str, object], ...] = (
            ("modality", self.modality),
            ("digests.dataset", self.digests["dataset"]),
            ("path", self.path),
        )
        for field_name, value in identity:
            if not isinstance(value, str) or not value.strip():
                raise errors.InputError(f"{field_name} must be non-empty")
        samples: object = self.samples
        signature: object = self.signature
        if (
            isinstance(samples, bool)
            or not isinstance(samples, int)
            or samples < 0
        ):
            raise errors.InputError("samples must be a non-negative integer")
        if self.schema_version not in VERSIONS:
            raise errors.InputError(
                "unsupported modality curation schema version"
            )
        if not isinstance(signature, str) or not containers.hashlike(signature):
            raise errors.InputError("signature must be a SHA-256 digest")
        self.agree()

    def agree(self) -> None:
        """Normalises the curation and card and checks they match the record.

        Raises:
            ValueError: If the curation or dataset card disagrees with the
                modality, dataset hash or sample count, or is not finite
                JSON.
        """
        roundtrip = containers.normalise(self.curation, "curation")
        normalized_card = containers.normalise(self.card, "card")
        for name, record in (
            ("curation", roundtrip),
            ("card", normalized_card),
        ):
            if "modality" in record and record.get("modality") != self.modality:
                raise errors.InputError(f"{name} does not match modality")
            digest, samples = curation.identify(record)
            if digest != self.digests["dataset"]:
                raise errors.InputError(f"{name} does not match dataset_hash")
            if samples != self.samples:
                raise errors.InputError(f"{name} does not match sample_count")
        object.__setattr__(self, "digests", dict(self.digests))
        object.__setattr__(self, "curation", roundtrip)
        object.__setattr__(self, "card", normalized_card)

    def layout(self, version: int) -> records.Layout:
        """Builds the unsigned evidence laid out as at a schema version.

        Args:
            version: 1 or the current version.

        Returns:
            JSON-native mapping of the version, record type, identity,
            curation and dataset card.
        """
        content: records.Layout = {
            "record_type": "modality_curation",
            "modality": self.modality,
            "digests": dict(self.digests),
            "samples": self.samples,
            "curation": dict(self.curation),
            "card": dict(self.card),
        }
        if version == records.ORIGIN:
            content = MIGRATION.downgrade(content)
        return {records.KEY: version, **content}

    def preimage(self) -> dict[str, object]:
        """Builds the exact curation content covered by the signature.

        Returns:
            The unsigned evidence laid out as at `schema_version`.
        """
        return self.layout(self.schema_version)

    @property
    def digest(self) -> str:
        """The deterministic digest of the unsigned evidence at version 1."""
        return attestation.checksum(self.layout(records.ORIGIN))

    def verify_signature(self, signing_key: str | bytes) -> bool:
        """Checks the persisted curation signature.

        An empty string key raises ValueError from the signing helper.

        Args:
            signing_key: Key the signature should have been made with.

        Returns:
            True if the signature matches the unsigned evidence, else False.
        """
        expected = signing.mac(self.preimage(), signing_key)
        return hmac.compare_digest(self.signature, expected)

    def to_dict(self) -> dict[str, object]:
        """Converts the result to JSON-native curation evidence.

        Returns:
            The unsigned content, laid out as at `schema_version`, plus the
            content digest (`digest`; `content_digest` at version 1), path
            and a `signature` entry holding the algorithm and value.
        """
        record = self.preimage()
        name = (
            "content_digest"
            if self.schema_version == records.ORIGIN
            else "digest"
        )
        record[name] = self.digest
        record["path"] = self.path
        record["signature"] = {
            "algorithm": signing.SIGNATURE_ALGORITHM,
            "value": self.signature,
        }
        return record

    @classmethod
    def from_dict(cls, record: Mapping[str, object]) -> Self:
        """Restores a curation result without trusting its signature claim.

        A version-1 record is migrated to the current field names but keeps
        its stored version, so its signature is checked over the layout it
        was signed in.

        Args:
            record: Mapping produced by `to_dict`.

        Returns:
            The restored result.

        Raises:
            ValueError: If the schema version, record type, identity, evidence,
                signature or content digest is unsupported, malformed or does
                not match.
        """
        version = record.get(records.KEY)
        if (
            isinstance(version, bool)
            or not isinstance(version, int)
            or version not in VERSIONS
        ):
            raise errors.InputError(
                "unsupported modality curation schema version"
            )
        layout = dict(record)
        del layout[records.KEY]
        if version == records.ORIGIN:
            layout = MIGRATION.upgrade(layout)
        if layout.get("record_type") != "modality_curation":
            raise errors.InputError("invalid modality curation record type")
        modality = layout.get("modality")
        digests = layout.get("digests")
        path = layout.get("path")
        samples = layout.get("samples")
        if not isinstance(digests, Mapping) or not all(
            isinstance(value, str) and value.strip()
            for value in (modality, digests.get("dataset"), path)
        ):
            raise errors.InputError("modality curation identity is invalid")
        if isinstance(samples, bool) or not isinstance(samples, int):
            raise errors.InputError("modality curation samples is invalid")
        evidence = layout.get("curation")
        card = layout.get("card")
        if not isinstance(evidence, Mapping) or not isinstance(card, Mapping):
            raise errors.InputError("modality curation evidence is invalid")
        result = cls(
            modality=cast("str", modality),
            digests=cast("Mapping[str, str]", digests),
            samples=samples,
            curation=evidence,
            card=card,
            path=cast("str", path),
            signature=unseal(layout.get("signature")),
            schema_version=version,
        )
        digest = layout.get("digest", layout.get("content_digest"))
        if digest is not None:
            result.attest(digest)
        return result

    def attest(self, digest: object) -> None:
        """Checks a persisted content digest against the evidence.

        Args:
            digest: The persisted `digest` entry (`content_digest` before
                the key was renamed).

        Raises:
            ValueError: If it is not a SHA-256 digest or differs from the
                digest of this result's content.
        """
        if not isinstance(digest, str) or not containers.hashlike(digest):
            raise errors.InputError("modality curation digest is invalid")
        if not hmac.compare_digest(digest, self.digest):
            raise errors.InputError(
                "modality curation digest does not match evidence"
            )

    @classmethod
    def read(cls, path: str | Path) -> Self:
        """Reads a persisted curation result.

        Args:
            path: JSON file written from `to_dict`.

        Returns:
            The parsed result (signature not yet verified).

        Raises:
            ValueError: If the file does not hold a JSON object.
        """
        record = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(record, Mapping):
            raise errors.InputError(
                "modality curation record must be an object"
            )
        return cls.from_dict(record)

    @classmethod
    def read_verified(cls, path: str | Path, signing_key: str | bytes) -> Self:
        """Reads curation evidence and requires a valid content signature.

        Args:
            path: JSON file written from `to_dict`.
            signing_key: Key the signature should have been made with.

        Returns:
            The parsed result with a verified signature.

        Raises:
            ValueError: If the signature does not verify.
        """
        result = cls.read(path)
        if not result.verify_signature(signing_key):
            raise errors.InputError(
                "modality curation signature does not verify"
            )
        return result


class Curator(abc.ABC):
    """Polymorphic boundary for source identity and data understanding."""

    @property
    @abc.abstractmethod
    def modality(self) -> modalities.ModalityIdentity:
        """The declared dataset family."""

    @property
    @abc.abstractmethod
    def digest(self) -> str:
        """The immutable dataset identity."""

    @property
    @abc.abstractmethod
    def samples(self) -> tuple[Any, ...]:
        """The samples in canonical source order."""

    @abc.abstractmethod
    def curate(
        self,
    ) -> curation.Curation | document.Curation:
        """Builds reviewable data-understanding evidence.

        Returns:
            The curation report for the dataset.
        """

    @abc.abstractmethod
    def remarks(self) -> tuple[str, ...]:
        """Lists representation and curation limits.

        Returns:
            Plain statements of what the representation and curation do not
            cover.
        """


@dataclasses.dataclass(frozen=True, slots=True)
class ExtensionCurator(Curator):
    """Adapts arbitrary caller-owned records to the curation-only contract.

    The callback owns domain-specific profiling and validation. This adapter
    binds its result to the extension identity, immutable dataset identity,
    and sample count before the signed curation project can publish evidence.
    It intentionally does not provide split, feature, label, or model hooks.

    Attributes:
        extension: Namespaced extension modality identity.
        sample_values: Caller-owned records in canonical order.
        dataset_hash_value: Caller-owned immutable dataset identity.
        curation_callback: Callable that returns the curation report.
        limitations: Non-empty statements of the dataset's boundaries.
    """

    extension: modalities.Bespoke
    sample_values: tuple[Any, ...]
    dataset_hash_value: str
    curation_callback: Callable[[], curation.Curation]
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validates the caller-owned curation boundary.

        Raises:
            TypeError: If `extension` is not an `Bespoke` or the
                callback is not callable.
            ValueError: If the dataset hash is blank or the limitations are
                empty or contain blank values.
        """
        if not isinstance(self.extension, modalities.Bespoke):
            raise errors.KindError("extension must be Bespoke")
        if not self.dataset_hash_value.strip():
            raise errors.InputError("dataset_hash_value must not be empty")
        if not callable(self.curation_callback):
            raise errors.KindError("curation_callback must be callable")
        if not self.limitations or any(
            not value.strip() for value in self.limitations
        ):
            raise errors.InputError("limitations must contain non-empty values")
        object.__setattr__(self, "sample_values", tuple(self.sample_values))
        object.__setattr__(self, "limitations", tuple(self.limitations))

    @property
    def modality(self) -> modalities.Bespoke:
        """The namespaced extension identity."""
        return self.extension

    @property
    def digest(self) -> str:
        """The caller-owned immutable dataset identity."""
        return self.dataset_hash_value

    @property
    def samples(self) -> tuple[Any, ...]:
        """The records in caller-defined canonical order."""
        return self.sample_values

    def curate(
        self,
    ) -> curation.Curation:
        """Runs and validates the caller-owned curation callback.

        Returns:
            The report returned by the callback.

        Raises:
            TypeError: If the callback does not return a curation report.
            ValueError: If the report's modality, dataset hash or sample count
                does not match the extension.
        """
        report = self.curation_callback()
        if not isinstance(report, curation.Curation):
            raise errors.KindError(
                "curation_callback must return a curation report"
            )
        expected_modality = modalities.serialise(self.extension)
        if report.modality != expected_modality:
            raise errors.InputError(
                "curation report modality does not match extension"
            )
        if report.dataset_hash != self.dataset_hash_value:
            raise errors.InputError(
                "curation report hash does not match extension"
            )
        if report.sample_count != len(self.sample_values):
            raise errors.InputError(
                "curation report count does not match extension"
            )
        return report

    def remarks(self) -> tuple[str, ...]:
        """Lists the explicit caller-owned boundaries for the dataset card.

        Returns:
            The limitations given at construction.
        """
        return self.limitations


class Preparer(Curator):
    """Polymorphic boundary between a dataset family and the lifecycle."""

    modality: ClassVar[modalities.ModalityIdentity]  # pyright: ignore[reportIncompatibleMethodOverride]  # each adapter sets a class attribute, which satisfies the abstract Curator.modality property

    @abc.abstractmethod
    def split(self, seed: int) -> containers.Partitions:
        """Splits the dataset into leakage-aware partitions.

        Args:
            seed: Random seed for the split.

        Returns:
            Train, validation and test partitions.
        """

    @abc.abstractmethod
    def features(self, samples: Sequence[Any]) -> containers.Batch:
        """Builds bounded features for the supplied samples.

        Args:
            samples: Samples to featurize.

        Returns:
            Feature batch with one row per sample, in order.
        """

    @abc.abstractmethod
    def sample_id(self, sample: Any) -> str:
        """Gives the stable identity of one sample.

        Args:
            sample: Sample to identify.

        Returns:
            The sample's identifier.
        """

    @abc.abstractmethod
    def label(self, sample: Any) -> object | None:
        """Gives the supervised label of one sample.

        Args:
            sample: Sample to label.

        Returns:
            The label, or None if the sample has none.
        """

    @abc.abstractmethod
    def monitor(self, observed: Preparer) -> dict[str, object]:
        """Compares this reference adapter with an observed adapter.

        Args:
            observed: Adapter over the newly observed data.

        Returns:
            Drift evidence as a JSON-native mapping.
        """

    def elucidate(
        self,
        sample: Any,  # noqa: ARG002 -- hook signature that subclasses override
        model: LogisticRegression,  # noqa: ARG002 -- hook signature that subclasses override
        feature_names: Sequence[str],  # noqa: ARG002 -- hook signature that subclasses override
        coefficients: numpy.ndarray,  # noqa: ARG002 -- hook signature that subclasses override
    ) -> dict[str, object]:
        """Builds explicit modality-specific explanation evidence.

        The default is an honest unavailable result. Adapters override this
        hook only when they own a decoder or perturbation policy that can be
        represented and tested without inventing raw-modality semantics.

        Args:
            sample: Sample to explain.
            model: Fitted classifier.
            feature_names: Names of the model's features.
            coefficients: Linear coefficients for the predicted class, one per
                feature.

        Returns:
            Mapping with a "status" key; in this base class always
            "not_available" with the reason and limitations.
        """
        return {
            "status": "not_available",
            "modality": modalities.serialise(self.modality),
            "reason": "the adapter has no modality-specific explanation policy",
            "limitations": [
                "The lifecycle explanation remains in feature space.",
                "No raw-modality or causal interpretation is provided.",
            ],
        }

    def explanation_ground_truth(
        self,
        sample: Any,
        feature_names: Sequence[str],
    ) -> quality.Truth | None:
        """Gives caller-owned independent explanation evidence, if known.

        Built-in adapters return no reference by default because raw-modality
        semantics cannot be inferred from a feature matrix. A domain adapter
        may override this method when it carries an independently authored
        annotation, benchmark, or synthetic oracle.

        Args:
            sample: Sample being explained.
            feature_names: Names of the model's features.

        Returns:
            The ground truth for the sample, or None if none is known (always
            None in this base class).
        """
        del sample, feature_names
        return None

    def labels(self, samples: Sequence[Any]) -> tuple[object, ...]:
        """Gives the labels of several samples.

        Args:
            samples: Samples to label.

        Returns:
            Labels in sample order.
        """
        return tuple(self.label(sample) for sample in samples)

    def cohorts(self, samples: Sequence[Any]) -> tuple[str, ...]:
        """Gives the stable grouping identities used for split evidence.

        Args:
            samples: Samples to identify.

        Returns:
            The sample identifier of each sample, in order.
        """
        return tuple(self.sample_id(sample) for sample in samples)


def drift(
    reference: curation.Curation, current: curation.Curation
) -> dict[str, object]:
    """Compares the curation of reference and observed data.

    Args:
        reference: Curation of the reference dataset.
        current: Curation of the newly observed dataset.

    Returns:
        Mapping with both dataset hashes, the sample count change (observed
        minus reference) and, per profile key, whether its value differs.
    """
    return {
        "reference_hash": reference.dataset_hash,
        "observed_hash": current.dataset_hash,
        "sample_count_delta": current.sample_count - reference.sample_count,
        "profile_delta": {
            key: current.profile.get(key) != reference.profile.get(key)
            for key in sorted(set(current.profile) | set(reference.profile))
        },
    }


def unavailable(method: str, reason: str) -> dict[str, object]:
    """Reports that one explanation method cannot run on a sample.

    Args:
        method: The explanation method.
        reason: Why it cannot run.

    Returns:
        Mapping with status "not_available", the method and the reason.
    """
    return {"status": "not_available", "method": method, "reason": reason}
