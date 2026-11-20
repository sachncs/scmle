"""Model bundle: a validated, portable inference pipeline.

SECURITY: bundles are serialized with joblib (pickle). Loading executes code,
so only load bundles from sources you trust. Checksums detect corruption or
tampering after saving, not malice at the source.

`model` holds `Bundle`, built from `core` (its state and plain predictions),
`archive` (saving and verified loading), `intervals`, `explanation` and
`outputs`; `layout`, `provenance`, `seal` and `environment` hold the files,
the manifest, the signature and the build-environment probes.
"""

# Callers reach joblib through this package, as they did through the module
# it replaces (tests/unit/test_service_hardening.py patches its `load`).
import joblib

from scmle.artifacts.bundle.archive import authenticate
from scmle.artifacts.bundle.archive import hydrate
from scmle.artifacts.bundle.archive import resolve
from scmle.artifacts.bundle.archive import restore
from scmle.artifacts.bundle.archive import unpack
from scmle.artifacts.bundle.archive import unpickle
from scmle.artifacts.bundle.environment import COMMIT_LENGTH
from scmle.artifacts.bundle.environment import detect
from scmle.artifacts.bundle.environment import GIT_SECONDS
from scmle.artifacts.bundle.environment import inventory
from scmle.artifacts.bundle.environment import pools
from scmle.artifacts.bundle.environment import revision
from scmle.artifacts.bundle.environment import TRACKED_LIBRARIES
from scmle.artifacts.bundle.explanation import BINARY_CLASSES
from scmle.artifacts.bundle.layout import BUNDLE_FORMAT
from scmle.artifacts.bundle.layout import CALIBRATOR_FILE
from scmle.artifacts.bundle.layout import CONFORMAL_FILE
from scmle.artifacts.bundle.layout import COVERAGE_FILE
from scmle.artifacts.bundle.layout import CV_PLUS_FILE
from scmle.artifacts.bundle.layout import HISTORY_FILE
from scmle.artifacts.bundle.layout import MANIFEST_FILE
from scmle.artifacts.bundle.layout import MODEL_FILE
from scmle.artifacts.bundle.layout import NOVELTY_FILE
from scmle.artifacts.bundle.layout import QUANTILE_FILE
from scmle.artifacts.bundle.layout import REFERENCE_FILE
from scmle.artifacts.bundle.layout import SCHEMA_FILE
from scmle.artifacts.bundle.layout import shasum
from scmle.artifacts.bundle.layout import SHIFT_WEIGHTS_FILE
from scmle.artifacts.bundle.model import brier
from scmle.artifacts.bundle.model import Bundle
from scmle.artifacts.bundle.model import DIRICHLET_MIN_CLASSES
from scmle.artifacts.bundle.model import MAX_KNOWN_CATEGORIES
from scmle.artifacts.bundle.model import PARITY_TOLERANCE
from scmle.artifacts.bundle.provenance import Manifest
from scmle.artifacts.bundle.provenance import truncate
from scmle.artifacts.bundle.seal import sign
from scmle.artifacts.bundle.seal import SIGNATURE_ALGORITHM
from scmle.artifacts.bundle.seal import SIGNATURE_FILE
from scmle.artifacts.bundle.seal import verify

__all__ = [
    "BINARY_CLASSES",
    "BUNDLE_FORMAT",
    "CALIBRATOR_FILE",
    "COMMIT_LENGTH",
    "CONFORMAL_FILE",
    "COVERAGE_FILE",
    "CV_PLUS_FILE",
    "DIRICHLET_MIN_CLASSES",
    "GIT_SECONDS",
    "HISTORY_FILE",
    "MANIFEST_FILE",
    "MAX_KNOWN_CATEGORIES",
    "MODEL_FILE",
    "NOVELTY_FILE",
    "PARITY_TOLERANCE",
    "QUANTILE_FILE",
    "REFERENCE_FILE",
    "SCHEMA_FILE",
    "SHIFT_WEIGHTS_FILE",
    "SIGNATURE_ALGORITHM",
    "SIGNATURE_FILE",
    "TRACKED_LIBRARIES",
    "Bundle",
    "Manifest",
    "authenticate",
    "brier",
    "detect",
    "hydrate",
    "inventory",
    "joblib",
    "pools",
    "resolve",
    "restore",
    "revision",
    "shasum",
    "sign",
    "truncate",
    "unpack",
    "unpickle",
    "verify",
]
