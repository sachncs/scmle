"""The deployment contract every provider adapter must satisfy.

One typed specification says what runs (an image by digest, a model bundle by
digest), how it is configured, which secrets it may read (by reference, never
by value), how much it may use and how it proves it is alive and ready. An
adapter for a platform (local, AWS, GCP) turns that spec into infrastructure
calls and reports what is actually running; `conformance_problems` compares
the two. Adapters differ only in those calls. Nothing here talks to a cloud.

`references` holds the pinned image, bundle and secret references and the
value checks, `deployment` the validated spec, and `runtime` the platform's
report, the conformance comparison and the adapter interface.
"""

from scmle.deploy.contract.deployment import canonize
from scmle.deploy.contract.deployment import Deployment
from scmle.deploy.contract.deployment import ENVIRONMENT_VARIABLE
from scmle.deploy.contract.deployment import harden
from scmle.deploy.contract.deployment import nonsecret
from scmle.deploy.contract.deployment import PORT_LIMIT
from scmle.deploy.contract.deployment import Probes
from scmle.deploy.contract.deployment import PRODUCTION
from scmle.deploy.contract.deployment import Quota
from scmle.deploy.contract.deployment import Resources
from scmle.deploy.contract.deployment import SECRET_WORDS
from scmle.deploy.contract.deployment import SPEC_FORMAT
from scmle.deploy.contract.deployment import TRUSTED_VARIABLE
from scmle.deploy.contract.references import Architecture
from scmle.deploy.contract.references import BundleRef
from scmle.deploy.contract.references import DIGEST_PATTERN
from scmle.deploy.contract.references import Environment
from scmle.deploy.contract.references import ImageRef
from scmle.deploy.contract.references import NAME_PATTERN
from scmle.deploy.contract.references import PRIVATE_KEY_MARKER
from scmle.deploy.contract.references import require_digest
from scmle.deploy.contract.references import require_positive
from scmle.deploy.contract.references import SecretRef
from scmle.deploy.contract.references import SecretSource
from scmle.deploy.contract.references import simple
from scmle.deploy.contract.references import VARIABLE_PATTERN
from scmle.deploy.contract.references import VERSION_PATTERN
from scmle.deploy.contract.runtime import conformance_problems
from scmle.deploy.contract.runtime import diff
from scmle.deploy.contract.runtime import divergences
from scmle.deploy.contract.runtime import mounts
from scmle.deploy.contract.runtime import Provider
from scmle.deploy.contract.runtime import Runtime

__all__ = [
    "DIGEST_PATTERN",
    "ENVIRONMENT_VARIABLE",
    "NAME_PATTERN",
    "PORT_LIMIT",
    "PRIVATE_KEY_MARKER",
    "PRODUCTION",
    "SECRET_WORDS",
    "SPEC_FORMAT",
    "TRUSTED_VARIABLE",
    "VARIABLE_PATTERN",
    "VERSION_PATTERN",
    "Architecture",
    "BundleRef",
    "Deployment",
    "Environment",
    "ImageRef",
    "Probes",
    "Provider",
    "Quota",
    "Resources",
    "Runtime",
    "SecretRef",
    "SecretSource",
    "canonize",
    "conformance_problems",
    "diff",
    "divergences",
    "harden",
    "mounts",
    "nonsecret",
    "require_digest",
    "require_positive",
    "simple",
]
