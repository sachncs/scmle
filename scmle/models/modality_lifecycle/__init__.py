"""Complete, bounded baseline lifecycles for non-tabular datasets.

The lifecycle is deliberately explicit about what a modality baseline sees.
Each adapter owns the modality-specific validation, split, and representation
policy; the stage runner owns the shared evidence, model, packaging,
inference, monitoring, documentation, and recovery contracts. The baseline
model is a regularized linear classifier. It is useful for proving lifecycle
integrity, not for claiming state-of-the-art representation quality.
"""

from scmle.models.modality_lifecycle.checkpoints import Checkpoint
from scmle.models.modality_lifecycle.constants import CHANNELED_PIXEL_RANK
from scmle.models.modality_lifecycle.constants import CURATION_SCHEMA_VERSION
from scmle.models.modality_lifecycle.constants import MATRIX_RANK
from scmle.models.modality_lifecycle.constants import MAX_GRAPH_CONTEXT_EDGES
from scmle.models.modality_lifecycle.constants import MAX_GRAPH_CONTEXT_NODES
from scmle.models.modality_lifecycle.constants import MAX_RAW_GRAPH_FEATURES
from scmle.models.modality_lifecycle.constants import (
    MAX_REPEATED_EXPLANATION_SAMPLES,
)
from scmle.models.modality_lifecycle.constants import MAX_SNR_DB
from scmle.models.modality_lifecycle.constants import MAX_SNR_VALUES
from scmle.models.modality_lifecycle.constants import MIN_CLASS_COUNT
from scmle.models.modality_lifecycle.constants import MIN_STABILITY_COEFFICIENT
from scmle.models.modality_lifecycle.constants import PACKAGE_SCHEMA_VERSION
from scmle.models.modality_lifecycle.constants import REPORT_SCHEMA_VERSION
from scmle.models.modality_lifecycle.constants import RGB_CHANNEL_COUNT
from scmle.models.modality_lifecycle.constants import SHA256_LENGTH
from scmle.models.modality_lifecycle.containers import Batch
from scmle.models.modality_lifecycle.containers import Cancellation
from scmle.models.modality_lifecycle.containers import hashlike
from scmle.models.modality_lifecycle.containers import LifecycleCancelledError
from scmle.models.modality_lifecycle.containers import normalise
from scmle.models.modality_lifecycle.containers import Package
from scmle.models.modality_lifecycle.containers import Partitions
from scmle.models.modality_lifecycle.curators import Curation
from scmle.models.modality_lifecycle.curators import Curator
from scmle.models.modality_lifecycle.curators import ExtensionCurator
from scmle.models.modality_lifecycle.curators import Preparer
from scmle.models.modality_lifecycle.explanation import ablate
from scmle.models.modality_lifecycle.explanation import ModelExplainabilityStage
from scmle.models.modality_lifecycle.explanation import perturb
from scmle.models.modality_lifecycle.explanation import replicate
from scmle.models.modality_lifecycle.explanation import sensitivities
from scmle.models.modality_lifecycle.explanation import Stability
from scmle.models.modality_lifecycle.lifecycle import clocked
from scmle.models.modality_lifecycle.lifecycle import Lifecycle
from scmle.models.modality_lifecycle.plans import coherent
from scmle.models.modality_lifecycle.plans import Plan
from scmle.models.modality_lifecycle.plans import typecheck
from scmle.models.modality_lifecycle.predictors import Predictor
from scmle.models.modality_lifecycle.preparers.audio import AudioPreparer
from scmle.models.modality_lifecycle.preparers.document import DocumentPreparer
from scmle.models.modality_lifecycle.preparers.document import Plaintext
from scmle.models.modality_lifecycle.preparers.geospatial import (
    GeospatialPreparer,
)
from scmle.models.modality_lifecycle.preparers.graph import GraphPreparer
from scmle.models.modality_lifecycle.preparers.image import ImagePreparer
from scmle.models.modality_lifecycle.preparers.raster import Imagery
from scmle.models.modality_lifecycle.preparers.video import Footage
from scmle.models.modality_lifecycle.reporting import Record
from scmle.models.modality_lifecycle.reporting import rehydrate
from scmle.models.modality_lifecycle.reporting import require
from scmle.models.modality_lifecycle.reporting import unpack
from scmle.models.modality_lifecycle.reporting import unseal
from scmle.models.modality_lifecycle.state import Context
from scmle.models.modality_lifecycle.state import Stage
from scmle.models.modality_lifecycle.steps import Composition
from scmle.models.modality_lifecycle.steps import Curating
from scmle.models.modality_lifecycle.steps import Documentation
from scmle.models.modality_lifecycle.steps import Evaluation
from scmle.models.modality_lifecycle.steps import Inference
from scmle.models.modality_lifecycle.steps import Ingestion
from scmle.models.modality_lifecycle.steps import Modelling
from scmle.models.modality_lifecycle.steps import Monitoring
from scmle.models.modality_lifecycle.steps import Packaging
from scmle.models.modality_lifecycle.steps import Preparation
from scmle.models.modality_lifecycle.steps import Profiling
from scmle.models.modality_lifecycle.steps import Recovery
from scmle.models.modality_lifecycle.steps import Splitting
from scmle.models.modality_lifecycle.steps import Validation
from scmle.models.modality_lifecycle.verification import authenticate
from scmle.models.modality_lifecycle.verification import file_digest
from scmle.models.modality_lifecycle.verification import load_package
from scmle.models.modality_lifecycle.verification import require_feature_names
from scmle.models.modality_lifecycle.verification import verify_package_manifest
from scmle.models.modality_lifecycle.verification import vet

__all__ = [
    "CHANNELED_PIXEL_RANK",
    "CURATION_SCHEMA_VERSION",
    "MATRIX_RANK",
    "MAX_GRAPH_CONTEXT_EDGES",
    "MAX_GRAPH_CONTEXT_NODES",
    "MAX_RAW_GRAPH_FEATURES",
    "MAX_REPEATED_EXPLANATION_SAMPLES",
    "MAX_SNR_DB",
    "MAX_SNR_VALUES",
    "MIN_CLASS_COUNT",
    "MIN_STABILITY_COEFFICIENT",
    "PACKAGE_SCHEMA_VERSION",
    "REPORT_SCHEMA_VERSION",
    "RGB_CHANNEL_COUNT",
    "SHA256_LENGTH",
    "AudioPreparer",
    "Batch",
    "Cancellation",
    "Checkpoint",
    "Composition",
    "Context",
    "Curating",
    "Curation",
    "Curator",
    "DocumentPreparer",
    "Documentation",
    "Evaluation",
    "ExtensionCurator",
    "Footage",
    "GeospatialPreparer",
    "GraphPreparer",
    "ImagePreparer",
    "Imagery",
    "Inference",
    "Ingestion",
    "Lifecycle",
    "LifecycleCancelledError",
    "ModelExplainabilityStage",
    "Modelling",
    "Monitoring",
    "Package",
    "Packaging",
    "Partitions",
    "Plaintext",
    "Plan",
    "Predictor",
    "Preparation",
    "Preparer",
    "Profiling",
    "Record",
    "Recovery",
    "Splitting",
    "Stability",
    "Stage",
    "Validation",
    "ablate",
    "authenticate",
    "clocked",
    "coherent",
    "file_digest",
    "hashlike",
    "load_package",
    "normalise",
    "perturb",
    "rehydrate",
    "replicate",
    "require",
    "require_feature_names",
    "sensitivities",
    "typecheck",
    "unpack",
    "unseal",
    "verify_package_manifest",
    "vet",
]
