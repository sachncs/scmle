"""scml: autonomous, leakage-safe ML lifecycle framework."""

__version__ = "0.1.0b1"

from scml.api.predict import load_model
from scml.api.predict import predict
from scml.api.project import MLProject
from scml.api.project import ModalityCurationProject
from scml.api.project import ModalityProject
from scml.api.results import RunResult
from scml.artifacts.bundle import ModelBundle
from scml.evaluation.metrics import register_metric
from scml.evaluation.parameter_estimation import EstimandSpec
from scml.evaluation.parameter_estimation import estimate_parameters
from scml.evaluation.parameter_estimation import EstimationPolicy
from scml.evaluation.parameter_estimation import EstimationResult
from scml.evaluation.parameter_estimation import ParameterEstimate
from scml.inference.modality_serving import ModalityHttpServing
from scml.inference.modality_serving import ModalitySampleDecoder
from scml.models.custom import register_estimator
from scml.models.custom import unregister_estimator
from scml.models.modality_lifecycle import ExtensionCurationAdapter
from scml.models.modality_lifecycle import LifecycleCancellation
from scml.models.modality_lifecycle import LifecycleCancelledError
from scml.models.modality_lifecycle import LifecycleCheckpoint
from scml.models.modality_lifecycle import ModalityCurationAdapter
from scml.models.modality_lifecycle import ModalityCurationResult
from scml.models.modality_lifecycle import ModalityPredictor

__all__ = [
    "EstimandSpec",
    "EstimationPolicy",
    "EstimationResult",
    "ExtensionCurationAdapter",
    "LifecycleCancellation",
    "LifecycleCancelledError",
    "LifecycleCheckpoint",
    "MLProject",
    "ModalityCurationAdapter",
    "ModalityCurationProject",
    "ModalityCurationResult",
    "ModalityHttpServing",
    "ModalityPredictor",
    "ModalityProject",
    "ModalitySampleDecoder",
    "ModelBundle",
    "ParameterEstimate",
    "RunResult",
    "__version__",
    "estimate_parameters",
    "load_model",
    "predict",
    "register_estimator",
    "register_metric",
    "unregister_estimator",
]
