"""scml: autonomous, leakage-safe ML lifecycle framework."""

__version__ = "0.1.0b1"

from scml.api.non_tabular import NonTabularModelCapability
from scml.api.non_tabular import NonTabularModelProject
from scml.api.non_tabular import NonTabularModelResult
from scml.api.non_tabular import UnsupportedNonTabularModelError
from scml.api.predict import load_model
from scml.api.predict import predict
from scml.api.project import MLProject
from scml.api.project import ModalityCurationProject
from scml.api.project import ModalityProject
from scml.api.results import RunResult
from scml.api.sequential import SequentialEstimationProject
from scml.artifacts.bundle import ModelBundle
from scml.data.catalog import DatasetCatalog
from scml.evaluation.longitudinal import LongitudinalEstimand
from scml.evaluation.longitudinal import LongitudinalEstimationPolicy
from scml.evaluation.longitudinal import LongitudinalEstimationResult
from scml.evaluation.longitudinal import LongitudinalIPWEstimator
from scml.evaluation.longitudinal import PeriodATE
from scml.evaluation.metrics import register_metric
from scml.evaluation.parameter_estimation import BayesianPrior
from scml.evaluation.parameter_estimation import EstimandSpec
from scml.evaluation.parameter_estimation import estimate_parameters
from scml.evaluation.parameter_estimation import EstimationPolicy
from scml.evaluation.parameter_estimation import EstimationResult
from scml.evaluation.parameter_estimation import ParameterEstimate
from scml.evaluation.sequential import SequentialEstimand
from scml.evaluation.sequential import SequentialEstimationPolicy
from scml.evaluation.sequential import SequentialEstimationResult
from scml.evaluation.sequential import SequentialIPWEstimator
from scml.evaluation.sequential import SequentialPeriodDiagnostics
from scml.evaluation.sequential import SequentialRegimeEstimate
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
    "BayesianPrior",
    "DatasetCatalog",
    "EstimandSpec",
    "EstimationPolicy",
    "EstimationResult",
    "ExtensionCurationAdapter",
    "LifecycleCancellation",
    "LifecycleCancelledError",
    "LifecycleCheckpoint",
    "LongitudinalEstimand",
    "LongitudinalEstimationPolicy",
    "LongitudinalEstimationResult",
    "LongitudinalIPWEstimator",
    "MLProject",
    "ModalityCurationAdapter",
    "ModalityCurationProject",
    "ModalityCurationResult",
    "ModalityHttpServing",
    "ModalityPredictor",
    "ModalityProject",
    "ModalitySampleDecoder",
    "ModelBundle",
    "NonTabularModelCapability",
    "NonTabularModelProject",
    "NonTabularModelResult",
    "ParameterEstimate",
    "PeriodATE",
    "RunResult",
    "SequentialEstimand",
    "SequentialEstimationPolicy",
    "SequentialEstimationProject",
    "SequentialEstimationResult",
    "SequentialIPWEstimator",
    "SequentialPeriodDiagnostics",
    "SequentialRegimeEstimate",
    "UnsupportedNonTabularModelError",
    "__version__",
    "estimate_parameters",
    "load_model",
    "predict",
    "register_estimator",
    "register_metric",
    "unregister_estimator",
]
