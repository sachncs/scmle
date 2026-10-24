"""SCMLE: mathematics meets engineering for trustworthy ML systems."""

__version__ = "0.1.0b1"

from scmle.api.non_tabular import NonTabularModelCapability
from scmle.api.non_tabular import NonTabularModelProject
from scmle.api.non_tabular import NonTabularModelResult
from scmle.api.non_tabular import UnsupportedNonTabularModelError
from scmle.api.predict import load_model
from scmle.api.predict import predict
from scmle.api.project import MLProject
from scmle.api.project import ModalityCurationProject
from scmle.api.project import ModalityProject
from scmle.api.results import RunResult
from scmle.api.sequential import SequentialEstimationProject
from scmle.artifacts.bundle import ModelBundle
from scmle.data.catalog import DatasetCatalog
from scmle.evaluation.longitudinal import LongitudinalEstimand
from scmle.evaluation.longitudinal import LongitudinalEstimationPolicy
from scmle.evaluation.longitudinal import LongitudinalEstimationResult
from scmle.evaluation.longitudinal import LongitudinalIPWEstimator
from scmle.evaluation.longitudinal import PeriodATE
from scmle.evaluation.metrics import register_metric
from scmle.evaluation.parameter_estimation import BayesianPrior
from scmle.evaluation.parameter_estimation import EstimandSpec
from scmle.evaluation.parameter_estimation import estimate_parameters
from scmle.evaluation.parameter_estimation import EstimationPolicy
from scmle.evaluation.parameter_estimation import EstimationResult
from scmle.evaluation.parameter_estimation import ParameterEstimate
from scmle.evaluation.sequential import SequentialEstimand
from scmle.evaluation.sequential import SequentialEstimationPolicy
from scmle.evaluation.sequential import SequentialEstimationResult
from scmle.evaluation.sequential import SequentialIPWEstimator
from scmle.evaluation.sequential import SequentialPeriodDiagnostics
from scmle.evaluation.sequential import SequentialRegimeEstimate
from scmle.inference.modality_serving import ModalityHttpServing
from scmle.inference.modality_serving import ModalitySampleDecoder
from scmle.models.custom import register_estimator
from scmle.models.custom import unregister_estimator
from scmle.models.modality_lifecycle import ExtensionCurationAdapter
from scmle.models.modality_lifecycle import LifecycleCancellation
from scmle.models.modality_lifecycle import LifecycleCancelledError
from scmle.models.modality_lifecycle import LifecycleCheckpoint
from scmle.models.modality_lifecycle import ModalityCurationAdapter
from scmle.models.modality_lifecycle import ModalityCurationResult
from scmle.models.modality_lifecycle import ModalityPredictor

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
