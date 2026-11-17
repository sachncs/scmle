"""The axes of a capability: modality, task-level promises and status."""

import enum


class Modality(enum.StrEnum):
    """Data modality named by a capability boundary."""

    TABULAR = "tabular"
    TEXT = "text"
    DOCUMENT = "document"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    GRAPH = "graph"
    GEOSPATIAL = "geospatial"
    RASTER = "raster"


class Evaluation(enum.StrEnum):
    """Evaluation policy required by a capability."""

    STANDARD = "standard"
    CALIBRATED = "calibrated"
    SURVIVAL = "survival"
    RANKING = "ranking"
    FORECASTING = "forecasting"
    UPLIFT = "uplift"
    MULTILABEL = "multilabel"
    MULTIOUTPUT = "multioutput"


class Uncertainty(enum.StrEnum):
    """Uncertainty output promised by a capability."""

    NONE = "none"
    CALIBRATED_PROBABILITY = "calibrated_probability"
    PREDICTION_INTERVAL = "prediction_interval"
    CONFORMAL = "conformal"
    SURVIVAL_CURVE = "survival_curve"
    ABSTENTION = "abstention"


class Fairness(enum.StrEnum):
    """Fairness analysis or mitigation promised by a capability."""

    NONE = "none"
    GROUP_REPORT = "group_report"
    INTERSECTIONAL_REPORT = "intersectional_report"
    MITIGATION = "mitigation"


class Explainability(enum.StrEnum):
    """Explanation contract promised by a capability."""

    NONE = "none"
    GLOBAL = "global"
    LOCAL = "local"
    COUNTERFACTUAL = "counterfactual"
    MODALITY_SPECIFIC = "modality_specific"
    FULL = "full"


class Inference(enum.StrEnum):
    """Inference surface promised by a capability."""

    LIBRARY = "library"
    BATCH = "batch"
    ONNX = "onnx"
    REFERENCE_HTTP = "reference_http"
    STREAMING = "streaming"


class Status(enum.StrEnum):
    """Evidence level assigned to a registered capability."""

    SUPPORTED = "supported"
    EXPERIMENTAL = "experimental"
    EXTENSION_ONLY = "extension-only"


class ModelFamily(enum.StrEnum):
    """Named model families with an explicit support boundary.

    These values describe model-family identity only. A family is usable for a
    run only after a complete :class:`Coordinate` declaration is registered.
    """

    TRANSFORMER = "transformer"
    TEXT_TRANSFORMER = "text_transformer"
    VISION_TRANSFORMER = "vision_transformer"
    AUDIO_TRANSFORMER = "audio_transformer"
    VIDEO_TRANSFORMER = "video_transformer"
    MULTIMODAL_TRANSFORMER = "multimodal_transformer"
    JEPA = "jepa"
