"""Bounds and schema versions shared by the modality lifecycle."""

from __future__ import annotations

MAX_REPEATED_EXPLANATION_SAMPLES = 128


MAX_RAW_GRAPH_FEATURES = 64


MAX_GRAPH_CONTEXT_NODES = 64


MAX_GRAPH_CONTEXT_EDGES = 128


PACKAGE_SCHEMA_VERSION = 1


CURATION_SCHEMA_VERSION = 2


SHA256_LENGTH = 64


MATRIX_RANK = 2


CHANNELED_PIXEL_RANK = 3


RGB_CHANNEL_COUNT = 3


REPORT_SCHEMA_VERSION = 3


MAX_SNR_VALUES = 8


MAX_SNR_DB = 120.0


MIN_CLASS_COUNT = 2


MIN_STABILITY_COEFFICIENT = 1e-12


SHORTFALL = (
    "experimental until parity: the modality runs the bounded lifecycle (one "
    "split, one regularized linear classifier), not the shared pipeline "
    "engine, so it has no candidate search, cross-validation, baseline "
    "comparison, prediction intervals, locked test set and vault, or analysis "
    "charter"
)
"""Why every built-in non-tabular modality stays experimental (task 4.5).

A modality loses this reason only when it runs the shared engine's stages
from planning to packaging with the same controls as the tabular path.
"""
