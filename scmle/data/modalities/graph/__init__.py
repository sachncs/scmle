"""Dependency-free graph data contracts, ingestion, and baseline preparation.

This module intentionally stops at a data foundation. It represents a graph,
validates its structural invariants, profiles its contents, creates leakage-
safe inductive node splits, and prepares bounded numeric features. Graph
models, graph explainers, and graph serving are not part of this module.
"""

from scmle.data.modalities.graph.dataset import Dataset
from scmle.data.modalities.graph.elements import Chronology
from scmle.data.modalities.graph.elements import Edge
from scmle.data.modalities.graph.elements import GraphError
from scmle.data.modalities.graph.elements import Issue
from scmle.data.modalities.graph.elements import Node
from scmle.data.modalities.graph.elements import normalize
from scmle.data.modalities.graph.elements import ParseError
from scmle.data.modalities.graph.elements import TemporalError
from scmle.data.modalities.graph.elements import Validation
from scmle.data.modalities.graph.elements import ValidationError
from scmle.data.modalities.graph.evidence import Connectivity
from scmle.data.modalities.graph.evidence import Degrees
from scmle.data.modalities.graph.evidence import Features
from scmle.data.modalities.graph.evidence import Partition
from scmle.data.modalities.graph.evidence import Profile
from scmle.data.modalities.graph.files import edgelist
from scmle.data.modalities.graph.files import link
from scmle.data.modalities.graph.identity import identify
from scmle.data.modalities.graph.preparation import chronicle
from scmle.data.modalities.graph.preparation import debut
from scmle.data.modalities.graph.preparation import featurize
from scmle.data.modalities.graph.preparation import induct
from scmle.data.modalities.graph.preparation import prepare
from scmle.data.modalities.graph.preparation import valence

__all__ = [
    "Chronology",
    "Connectivity",
    "Dataset",
    "Degrees",
    "Edge",
    "Features",
    "GraphError",
    "Issue",
    "Node",
    "ParseError",
    "Partition",
    "Profile",
    "TemporalError",
    "Validation",
    "ValidationError",
    "chronicle",
    "debut",
    "edgelist",
    "featurize",
    "identify",
    "induct",
    "link",
    "normalize",
    "prepare",
    "valence",
]
