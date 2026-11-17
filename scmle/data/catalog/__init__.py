"""General-purpose dataset curation and catalog evidence.

This package observes caller-owned data before modelling.  It deliberately
stops at data understanding: it does not infer a task, select a model, or
claim that an arbitrary modality is supported for training.

The package keeps every name the single module catalog.py had, so callers
are unchanged, and the classes in `__all__` keep ``scmle.data.catalog`` as
their module, which the public API snapshot records in `Catalog`'s
signature. It splits by concern: `canon` (canonical JSON and digests),
`identity` (sources), `evidence` (schema, missingness, duplicate and label
summaries), `inspection` (the data observers) and `manifest` (`Catalog`).
"""

# ruff: noqa: F401 - the names below are re-exported for existing callers.

from scmle.data.catalog.canon import canonical_collection
from scmle.data.catalog.canon import canonicalize
from scmle.data.catalog.canon import canonize
from scmle.data.catalog.canon import digest_value
from scmle.data.catalog.canon import normalise
from scmle.data.catalog.evidence import Duplication
from scmle.data.catalog.evidence import Field
from scmle.data.catalog.evidence import Labeling
from scmle.data.catalog.evidence import Missingness
from scmle.data.catalog.identity import assay
from scmle.data.catalog.identity import DIGEST_LENGTH
from scmle.data.catalog.identity import normalize
from scmle.data.catalog.identity import Source
from scmle.data.catalog.inspection import coerce
from scmle.data.catalog.inspection import dedupe
from scmle.data.catalog.inspection import describe
from scmle.data.catalog.inspection import examine
from scmle.data.catalog.inspection import fallback
from scmle.data.catalog.inspection import frame_content
from scmle.data.catalog.inspection import frame_for_mapping
from scmle.data.catalog.inspection import frame_for_pandas
from scmle.data.catalog.inspection import MATRIX_DIMENSIONS
from scmle.data.catalog.inspection import measure
from scmle.data.catalog.inspection import Observation
from scmle.data.catalog.inspection import sequential
from scmle.data.catalog.inspection import summarize
from scmle.data.catalog.inspection import tabulate
from scmle.data.catalog.inspection import tally
from scmle.data.catalog.manifest import Catalog
from scmle.data.catalog.manifest import CATALOG_SCHEMA_VERSION
from scmle.data.catalog.manifest import census

__all__ = [
    "Catalog",
    "Duplication",
    "Field",
    "Labeling",
    "Missingness",
    "Source",
]
