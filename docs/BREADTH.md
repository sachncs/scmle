# Breadth: what is qualified, what is partial, what is not supported

No slice is qualified. This page says, for each task family and modality,
what exists in the repository, what does not, and what a slice needs before
it may be called production-supported. It is the audit F15 item 15 asks for,
and it is a status table, not a roadmap.

## The rule

A slice is **qualified** only when all three hold:

1. its robustness, drift, explanation and monitoring benchmark was defined
   before it was built (the files in
   [`benchmarks/breadth/`](../benchmarks/breadth/__init__.py) do that) and
   was then run, with a public or domain benchmark and an independent
   replication;
2. it completes ingestion, evaluation, packaging, deployment,
   monitoring and recovery;
3. its deployment requirement was declared by a named owner.

**Partial** means some code exists and is tested but not all of that. **Not
supported** means nothing in the repository does the job. An unsupported
algorithm is never added to the production registry; the guard test
`tests/unit/test_breadth.py::test_unsupported_algorithms_are_not_in_the_model_registry` fails if one is.

## Status

`Built` is the word in the definition file. `Qualified` is `no` for every
slice. All fourteen benchmark definitions are `not_run`, and every threshold
in them is unset until the owner fixes it.

| Slice | Built | Qualified | What exists | What is missing |
| --- | --- | --- | --- | --- |
| `clustering` | `tabular_beta_path` | no | `scmle.unsupervised.clustering.discover`: k-means, Gaussian mixture and agglomerative at several counts, scored by silhouette, Davies-Bouldin and Calinski-Harabasz, with bootstrap stability (mean adjusted Rand index) and `compare`. A user path: `scmle cluster` and `scmle.api.clustering.cluster` save a bundle of task `clustering` (signed when a key is set) with a `clustering.json` report, `scmle predict` and `Bundle.predict` assign new rows, and `clustering.evaluate` scores a saved bundle on other rows (internal indices, and agreement with known groups when named); unit tests on planted blobs and uniform noise, and an integration test from the API and the CLI to a bundle that predicts | Not in `Project`, the charter run or monitoring, and not a task type of the run. No served deployment proved, no drift, explanation or monitoring benchmark, no robustness benchmark run, no declared use |
| `association` | `none` | no | Nothing | Everything |
| `dimensionality_reduction` | `none` | no | Nothing as an estimator or task. The `select_features` search strategy picks columns for a supervised model; it is not an unsupervised representation | Everything |
| `forecasting` | `tabular_beta_path` | no | `Task.FORECAST`, forecasting estimators, the feature graph's lag and rolling operations, a rolling validation strategy; panel data with an entity and a time column (`scmle.models.forecast.Panel`), rolling-origin evaluation (`scmle.evaluation.forecasting.backtest`) and three families (seasonal naive, exponential smoothing, direct multi-step boosting on lag features): boosting beats seasonal naive on trending seasonal series and is not shown to on a pure season; the tabular-beta capability registry lists baseline, linear, tree and boosting paths as supported with library inference only and no portable explanation or uncertainty contract; `benchmarks/real_tasks.py` compares one real series with last-value and seasonal-naive | Robustness, drift, explanation and monitoring benchmarks; a declared horizon and decision; deployment of a forecaster; an independent replication |
| `ranking` | `tabular_beta_path` | no | `Task.RANKING`, pointwise and LightGBM and XGBoost rankers, NDCG evaluation, the same registry entries and library-only limits | The same list as forecasting |
| `document_nlp` | `experimental_lifecycle` | no | The experimental `document` kind on the shared stages (`scmle run --modality document`): class folders or a listing table of UTF-8 text, Markdown or PDF (PDF needs `pypdf`, the `documents` extra) become one row per document, the text column declared as text so a corpus under 30 documents keeps it; TF-IDF terms are fitted inside the folds, and chunks of one document and exact or near-duplicate content stay on one side of every split and fold. Binary, multiclass and regression on the shared model families, with search, cross-validation, the locked test set, explanation and packaging. The older document and text adapters still run the fifteen-stage local lifecycle with a linear baseline. Extracted elements and citations are data contracts | No OCR and no semantic reading beyond terms and counts. A served model takes described rows, not raw files. No OCR confidence gate, no citation-grounding test, no benchmark, no served deployment |
| `image` | `experimental_lifecycle` | no | The experimental `image` kind on the shared stages (`--modality image`, Pillow from the `multimodal` extra): one row of numeric descriptors per file, computed before the split with no fitted state; near-duplicate islands and declared groups stay on one side of every split and fold. Binary, multiclass and regression on the shared model families; occlusion explanations keep the original class; the bundle is the standard signed one. The older image adapter and its local lifecycle remain | Hand-built descriptors, no learned representation. A served model takes descriptor rows, not raw images. Decoding and perturbation benchmarks, runtime controls proved on a deployment, explanation and drift benchmarks |
| `audio` | `experimental_lifecycle` | no | The experimental `audio` kind on the shared stages (`--modality audio`): a manifest or label folders of WAV clips (FLAC needs `soundfile`) read in bounded chunks into numeric features; copies, near-copies and declared groups (a speaker, a session) stay on one side of every split and fold. Binary, multiclass and regression on the shared model families. The older audio adapter and its local lifecycle remain | No transcript, speaker or event claim. A served model takes feature rows, not raw audio. No noise-robustness or drift benchmark; no deployment |
| `video` | `experimental_lifecycle` | no | The experimental `video` kind on the shared stages (`--modality video`, `ffmpeg` and `ffprobe` on the PATH): evenly spaced frames decoded through an FFmpeg pipe, never the whole file in memory, into descriptor rows; near-duplicate videos split together, and a declared time column gives a time-ordered split with folds. Binary, multiclass and regression on the shared model families. The older video adapter and its local lifecycle remain | The same list as audio, plus decode failure handling at scale. A served model takes descriptor rows, not raw video |
| `geospatial` | `experimental_lifecycle` | no | Two experimental kinds on the shared stages. `raster` (`--modality raster`): `.npy` tiles, or GeoTIFF through `rasterio`, read in bounded strips into feature rows; copies, near-copies, declared groups and spatial blocks anchored at the CRS origin stay on one side of every split and fold; binary, multiclass and regression on the shared model families. `geospatial` vector (`--modality geospatial`): GeoJSON and line-by-line GeoJSON (other formats need `geopandas`) become one row per feature with geometry measures, split by spatial blocks sized from the label's spatial range. Both are experimental and need the opt-in `models.experimental_models`. The older vector and raster adapters and their local lifecycle remain | Vector also has two stated reasons of its own: a served model takes the geometry columns, not raw GeoJSON, and the held-out test interval resamples rows, not spatial blocks, so it can be too narrow. A served raster model takes feature rows. Reprojection invariance, raster operations and spatial joins at scale, benchmarks, deployment |
| `graph` | `experimental_lifecycle` | no | The experimental `graph` kind on the shared stages (`--modality graph`): a node table and an edge file (node ids read as text) become one row per labelled node with its features and structural features (degrees, strength, mean neighbour degree) counted only from edges known when the node is first seen; a timed graph holds out the latest-seen nodes. Binary, multiclass and regression on the shared model families. The older graph adapter and its local lifecycle remain | Structure enters only as per-node counts: no graph model, no graph explainer. A served model takes node feature rows. No benchmark |
| `pretrained_multimodal` | `none` | no | Model-family declarations (transformer, text, vision, audio, video, multimodal, JEPA) that name a boundary without support, and an experimental tabular FT-Transformer and tabular JEPA behind an explicit opt-in | No pretrained weights, no multimodal model, no contamination check, no use case |
| `accelerated_training` | `none` | no | A job specification that requires a stated `gpu_reason`, and a Ray backend test | No workload that justifies a GPU or a cluster, no equivalence test across devices |
| `streaming` | `none` | no | A reader that streams a table in chunks within a memory limit, and a telemetry spool that delivers at least once | No event-streaming pipeline, no stated delivery semantics for one, no recovery proof |

The "experimental" modality lifecycles are baselines that prove the lifecycle
holds together on the author's machine. Their own reports say so and carry
the limits quoted above; the declaration they produce has the status
`experimental`, never `supported`, and only the contract module can name a
supported modality (a guard test checks this).

## Data kinds on the shared stages

Separately from those lifecycles, every data kind is registered in
`scmle.pipeline.kinds.KINDS` and runs through the same stages as a table:
ingestion reads through `kind.read`, profiling surveys through `kind.survey`,
the split takes `kind.constrain`, time-budget admission prices `kind.cost`,
training featurises through `kind.featurise` and packaging decodes through
`kind.decode`. `scmle run DATA --target Y --modality KIND` picks the kind;
the kinds are `tabular`, `document`, `graph`, `image`, `video`, `audio`,
`raster` and `geospatial`.

One shared splitter (`rows.Cohorts`) keeps copies, near-copies, declared
groups and spatial blocks on one side of every split and fold. With a
time-ordered protocol it purges development rows that share a cohort with a
test row; an exchangeable protocol becomes the group split, with a note; a
declared group with missing values is refused; when no cohort holds two rows
the declared protocol applies.

Two split strategies serve panels and places: `grouped_temporal`
(`temporal.Succession`) keeps each group whole and on one side of the test
split and every fold, ordered by each group's first timestamp, and
`spatial_block` (`spatial.Tiling`) keeps square blocks of space whole, with
the block side derived from the label's semivariogram.

Each kind declares its own capability records (`Kind.declare()`), and
`roster.enrol` enrols them. Every kind except `tabular` is declared
`experimental`, not `supported`: supported means the whole lifecycle is
proven, and deployment, monitoring and recovery are not proven for these
kinds. A run of any of them needs the opt-in `models.experimental_models` in
the `--config` file, naming the model families the task accepts; without it
the run is refused with an `UnsupportedError`. `geospatial` (vector) has two
more reasons: a served model takes the geometry columns, not raw GeoJSON, and
the held-out test interval resamples rows, not spatial blocks, so it can be
too narrow.

What every non-tabular kind does not do: a served model takes described
rows (descriptor or feature rows), not raw files; the descriptors are
hand-built, not a learned or pretrained representation. Admission uses each
kind's own cost, so a media kind needs a larger time budget than a table of
the same row count. Optional libraries are needed for some inputs: Pillow
(images, the `multimodal` extra), `ffmpeg` and `ffprobe` on the PATH (video),
`soundfile` (FLAC audio), `rasterio` (GeoTIFF), `pypdf` (PDF, the `documents`
extra) and `geopandas` (vector formats other than GeoJSON). The `Built`
column keeps the definition files' word, which predates these kinds.

## Streaming and delivery semantics

Exactly-once streaming is not claimed. SCMLE has no event-streaming pipeline.
The word "streaming" in the code means reading a table that is larger than
memory in chunks. The one delivery path that exists, the telemetry spool,
delivers at least once, so a destination can see duplicates. A streaming
slice would first need the owner to state the semantics the consumer needs,
and its evidence would be a kill at every stage of the whole pipeline with the
effects compared with a batch computation; per-stage results do not add up to
an end-to-end one.

## Benchmark definitions

Each file in `benchmarks/breadth/` defines four benchmarks (robustness,
drift, explanation, monitoring) with the damage or signal, the ground truth,
the metric and the pass rule, plus the deployment requirement to be declared,
the external referent and the lifecycle stages required. Parameters are null
until the owner fixes them before building, not by fitting them to a result.
`tests/unit/test_breadth.py` checks that each file is well formed, that
nothing in it is run, fixed or declared, and that the cited evidence exists.

## Registry guard

`tests/unit/test_breadth.py` fails if the production registry gains a
supported capability outside tabular library inference, a supported
pretrained, multimodal or JEPA family, a supported streaming path, a
clustering, association, dimensionality-reduction, forecasting-specific or
deep modality model name in the model registry, a clustering or association
task type, or a supported modality declaration outside the contract module.
An adapter a caller registers is `extension-only`.
