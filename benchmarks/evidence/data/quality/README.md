# Data for `benchmarks/evidence/quality.py` (claim C4)

Fetched from OpenML by task id with `quality.fetch` (TabArena-v0.1 curated
tasks, development set of `benchmarks/sota/split.json`). Per task:

* `<task>.pq`: the dataset's Parquet file exactly as OpenML publishes it.
* `<task>.json`: the source URL, dataset id and version, the licence OpenML
  states, the target, ignored columns, the Parquet file's SHA-256 and the
  test row ids of repeat 0 fold 0, extracted from the task's official split
  file (`data_splits_url`).

| Task | Dataset (version) | Name | Licence (as stated on OpenML) |
| --- | --- | --- | --- |
| 363621 | 46913 (2) | blood-transfusion-service-center | CC BY 4.0 |
| 363696 | 46952 (2) | qsar-biodeg | CC BY 4.0 |
| 363682 | 46938 (2) | Is-this-a-good-customer | CC0: Public Domain |
| 363684 | 46940 (1) | Marketing_Campaign | Public |
| 363674 | 46930 (1) | hazelnut-spread-contaminant-detection | CC BY-SA |
| 363685 | 46941 (3) | maternal_health_risk | CC BY 4.0 |
| 363711 | 46980 (3) | MIC | CC BY 4.0 |
| 363702 | 46958 (4) | splice | CC BY 4.0 |
| 363704 | 46960 (1) | students_dropout_and_academic_success | CC BY 4.0 |
| 363699 | 46955 (1) | SDSS17 | Public Domain |
| 363698 | 46954 (8) | QSAR_fish_toxicity | CC BY 4.0 |
| 363625 | 46917 (8) | concrete_compressive_strength | CC BY 4.0 |
| 363612 | 46904 (9) | airfoil_self_noise | CC BY 4.0 |
| 363708 | 46964 (7) | wine_quality | CC BY 4.0 |
| 363686 | 46942 (3) | miami_housing | CC BY-NC-SA (non-commercial) |

Source pages: `https://www.openml.org/t/<task>` and
`https://www.openml.org/d/<dataset>`. The files are unchanged copies kept
for reproducible benchmarking with attribution; the share-alike and
non-commercial terms of the CC BY-SA and CC BY-NC-SA files apply to them.
