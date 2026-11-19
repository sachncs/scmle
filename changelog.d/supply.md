### Security

- **A bundle member swapped after its checksum is checked no longer loads.**
  `Bundle.load` checksummed each pickled member (model, calibrator, novelty
  detector, quantile and CV+ models, shift weights) and then opened the file
  again to unpickle it, and checked the manifest signature over a second
  read of the manifest file. A file replaced between the two reads ran
  unauthenticated code. Each member is now read once, checked against the
  signed manifest and loaded from those bytes (`archive.resolve` returns
  bytes, `archive.unpickle(data, name)`). The signature and the schema
  checksum cover the bytes that were parsed (`archive.Snapshot`,
  `seal.verify_signature(root, key, text)`). The JSON and Parquet members
  are read the same way. `tests/unit/test_bundle_swap.py` covers it.

### C9: a secure, private, trustworthy supply chain

- `scmle.security.deserialisation` finds every call that deserialises
  (pickle, joblib, cloudpickle, dill, marshal, shelve, `pandas.read_pickle`,
  `torch.load` without `weights_only=True`, unsafe `yaml` loads, `numpy.load`
  with `allow_pickle=True`, `keras` with `safe_mode=False`) in a package's
  source. It does not import the package. Each call gets a verdict on where its
  bytes come from: `authenticated`, `forwarded` (a helper whose every caller
  passes authenticated bytes), `roundtrip` (a buffer the same function
  dumped) or `unauthenticated`. `python -m scmle.security.deserialisation
  scmle` exits 1 on an unauthenticated site. `scmle/` has 8 sites, none of
  them unauthenticated.
- `tests/unit/test_deserialisation_paths.py` attacks every site. It writes an
  unsigned tripwire pickle and one signed with the wrong key, and calls the
  public entry point. Each attack must be refused and the tripwire must never
  run. A new site with no attack fails the test. A positive control signs
  the same tripwire with the right key and requires it to run, so an attack
  that is refused before it reaches the deserialiser cannot pass. The round
  trip, which reads no input, has one attack (it must open no file) and no
  control.
- `scmle.security.leaks` scans everything a run wrote for the values of its
  input table. It looks for:
  - string cells outside the declared categorical levels (names,
    identifiers, free text);
  - category levels seen in fewer than 5 rows (k-anonymity);
  - raw-row fragments (two precise numeric cells of one row in one file),
    float cells with 12 significant digits and integer cells with 12 digits
    (account numbers, long identifiers);
  - given secrets, and environment secrets.
  It searches UTF-8, UTF-16, UTF-32, JSON- and HTML-escaped forms and
  unsalted SHA-256, SHA-1 and MD5 digests, and integers as decimal text and
  int64 and float64 bytes. It opens zlib, gzip, bz2 and xz streams, scmle
  envelopes, zip, Parquet and SQLite, and never unpickles. The report names
  values by an HMAC-SHA256 prefix under a random salt drawn per scan, so a
  label cannot be reversed by a dictionary search. `python -m scmle.security.leaks
  RUN --data train.csv --schema RUN/model/schema.json` exits 1 on any leak.
- `release.yml` now:
  - attaches SLSA level 3 provenance (`slsa-github-generator`
    `generator_generic_slsa3` v2.1.0). The generator accepts only a full
    release tag, so it is referenced by tag, with its commit recorded beside
    it and checked;
  - signs every distribution with Sigstore before the GitHub release
    (`gh-action-sigstore-python`, pinned by commit);
  - runs `pip-audit` over every locked dependency before upload, judged by
    `scmle.deploy.vuln_gate` (fail closed);
  - refuses to release unless an active ruleset protects the release tag
    from deletion and update.
- `benchmarks/evidence/supply.py` measures the three clauses:
  deserialisation (static scan plus attacks), artifacts (real runs on a
  PII-planted corpus, scanned; a crashed validation run makes the clause
  `not measured`, never a clean scan), and release (checks on the parsed
  workflow, so a comment cannot satisfy one: pip-audit, the gate, upload,
  Sigstore and the GitHub release found as steps and in order, none
  skippable by `if` or `continue-on-error`; the generator exempt from
  commit pinning only at an exact `vX.Y.Z` tag; and live checks of the
  generator tag, the tag rulesets and pip-audit). A run refused for too
  small a budget is retried once at twice the budget it names.
