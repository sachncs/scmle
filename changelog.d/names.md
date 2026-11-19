### Changed

- Phase 6 rename maps (`scripts/renames/*.toml`, not applied yet): resolved
  the `[review]` tables. 998 of 1,653 names now have one word in the mapped
  table, and every lane passes `scripts/rename.py --dry-run` with no
  refusals (data_assurance 339, models 180, pipeline 143, rest 336).
  Stored-record attributes take the written key that `docs/RECORD_KEYS.md`
  already records. 656 names stay in `[review]`, each with its reason:
  stored keys with no recorded key rename (renaming them changes the written
  layout, so they need a migration first; the chosen word is the first
  candidate), keys flattened from a nested written key, symbols kept on
  purpose (`p_value`, `schema_version`, `key_id`, e-values), all-caps
  constants, parallel families that need a module split, CLI handlers found
  by `getattr`, library protocol names, and definitions the refactor
  removed.
