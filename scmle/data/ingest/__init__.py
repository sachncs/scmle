"""Raw-data ingestion and feature construction (feature F9).

Reads relational sources by aggregate queries run in the source, infers the
grain, keys and joins of the tables, resolves duplicate entities, and builds
features only through a cutoff-enforcing view. Every inference is a
statistical test or a deterministic search; none uses a language model.
What cannot be verified is reported as `unverified`.
"""
