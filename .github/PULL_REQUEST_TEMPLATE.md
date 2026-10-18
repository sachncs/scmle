## What and why

## Checklist

- [ ] Tests added or updated (a regression test for every fixed bug)
- [ ] `ruff check .`, `ruff format --check .`, `mypy scml benchmarks` and `pytest` pass
- [ ] `tests/leakage` stays green for anything touching splits or evaluation
- [ ] `CHANGELOG.md` updated under "Unreleased"
- [ ] Works on Linux, macOS and Windows (no POSIX-only calls; text files use `encoding="utf-8"`)
