"""Custody of the evidence (plan section 6).

Every evidence set is split three ways: `development` (building and
debugging), `validation` (deciding scorecard rows during stages B and C) and
`acceptance` (deciding the rows marked "acceptance", once, in stage D). The
acceptance split is sealed: the SHA-256 of its contents is recorded here and
`problems` reports any set whose items or acceptance contents changed since.

Sets, in the order they are assigned:

* `leakage`: the clean and leaking cases of `leakage.catalog` (claim C2).
* `refusal`: the real and simulated tasks of `refusal.catalog` (C14); the
  planted tasks are not in custody here. The simulated group holds
  `refusal.worlds()` = 2 x 59 worlds.
* `quality`: the OpenML tasks of `quality.TASKS` (C4), each keyed by its
  table (`quality.TABLES`, or the id another set uses for the same table,
  `quality.SAME`), so a table shared with an earlier set keeps its split.
  Each problem type holds enough tasks for `minimum` in both deciding
  splits (`quality`'s selection rule).
* `reading`: the awkward files of `reading.corpus` (C5); every group holds
  at least `2 GROUPED` files, generated ones where too few were written.
* `coverage`: the simulation seeds of `coverage.run`, `(family, trial)` for
  `trial < coverage.TRIALS` (C3).
* `practitioner`: the practitioner-study tasks (C4, C13); empty until the
  study tasks exist (plan section 10).

Assignment. An item belongs to a `stratum` (a corpus, task group, problem
type, file group or simulation family: the unit its row is decided on) and
to a `source` (the data it is built from). A source is never split: every
item built from the same table goes to one split, in every set, so a table
seen in development for one claim is not sealed for another. Sources are
ordered by `SHA-256(SEED:source)`; sources already placed by an earlier set
keep their split; the rest are taken in that order, each into the first
split, of the one that decides the set's row and then the other of
validation and acceptance, that still needs items of one of its strata, or
into development when neither does.

Shares, derived from acceptance.toml. Validation and acceptance must each be
able to decide the row on their own, so each needs at least the smallest
sample the row's rule can pass on at all, `need`: `ceil(ln alpha /
ln(1 - p0))` for a ceiling `p0`, `ceil(ln alpha / ln r0)` for a floor `r0`
(`benchmarks.acceptance`). With `size` items in a stratum, the stated share
is `need / size` to validation, `need / size` to acceptance and the rest,
`1 - 2 need / size`, to development, which needs to decide nothing. A
stratum with fewer than `2 need` items is `short`: the deciding split is
filled first and the other takes what is left. Needs per set:

* `leakage`: clean cases `false_block_max` (ceiling, 59); leak cases
  `leak_recall_min` (floor, 29). Decided on acceptance.
* `refusal`: real and simulated tasks each `refusal_rate_max` (ceiling, 59).
  Decided on acceptance.
* `quality`: per problem type `quality.minimum()` = `ceil(ln alpha /
  ln 0.5)` (5), the fewest tasks whose largest gap bounds the median.
  Decided on acceptance.
* `reading`: the row is 100 percent correct or refused, which any count can
  decide, so one file of every group in each split. Decided on validation.
* `coverage`: per family `interval_coverage_min` (floor, 42). Decided on
  validation. Seeds are unlimited, so the measurement may draw new seeds
  inside its own splits to reach the planned 882.

    python -m benchmarks.evidence.custody --write
    python -m benchmarks.evidence.custody --check
"""

import argparse
from collections.abc import Callable
import dataclasses
import functools
import hashlib
import inspect
import json
import logging
from pathlib import Path
import sys
from typing import Any
import warnings

import numpy
import pandas

from benchmarks import acceptance
from benchmarks.evidence import coverage
from benchmarks.evidence import leakage
from benchmarks.evidence import quality
from benchmarks.evidence import reading
from benchmarks.evidence import refusal

PATH = Path(__file__).with_suffix(".json")
# The recorded seed of the assignment; changing it reassigns every set.
SEED = 20261117
DEVELOPMENT = "development"
VALIDATION = "validation"
ACCEPTANCE = "acceptance"
SPLITS = (DEVELOPMENT, VALIDATION, ACCEPTANCE)
# Files of every reading group in each split: the row is all-or-nothing.
GROUPED = 1


@dataclasses.dataclass(frozen=True)
class Item:
    """One piece of evidence.

    Attributes:
        name: Unique within its set.
        stratum: The unit its row is decided on.
        source: The data it is built from; never split across splits.
        content: Returns what the item is made of, for the seal.
    """

    name: str
    stratum: str
    source: str
    content: Callable[[], Any] = dataclasses.field(compare=False)


@dataclasses.dataclass(frozen=True)
class Set:
    """One evidence set and what its splits must be able to decide.

    Attributes:
        name: The set's name.
        claim: The scorecard claim it decides.
        decides: The split that decides the claim's row.
        needs: Items each of validation and acceptance needs per stratum.
        derivation: Why `needs` is what it is.
        items: Every item.
    """

    name: str
    claim: str
    decides: str
    needs: dict[str, int]
    derivation: str
    items: tuple[Item, ...]


def minimum(variable: str) -> int:
    """Smallest sample a statistical row can pass on at all.

    Args:
        variable: A statistical variable of acceptance.toml.

    Returns:
        Its rule's minimum at `alpha`.

    Raises:
        ValueError: If the variable has no rule.
    """
    variables = acceptance.load()
    rule = variables[variable].rule
    if rule is None:
        raise ValueError(f"{variable} has no statistical rule")
    return rule.minimum(variables["alpha"].value)


def origin(source: int | str) -> str:
    """The source key of a real table.

    Args:
        source: An OpenML data id or a scikit-learn `load_*` name.

    Returns:
        `openml:<id>` or `sklearn:<name>`.
    """
    if isinstance(source, str):
        return f"sklearn:{source}"
    return f"openml:{source}"


def screened(case: leakage.Case) -> tuple[Any, ...]:
    """What a leakage case is made of.

    Args:
        case: The case.

    Returns:
        Its definition and the table it screens.
    """
    return dataclasses.asdict(case), leakage.build(case)


def leaks() -> Set:
    """The leakage set: clean and leaking cases of real tables.

    Returns:
        The set.
    """
    items = tuple(
        Item(
            name,
            case.corpus,
            origin(case.table.source),
            functools.partial(screened, case),
        )
        for name, case in leakage.catalog().items()
    )
    return Set(
        "leakage",
        "C2",
        ACCEPTANCE,
        {
            "clean": minimum(leakage.BLOCKS),
            "leak": minimum(leakage.RECALL),
        },
        "clean: false_block_max ceiling minimum; leak: leak_recall_min "
        "floor minimum",
        items,
    )


def identify(task: refusal.Task) -> str:
    """The source key of a refusal task.

    Args:
        task: A real or simulated task.

    Returns:
        The real table's key, or `refusal:<name>` for a simulated task.
    """
    if task.group != "real":
        return f"refusal:{task.name}"
    found = inspect.getclosurevars(task.build).nonlocals
    return origin(found["identifier"])


def posed(task: refusal.Task) -> tuple[Any, ...]:
    """What a refusal task is made of.

    Args:
        task: The task.

    Returns:
        Its kind and the table, target and arguments it runs with.
    """
    return task.kind, task.build()


def refusals() -> Set:
    """The refusal set: real and simulated in-envelope tasks.

    Returns:
        The set.
    """
    items = tuple(
        Item(
            name,
            task.group,
            identify(task),
            functools.partial(posed, task),
        )
        for name, task in refusal.catalog().items()
        if task.group in ("real", "simulated")
    )
    need = minimum("refusal_rate_max")
    return Set(
        "refusal",
        "C14",
        ACCEPTANCE,
        {"real": need, "simulated": need},
        "refusal_rate_max ceiling minimum per group",
        items,
    )


def files(task: int) -> tuple[bytes, bytes]:
    """The cached files of one quality task.

    Args:
        task: OpenML task id.

    Returns:
        The Parquet file's bytes and the description's bytes.
    """
    return (
        (quality.DATA / f"{task}.pq").read_bytes(),
        (quality.DATA / f"{task}.json").read_bytes(),
    )


def tasks() -> Set:
    """The quality set: OpenML tasks per problem type.

    Returns:
        The set.
    """
    items = tuple(
        Item(
            str(task),
            kind,
            origin(quality.SAME.get(table, table)),
            functools.partial(files, task),
        )
        for kind, ids in quality.TASKS.items()
        for task in ids
        for table in (quality.TABLES[task],)
    )
    return Set(
        "quality",
        "C4",
        ACCEPTANCE,
        dict.fromkeys(quality.TASKS, quality.minimum()),
        "quality.minimum(): ceil(ln alpha / ln 0.5) tasks per problem type",
        items,
    )


def written(case: reading.Case) -> tuple[Any, ...]:
    """What an awkward file is made of.

    Args:
        case: The case.

    Returns:
        Its suffix, bytes, expected columns and expected cells.
    """
    return case.suffix, case.payload, case.columns, repr(case.rows)


def awkward() -> Set:
    """The reading set: awkward files and what reading them must give.

    Returns:
        The set.
    """
    corpus = reading.corpus()
    items = tuple(
        Item(
            case.name,
            case.group,
            f"reading:{case.name}",
            functools.partial(written, case),
        )
        for case in corpus
    )
    return Set(
        "reading",
        "C5",
        VALIDATION,
        dict.fromkeys(sorted({c.group for c in corpus}), GROUPED),
        "100 percent correct or refused: any count decides, so one file of "
        "every group in each split",
        items,
    )


def world(family: str, trial: int) -> tuple[Any, ...]:
    """The rows one coverage trial draws, in the order `coverage.run` does.

    Args:
        family: One of `coverage.FAMILIES`.
        trial: The trial number.

    Returns:
        The training, test and fresh draws.
    """
    rng = numpy.random.default_rng([trial, coverage.FAMILIES.index(family)])
    rows = (
        coverage.GROUPS * coverage.GROUP_ROWS
        if family == "binary_groups"
        else coverage.TEST_ROWS
    )
    return (
        coverage.simulate(family, rng, coverage.TRAIN_ROWS),
        coverage.simulate(family, rng, rows),
        coverage.simulate(family, rng, coverage.FRESH_ROWS),
    )


def seeds() -> Set:
    """The coverage set: simulation seeds per family.

    Returns:
        The set.
    """
    items = tuple(
        Item(
            f"{family}/{trial}",
            family,
            f"coverage:{family}:{trial}",
            functools.partial(world, family, trial),
        )
        for family in coverage.FAMILIES
        for trial in range(coverage.TRIALS)
    )
    return Set(
        "coverage",
        "C3",
        VALIDATION,
        dict.fromkeys(coverage.FAMILIES, minimum(coverage.VARIABLE)),
        "interval_coverage_min floor minimum per family",
        items,
    )


def study() -> Set:
    """The practitioner-study slot; empty until the tasks exist.

    Returns:
        The set, without items.
    """
    return Set(
        "practitioner",
        "C4, C13",
        ACCEPTANCE,
        {},
        "no tasks yet; plan section 10 fixes their number",
        (),
    )


def catalog() -> list[Set]:
    """Every evidence set, in the order sources are assigned.

    Returns:
        The sets; the leakage set first, as it needs the most real tables.
    """
    return [leaks(), refusals(), tasks(), awkward(), seeds(), study()]


def rank(source: str, seed: int) -> str:
    """The order key of a source.

    Args:
        source: The source key.
        seed: The recorded seed.

    Returns:
        `SHA-256(seed:source)` in hexadecimal.
    """
    return hashlib.sha256(f"{seed}:{source}".encode()).hexdigest()


def order(split: str) -> tuple[str, str]:
    """Validation and acceptance, the deciding split first.

    Args:
        split: The deciding split.

    Returns:
        The two deciding-capable splits in filling order.
    """
    if split == VALIDATION:
        return VALIDATION, ACCEPTANCE
    return ACCEPTANCE, VALIDATION


def assign(sets: list[Set], seed: int = SEED) -> dict[str, dict[str, str]]:
    """Places every item of every set in a split.

    Args:
        sets: The sets, in assignment order.
        seed: The recorded seed.

    Returns:
        Set name to item name to split.
    """
    fixed: dict[str, str] = {}
    result = {}
    for each in sets:
        groups: dict[str, list[Item]] = {}
        for item in each.items:
            groups.setdefault(item.source, []).append(item)
        counts = {s: dict.fromkeys(each.needs, 0) for s in SPLITS}
        placed = {}
        sources = sorted(groups, key=lambda s: rank(s, seed))
        sources.sort(key=lambda s: s not in fixed)
        for source in sources:
            split = fixed.get(source) or pick(groups[source], counts, each)
            for item in groups[source]:
                placed[item.name] = split
                strata = counts[split]
                strata[item.stratum] = strata.get(item.stratum, 0) + 1
            fixed[source] = split
        result[each.name] = placed
    return result


def pick(
    group: list[Item], counts: dict[str, dict[str, int]], each: Set
) -> str:
    """The split a new source goes to.

    Args:
        group: The source's items.
        counts: Items placed so far, per split and stratum.
        each: The set.

    Returns:
        The first of the deciding splits still short of a stratum the
        source has items in; development otherwise.
    """
    strata = {item.stratum for item in group}
    for split in order(each.decides):
        if any(counts[split].get(s, 0) < each.needs.get(s, 0) for s in strata):
            return split
    return DEVELOPMENT


def shares(each: Set, placed: dict[str, str]) -> dict[str, dict[str, Any]]:
    """The stated share and the count of every split, per stratum.

    Args:
        each: The set.
        placed: Item name to split.

    Returns:
        Stratum to its size, need, stated shares, counts and whether it is
        short of `2 need` items.
    """
    found = {}
    strata = sorted({i.stratum for i in each.items} | set(each.needs))
    for stratum in strata:
        members = [i for i in each.items if i.stratum == stratum]
        size = len(members)
        need = each.needs.get(stratum, 0)
        part = need / size if size else 0.0
        found[stratum] = {
            "size": size,
            "need": need,
            "share": {
                DEVELOPMENT: max(0.0, 1 - 2 * part),
                VALIDATION: min(part, 1.0),
                ACCEPTANCE: min(part, 1.0),
            },
            "counts": {
                s: sum(placed[i.name] == s for i in members) for s in SPLITS
            },
            "short": size < 2 * need,
        }
    return found


def fingerprint(value: Any, hasher: Any) -> None:
    """Feeds a value's contents to a hash, recursively.

    Args:
        value: Bytes, text, numbers, arrays, frames, series, mappings,
            sequences or dataclasses; anything else by its `repr`.
        hasher: A `hashlib` object.
    """
    if isinstance(value, pandas.DataFrame | pandas.Series):
        tabulate(value, hasher)
    elif isinstance(value, bytes):
        hasher.update(b"b%d:" % len(value) + value)
    elif isinstance(value, numpy.ndarray):
        hasher.update(f"array:{value.dtype}:{value.shape}".encode())
        hasher.update(numpy.ascontiguousarray(value).tobytes())
    elif isinstance(value, dict):
        hasher.update(b"dict%d:" % len(value))
        for key in sorted(value, key=repr):
            fingerprint(repr(key), hasher)
            fingerprint(value[key], hasher)
    elif isinstance(value, list | tuple):
        hasher.update(b"list%d:" % len(value))
        for part in value:
            fingerprint(part, hasher)
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        fingerprint(dataclasses.asdict(value), hasher)
    else:
        hasher.update(f"r:{value!r};".encode())


def tabulate(value: pandas.DataFrame | pandas.Series, hasher: Any) -> None:
    """Feeds a frame's or series' names, types and cells to a hash.

    Args:
        value: The frame or series.
        hasher: A `hashlib` object.
    """
    if isinstance(value, pandas.DataFrame):
        hasher.update(b"frame:")
        fingerprint([str(c) for c in value.columns], hasher)
        for column in range(value.shape[1]):
            tabulate(value.iloc[:, column], hasher)
        return
    hasher.update(b"series:")
    fingerprint([str(value.name), str(value.dtype)], hasher)
    try:
        cells = pandas.util.hash_pandas_object(value, index=True)
        hasher.update(cells.to_numpy().tobytes())
    except TypeError:
        hasher.update(value.to_csv().encode())


def contents(item: Item) -> str:
    """SHA-256 of what one item is made of.

    Args:
        item: The item.

    Returns:
        The hexadecimal digest.
    """
    hasher = hashlib.sha256()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logging.disable(logging.WARNING)
        try:
            fingerprint(item.content(), hasher)
        finally:
            logging.disable(logging.NOTSET)
    return hasher.hexdigest()


def seal(each: Set, names: list[str]) -> str:
    """SHA-256 of the acceptance split's contents.

    Args:
        each: The set.
        names: The acceptance items' names.

    Returns:
        The digest over every named item's name and contents, by name.
    """
    items = {i.name: i for i in each.items}
    hasher = hashlib.sha256()
    for name in sorted(names):
        hasher.update(f"{name}\0{contents(items[name])}\n".encode())
    return hasher.hexdigest()


def measure(sets: list[Set] | None = None, seed: int = SEED) -> dict[str, Any]:
    """Assigns every set and seals its acceptance split.

    Args:
        sets: The sets; None uses `catalog`.
        seed: The recorded seed.

    Returns:
        The JSON-ready document.
    """
    sets = catalog() if sets is None else sets
    placed = assign(sets, seed)
    described = []
    for each in sets:
        splits = placed[each.name]
        sealed = sorted(n for n, s in splits.items() if s == ACCEPTANCE)
        described.append(
            {
                "name": each.name,
                "claim": each.claim,
                "decides": each.decides,
                "needs": each.needs,
                "derivation": each.derivation,
                "strata": shares(each, splits),
                "items": dict(sorted(splits.items())),
                "acceptance": len(sealed),
                "seal": seal(each, sealed),
            }
        )
    return {"seed": seed, "splits": list(SPLITS), "sets": described}


def read(document: dict[str, Any] | None) -> dict[str, Any] | None:
    """The given document, or the stored one.

    Args:
        document: A document, or None to read `PATH`.

    Returns:
        The document, or None when nothing is stored.
    """
    if document is not None:
        return document
    if not PATH.exists():
        return None
    loaded: dict[str, Any] = json.loads(PATH.read_text(encoding="utf-8"))
    return loaded


def problems(
    document: dict[str, Any] | None = None, sets: list[Set] | None = None
) -> list[str]:
    """Why the custody record no longer holds.

    Args:
        document: The document; read from `PATH` when None.
        sets: The current sets; None uses `catalog`.

    Returns:
        Messages for a missing file, and for a set that is missing, new,
        whose items or their splits changed, or whose acceptance contents
        no longer match the seal (or cannot be read); empty when every
        seal holds.
    """
    found = read(document)
    if found is None:
        return [f"{PATH.name} is missing"]
    sets = catalog() if sets is None else sets
    seed = found.get("seed", SEED)
    placed = assign(sets, seed)
    stored = {s.get("name"): s for s in found.get("sets", [])}
    issues = [
        f"custody: set {name} is not in custody"
        for name in sorted(placed.keys() - stored.keys())
    ]
    issues += [
        f"custody: set {name} no longer exists"
        for name in sorted(stored.keys() - placed.keys())
    ]
    for each in sets:
        if each.name in stored:
            issues += check(each, stored[each.name], placed[each.name])
    return issues


def check(
    each: Set, record: dict[str, Any], placed: dict[str, str]
) -> list[str]:
    """What changed in one set since it was sealed.

    Args:
        each: The current set.
        record: Its stored record.
        placed: Its current assignment.

    Returns:
        Messages for changed items, splits or acceptance contents.
    """
    issues = []
    items = record.get("items", {})
    added = sorted(placed.keys() - items.keys())
    removed = sorted(items.keys() - placed.keys())
    moved = sorted(
        n for n in placed.keys() & items.keys() if placed[n] != items[n]
    )
    for label, names in (
        ("added", added),
        ("removed", removed),
        ("moved", moved),
    ):
        if names:
            issues.append(
                f"custody: {each.name}: {len(names)} items {label} since "
                f"sealing ({', '.join(names[:5])})"
            )
    sealed = [n for n, s in items.items() if s == ACCEPTANCE]
    if any(n not in placed for n in sealed):
        issues.append(
            f"custody: {each.name}: sealed acceptance items no longer exist"
        )
        return issues
    try:
        digest = seal(each, sealed)
    except Exception as error:  # noqa: BLE001 - any failure to read
        issues.append(
            f"custody: {each.name}: acceptance contents cannot be read "
            f"({type(error).__name__}: {error})"
        )
        return issues
    if digest != record.get("seal"):
        issues.append(
            f"custody: {each.name}: acceptance contents changed since sealing"
        )
    return issues


def summary(document: dict[str, Any] | None = None) -> tuple[str, str]:
    """The custody record in one line, and its verdict.

    Args:
        document: The document; read from `PATH` when None.

    Returns:
        The counts per split and the short strata; `pass` when every
        stratum has `2 need` items, `fail` when any is short, `not
        measured` when nothing is stored.
    """
    found = read(document)
    if found is None:
        return "no custody record", "not measured"
    totals = dict.fromkeys(SPLITS, 0)
    short = []
    for each in found["sets"]:
        for name, stratum in each["strata"].items():
            for split in SPLITS:
                totals[split] += stratum["counts"][split]
            if stratum["short"]:
                short.append(f"{each['name']}/{name}")
    text = (
        f"{len(found['sets'])} sets; "
        + ", ".join(f"{s} {totals[s]}" for s in SPLITS)
        + f"; short: {', '.join(short) if short else 'none'}"
    )
    return text, "fail" if short else "pass"


def main(arguments: list[str] | None = None) -> int:
    """Writes (`--write`) or checks (`--check`) the custody record.

    Args:
        arguments: Command-line arguments; None reads `sys.argv`.

    Returns:
        0 after `--write`; for `--check`, 0 when `problems` is empty and 1
        otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    options = parser.parse_args(arguments)
    if options.write:
        document = measure()
        PATH.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary(document)))
        return 0
    issues = problems()
    print(json.dumps({"ok": not issues, "problems": issues}, indent=2))
    return 0 if not issues else 1


if __name__ == "__main__":
    sys.exit(main())
