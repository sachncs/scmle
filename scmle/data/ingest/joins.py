"""Join discovery by exact inclusion dependencies.

A candidate link is a child column set `A`, a parent key `B` (an approximate
key from `grain`), compatible value kinds, and containment
`|A ∩ B| / |A|` over distinct non-null values, measured by an anti-join count
in the source. Containment alone is weak evidence: surrogate keys `1..N` of
unrelated tables include each other by chance. So a link is accepted
automatically only when (a) a foreign key is declared, or (b) containment is
exact and the names corroborate it and no other parent competes. Every other
link that clears the containment floor stays `candidate` and is asked, not
joined. Links where neither side is a key are `many_to_many`: reported, never
joined.
"""

from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
from typing import Any

from scmle.data.ingest import grain
from scmle.data.ingest import profile

ACCEPTED = "accepted"
CANDIDATE = "candidate"
MANY_TO_MANY = "many_to_many"
VIOLATED = "declared_violated"
ONE_TO_ONE = "one_to_one"
MANY_TO_ONE = "many_to_one"
MIN_COMPETING_LINKS = 2


@dataclasses.dataclass(frozen=True, slots=True)
class DeclaredKey:
    """A foreign key the schema declares.

    Attributes:
        child: Child table.
        child_columns: Child columns.
        parent: Parent table.
        parent_columns: Parent columns, same length.
    """

    child: str
    child_columns: tuple[str, ...]
    parent: str
    parent_columns: tuple[str, ...]

    def __post_init__(self) -> None:
        """Checks the column lists pair up.

        Raises:
            ValueError: If the child columns are empty or differ in length
                from the parent columns.
        """
        if not self.child_columns or len(self.child_columns) != len(
            self.parent_columns
        ):
            raise ValueError("declared key columns must pair up")


@dataclasses.dataclass(frozen=True, slots=True)
class Link:
    """A measured link and the decision taken on it.

    Attributes:
        child: Child table.
        child_columns: Child columns `A`.
        parent: Parent table.
        parent_columns: Parent columns `B`.
        containment: Share of distinct child values found in the parent.
        child_distinct: Distinct non-null values of `A`.
        cardinality: `many_to_one`, `one_to_one` or `many_to_many`.
        declared: Whether a schema declared it.
        name_support: Whether the names corroborate it.
        score: Ranking aid, never a probability.
        decision: `accepted`, `candidate`, `many_to_many`, `declared_violated`.
        why: One sentence on the decision.
    """

    child: str
    child_columns: tuple[str, ...]
    parent: str
    parent_columns: tuple[str, ...]
    containment: float
    child_distinct: int
    cardinality: str
    declared: bool
    name_support: bool
    score: float
    decision: str
    why: str

    @property
    def joinable(self) -> bool:
        """Whether synthesis may traverse the link (it was accepted)."""
        return self.decision == ACCEPTED

    def to_dict(self) -> dict[str, Any]:
        """Serializes the link.

        Returns:
            A JSON-ready mapping of the fields plus `joinable`.
        """
        return {
            **dataclasses.asdict(self),
            "child_columns": list(self.child_columns),
            "parent_columns": list(self.parent_columns),
            "joinable": self.joinable,
        }


@dataclasses.dataclass(frozen=True, slots=True)
class Step:
    """One hop of a path: along a link, up to the parent or down to children.

    Attributes:
        link: The link the hop follows.
        direction: `up` (child to parent) or `down` (parent to children).
    """

    link: Link
    direction: str

    @property
    def source(self) -> str:
        """The name of the table the hop starts from."""
        if self.direction == "up":
            return self.link.child
        return self.link.parent

    @property
    def target(self) -> str:
        """The name of the table the hop reaches."""
        if self.direction == "up":
            return self.link.parent
        return self.link.child


@dataclasses.dataclass(frozen=True, slots=True)
class JoinGraph:
    """Measured links, as `join_graph.json`.

    Attributes:
        links: Every measured link, best score first.
        partial: True when the budget stopped the search early.
    """

    links: tuple[Link, ...]
    partial: bool = False

    def accepted(self) -> tuple[Link, ...]:
        """Selects the links synthesis may traverse.

        Returns:
            The accepted links, in graph order.
        """
        return tuple(link for link in self.links if link.joinable)

    def questions(self) -> list[str]:
        """Lists the links a person must settle.

        Returns:
            One sentence per `candidate` or `declared_violated` link.
        """
        return [
            f"{item.child}{list(item.child_columns)} -> "
            f"{item.parent}{list(item.parent_columns)}: {item.why}"
            for item in self.links
            if item.decision in (CANDIDATE, VIOLATED)
        ]

    def to_dict(self) -> dict[str, Any]:
        """Serializes the graph, for `join_graph.json`.

        Returns:
            A JSON-ready mapping with `partial`, the links and the
            questions.
        """
        return {
            "partial": self.partial,
            "links": [link.to_dict() for link in self.links],
            "questions": self.questions(),
        }


def name_supports(
    child_columns: Sequence[str],
    parent: str,
    parent_columns: Sequence[str],
) -> bool:
    """Whether column and table names corroborate a link.

    True when the child and parent column names are the same identifier, or
    the child column names the parent table (`customer_id` to `customers`).
    A bare `id` or a shared non-key name (`label`) never corroborates.

    Args:
        child_columns: Child column names.
        parent: Parent table name.
        parent_columns: Parent column names.

    Returns:
        True if the names corroborate the link.
    """
    if len(child_columns) != 1:
        return list(child_columns) == list(parent_columns)
    child_tokens = grain.name_tokens(child_columns[0])
    parent_tokens = grain.name_tokens(parent_columns[0])
    id_like = bool(grain.ID_WORDS & set(child_tokens))
    if (
        child_tokens == parent_tokens
        and id_like
        and set(child_tokens) - grain.ID_WORDS
    ):
        return True
    table_tokens = grain.name_tokens(parent)
    named = bool(table_tokens) and all(t in child_tokens for t in table_tokens)
    return named and id_like


def contained(
    source: profile.SqlSource,
    child: str,
    child_columns: Sequence[str],
    parent: str,
    parent_columns: Sequence[str],
) -> int:
    """Counts distinct child values with no parent match (an anti-join).

    Args:
        source: Read-only source.
        child: Child table.
        child_columns: Child columns; rows with a null in any are ignored.
        parent: Parent table.
        parent_columns: Parent columns, paired with the child columns.

    Returns:
        The number of distinct non-null child combinations that no parent
        row matches; 0 if the budget refused the query.
    """
    present = " AND ".join(
        f"c.{profile.quote(c)} IS NOT NULL" for c in child_columns
    )
    same = " AND ".join(
        f"p.{profile.quote(b)} = c.{profile.quote(a)}"
        for a, b in zip(child_columns, parent_columns, strict=True)
    )
    names = ", ".join(f"c.{profile.quote(c)}" for c in child_columns)
    sql = profile.statement(
        "SELECT COUNT(*) FROM (SELECT DISTINCT {n} FROM {c} c "
        "WHERE {w}) c WHERE NOT EXISTS "
        "(SELECT 1 FROM {p} p WHERE {s})",
        n=names,
        c=profile.quote(child),
        w=present,
        p=profile.quote(parent),
        s=same,
    )
    return int(source.scalar(sql) or 0)


def distinct_of(
    source: profile.SqlSource, table: str, columns: Sequence[str]
) -> int:
    """Counts the distinct non-null combinations of a column set.

    Args:
        source: Read-only source.
        table: Table name.
        columns: The column set.

    Returns:
        The count; 0 if the budget refused the query.
    """
    present = " AND ".join(f"{profile.quote(c)} IS NOT NULL" for c in columns)
    return int(
        source.scalar(
            profile.statement(
                "SELECT COUNT(*) FROM (SELECT DISTINCT {n} FROM {t} WHERE {w})",
                n=profile.quote_all(columns),
                t=profile.quote(table),
                w=present,
            )
        )
        or 0
    )


def kinds_match(
    catalog: profile.Catalog,
    child: str,
    child_columns: Sequence[str],
    parent: str,
    parent_columns: Sequence[str],
) -> bool:
    """Checks that paired columns hold the same kind of value.

    Args:
        catalog: Profiles of the tables.
        child: Child table.
        child_columns: Child columns.
        parent: Parent table.
        parent_columns: Parent columns, paired with the child columns.

    Returns:
        True if every pair has equal kinds and none is `empty` or `other`.
    """
    for a, b in zip(child_columns, parent_columns, strict=True):
        left = catalog.table(child).column(a).kind
        right = catalog.table(parent).column(b).kind
        if left != right or left in ("empty", "other"):
            return False
    return True


def single_column_pairs(
    catalog: profile.Catalog, scans: Mapping[str, grain.KeyScan]
) -> list[tuple[str, tuple[str, ...], str, tuple[str, ...]]]:
    """Pairs every column with every single-column key of another table.

    Args:
        catalog: Profiles of the tables.
        scans: Key scan of each table.

    Returns:
        Tuples of child table, child columns, parent table and parent key
        columns.
    """
    pairs: list[tuple[str, tuple[str, ...], str, tuple[str, ...]]] = []
    for child in catalog.tables:
        for parent in catalog.tables:
            if parent.name == child.name:
                continue
            for key in scans[parent.name].keys:
                if len(key.columns) == 1:
                    pairs.extend(
                        (child.name, (column.name,), parent.name, key.columns)
                        for column in child.columns
                    )
    return pairs


def discover(
    source: profile.SqlSource,
    catalog: profile.Catalog,
    scans: Mapping[str, grain.KeyScan],
    declared: Sequence[DeclaredKey] = (),
    eps_fk: float = 0.0,
    stage: str = "joins",
) -> JoinGraph:
    """Measures candidate links and decides which may be joined.

    Args:
        source: Read-only source.
        catalog: Profiles of the tables.
        scans: Key scan of each table (`grain.infer_keys`).
        declared: Foreign keys the schema declares.
        eps_fk: Largest share of child values allowed to lack a parent for a
            link to remain a candidate. Automatic acceptance needs 0.
        stage: Budget stage name.

    Returns:
        The graph; `partial` when the budget stopped the search at a
        child-table boundary.
    """
    declared_set = {
        (d.child, d.child_columns, d.parent, d.parent_columns): d
        for d in declared
    }
    pairs = single_column_pairs(catalog, scans)
    pairs.extend(declared_set)
    pairs = list(dict.fromkeys(pairs))
    measured: list[Link] = []
    partial = False
    by_child: dict[str, list[tuple[Any, ...]]] = {}
    for pair in pairs:
        by_child.setdefault(pair[0], []).append(pair)
    for child_name in [t.name for t in catalog.tables]:
        work = by_child.get(child_name, [])
        if not work:
            continue
        rows = catalog.table(child_name).rows
        total = sum(rows + catalog.table(p[2]).rows for p in work)
        if source.meter is not None and not source.meter.admit(
            stage, child_name, total, 2 * len(work)
        ):
            partial = True
            break
        for child, a, parent, b in work:
            link = measure_link(
                source,
                catalog,
                scans,
                child,
                a,
                parent,
                b,
                (child, a, parent, b) in declared_set,
                eps_fk,
            )
            if link is not None:
                measured.append(link)
    return JoinGraph(tuple(resolve_competition(measured)), partial)


def measure_link(
    source: profile.SqlSource,
    catalog: profile.Catalog,
    scans: Mapping[str, grain.KeyScan],
    child: str,
    a: tuple[str, ...],
    parent: str,
    b: tuple[str, ...],
    is_declared: bool,
    eps_fk: float,
) -> Link | None:
    """Measures one candidate pair.

    Args:
        source: Read-only source.
        catalog: Profiles of the tables.
        scans: Key scan of each table.
        child: Child table.
        a: Child columns.
        parent: Parent table.
        b: Parent columns.
        is_declared: Whether the schema declares this key.
        eps_fk: Largest share of child values allowed to lack a parent.

    Returns:
        The measured link with its decision, or None when the pair is
        clearly not a link: kinds differ (undeclared), the child has no
        values, the parent columns are not a key, or an undeclared pair's
        containment is below `1 - eps_fk`.
    """
    if not is_declared and not kinds_match(catalog, child, a, parent, b):
        return None
    distinct = distinct_of(source, child, a)
    if distinct == 0:
        return None
    missing = contained(source, child, a, parent, b)
    containment = 1.0 - missing / distinct
    parent_key = b in {k.columns for k in scans[parent].keys}
    child_key = a in {k.columns for k in scans[child].keys}
    support = name_supports(a, parent, b)
    if not parent_key:
        return None
    if not is_declared and containment < 1.0 - eps_fk:
        return None
    cardinality = ONE_TO_ONE if child_key else MANY_TO_ONE
    score = (
        containment + (0.5 if support else 0.0) + (1.0 if is_declared else 0)
    )
    if is_declared:
        ok = containment >= 1.0 - eps_fk
        decision = ACCEPTED if ok else VIOLATED
        why = (
            "declared foreign key"
            if ok
            else f"declared key but containment is {containment:.4f}"
        )
    elif missing == 0 and support:
        decision, why = ACCEPTED, "exact containment and names agree"
    elif missing == 0:
        decision, why = CANDIDATE, "exact containment but no name support"
    else:
        decision = CANDIDATE
        why = f"containment {containment:.4f} below exact"
    return Link(
        child,
        a,
        parent,
        b,
        containment,
        distinct,
        cardinality,
        is_declared,
        support,
        score,
        decision,
        why,
    )


def resolve_competition(links: Sequence[Link]) -> list[Link]:
    """Demotes automatic accepts that compete for one child column.

    If several parents are accepted for the same child columns without a
    declaration, only a single parent whose table name the child column
    mentions keeps the acceptance; otherwise all become candidates.

    Args:
        links: The measured links.

    Returns:
        The links with competing accepts demoted to `candidate`, sorted by
        descending score, then child and parent name.
    """
    groups: dict[tuple[str, tuple[str, ...]], list[Link]] = {}
    for link in links:
        if link.joinable and not link.declared:
            groups.setdefault((link.child, link.child_columns), []).append(link)
    demote: set[Link] = set()
    for group in groups.values():
        if len(group) < MIN_COMPETING_LINKS:
            continue
        named = [
            item
            for item in group
            if set(grain.name_tokens(item.parent))
            <= set(grain.name_tokens(item.child_columns[0]))
        ]
        keep = named[0] if len(named) == 1 else None
        demote.update(item for item in group if item is not keep)
    out = [
        dataclasses.replace(
            link,
            decision=CANDIDATE,
            why="several parents fit; ambiguous, asked not joined",
        )
        if link in demote
        else link
        for link in links
    ]
    return sorted(out, key=lambda item: (-item.score, item.child, item.parent))


def many_to_many(
    source: profile.SqlSource,
    catalog: profile.Catalog,
    scans: Mapping[str, grain.KeyScan],
    eps_fk: float = 0.0,
) -> list[Link]:
    """Finds same-named non-key column pairs that overlap.

    The pairs are reported, never joined.

    Each column pair is charged to the budget (the rows of both tables and two
    queries) under the stage `many_to_many` before it is measured. When the
    budget cannot cover the next pair the search stops at that pair, the
    boundary is on the meter, and the pairs found so far are returned.

    Args:
        source: Read-only source.
        catalog: Profiles of the tables.
        scans: Key scan of each table.
        eps_fk: Largest share of left values allowed to lack a match.

    Returns:
        One `many_to_many` link per overlapping pair, where neither column
        is a key and the kinds match.
    """
    found: list[Link] = []
    for left in catalog.tables:
        for right in catalog.tables:
            if left.name >= right.name:
                continue
            for column in left.columns:
                if column.name not in {c.name for c in right.columns}:
                    continue
                a = (column.name,)
                keyed = a in {
                    k.columns for k in scans[left.name].keys
                } or a in {k.columns for k in scans[right.name].keys}
                if keyed or not kinds_match(
                    catalog, left.name, a, right.name, a
                ):
                    continue
                if source.meter is not None and not source.meter.admit(
                    "many_to_many",
                    f"{left.name}.{column.name}~{right.name}.{column.name}",
                    left.rows + right.rows,
                    2,
                ):
                    return found
                distinct = distinct_of(source, left.name, a)
                if not distinct:
                    continue
                missing = contained(source, left.name, a, right.name, a)
                containment = 1.0 - missing / distinct
                if containment >= 1.0 - eps_fk:
                    found.append(
                        Link(
                            left.name,
                            a,
                            right.name,
                            a,
                            containment,
                            distinct,
                            MANY_TO_MANY,
                            declared=False,
                            name_support=True,
                            score=containment,
                            decision=MANY_TO_MANY,
                            why="neither side is a key; reported, never joined",
                        )
                    )
    return found


def paths(
    graph: JoinGraph, target: str, depth: int = 2
) -> list[tuple[Step, ...]]:
    """Paths from the target table of at most `depth` accepted hops.

    Hops go up (child to parent, many-to-one) or down (parent to children,
    one-to-many, to be aggregated through the as-of view). A path never
    revisits a table. Many-to-many links are not in `accepted()`.

    Args:
        graph: The join graph.
        target: Table the paths start from.
        depth: Largest number of hops.

    Returns:
        Every path of one to `depth` hops, each a tuple of steps; empty if
        `depth` is below 1.
    """
    if depth < 1:
        return []
    found: list[tuple[Step, ...]] = []

    def extend(
        prefix: tuple[Step, ...], here: str, seen: frozenset[str]
    ) -> None:
        if len(prefix) >= depth:
            return
        for link in graph.accepted():
            options = []
            if link.child == here and link.parent not in seen:
                options.append(Step(link, "up"))
            if link.parent == here and link.child not in seen:
                options.append(Step(link, "down"))
            for step in options:
                path = (*prefix, step)
                found.append(path)
                extend(path, step.target, seen | {step.target})

    extend((), target, frozenset({target}))
    return found
