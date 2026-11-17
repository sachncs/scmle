"""The derived-feature graph: nodes, their order, and forecast features."""

import hashlib
import json
from typing import Any
from typing import Self

import pydantic

from scmle.contracts import errors
from scmle.contracts import records

GRAPH_VERSION = 1
DELAYED_SUFFIX = "__available"
HORIZON_SEPARATOR = "__h"
DEFAULT_LAGS = (1, 2, 3, 7)
DEFAULT_WINDOWS = (3, 7)


def horizons(value: str, horizon: int) -> list[str]:
    """Returns the future-value target names `<value>__h1 .. <value>__h<H>`.

    Args:
        value: The series value column.
        horizon: The number of future steps.

    Returns:
        One name for each step from 1 to `horizon`, in step order.
    """
    return [f"{value}{HORIZON_SEPARATOR}{k}" for k in range(1, horizon + 1)]


def derive(name: str, op: str, source: str, **params: Any) -> dict[str, Any]:
    """Returns a feature-node record over one source column.

    Args:
        name: The node's name.
        op: The operation.
        source: The source column.
        **params: The operation's parameters.

    Returns:
        The node record, with the source as its only input.
    """
    return {"name": name, "op": op, "inputs": [source], "params": params}


def featurize(record: dict[str, Any], value: str) -> list[dict[str, Any]]:
    """Returns lag, rolling and calendar nodes for a forecasting data record.

    Args:
        record: The data record (a dict of `Dataset` fields), giving the
            lags, windows, seasonal period, time column and calendar parts;
            an empty lags or windows list means the defaults.
        value: The series value column.

    Returns:
        Lag nodes (plus one for the seasonal period if it is not already a
        lag), a rolling mean and a rolling standard deviation node for each
        window, and one date-part node per calendar part when a time column
        is set.
    """
    period = record.get("seasonal_period")
    lags: list[int] = list(record.get("forecast_lags") or DEFAULT_LAGS)
    if period and period not in lags:
        lags.append(int(period))
    windows = list(record.get("forecast_windows") or DEFAULT_WINDOWS)
    nodes = [derive(f"{value}__lag{k}", "lag", value, k=k) for k in lags]
    for w in windows:
        nodes.append(
            derive(f"{value}__mean{w}", "rolling_mean", value, window=w)
        )
        nodes.append(derive(f"{value}__std{w}", "rolling_std", value, window=w))
    clock = record.get("time_column")
    if clock:
        nodes += [
            derive(f"{clock}__{part}", "datepart", clock, part=part)
            for part in record.get("forecast_calendar") or []
        ]
    return nodes


class Node(records.Schema):
    """One derived feature: `name = op(inputs; params)`.

    Attributes:
        name: Name of the new column; must be unique and not clash with data.
        op: Registered operation name (for example `log1p`, `ratio`, `bin`,
            `datepart`, `lag`, `rolling_mean`).
        inputs: Source columns or earlier nodes.
        params: Operation parameters.
    """

    name: str
    op: str
    inputs: list[str]
    params: dict[str, Any] = pydantic.Field(default_factory=dict)


class Derivation(records.Schema):
    """An ordered set of derived features.

    Nodes are listed in dependency order: a node may only use raw columns and
    nodes that appear before it, which makes the graph acyclic by
    construction. The spec is part of the project configuration, so changing
    it changes the configuration hash.

    Attributes:
        version: Graph schema version.
        nodes: The derived features, in dependency order.
    """

    version: int = GRAPH_VERSION
    nodes: list[Node] = pydantic.Field(default_factory=list)

    @pydantic.model_validator(mode="after")
    def verify(self) -> Self:
        """Validates unique names and dependency order.

        Returns:
            The specification, unchanged.

        Raises:
            ValueError: On duplicate names, a node without inputs, or a node
                using a later node or itself.
        """
        seen: set[str] = set()
        names = [node.name for node in self.nodes]
        if len(names) != len(set(names)):
            raise errors.InputError("feature node names must be unique")
        for node in self.nodes:
            if not node.inputs:
                raise errors.InputError(f"node '{node.name}' has no inputs")
            later = set(names) - seen - {node.name}
            used = later & set(node.inputs)
            if used or node.name in node.inputs:
                culprits = sorted(used | ({node.name} & set(node.inputs)))
                raise errors.InputError(
                    f"node '{node.name}' depends on a node defined later or "
                    f"on itself: {culprits}"
                )
            seen.add(node.name)
        return self

    def fingerprint(self) -> str:
        """Returns a content hash identifying this exact graph.

        Returns:
            The first 16 hex characters of a SHA-256 over the version and the
            nodes.
        """
        payload = json.dumps(
            [node.model_dump(mode="json") for node in self.nodes],
            sort_keys=True,
        )
        return hashlib.sha256(
            f"v{self.version}:{payload}".encode()
        ).hexdigest()[:16]
