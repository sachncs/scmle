"""Related tables: where they live, how they relate, what they say.

`stores` reads tables from a SQLite file, a Parquet folder or a CSV set;
`schema` declares their keys, times and links (`many_to_one`,
`one_to_many`) and holds the data to them; `engine` joins and aggregates
them as of each target row's cutoff, never reading a row from its future.

    store = stores.locate("warehouse/")            # any of the three kinds
    declared = schema.parse(json.loads(text))
    engine = Engine.open(declared, store)
    frame = engine.features(targets, "moment", grid("orders", ["amount"]))
"""

from scmle.data.relational.engine import Aggregate
from scmle.data.relational.engine import Engine
from scmle.data.relational.engine import grid
from scmle.data.relational.engine import PRIMITIVES
from scmle.data.relational.engine import WINDOWS
from scmle.data.relational.schema import Link
from scmle.data.relational.schema import parse
from scmle.data.relational.schema import Schema
from scmle.data.relational.schema import Table
from scmle.data.relational.stores import Database
from scmle.data.relational.stores import KINDS
from scmle.data.relational.stores import locate
from scmle.data.relational.stores import Store

__all__ = [
    "KINDS",
    "PRIMITIVES",
    "WINDOWS",
    "Aggregate",
    "Database",
    "Engine",
    "Link",
    "Schema",
    "Store",
    "Table",
    "grid",
    "locate",
    "parse",
]
