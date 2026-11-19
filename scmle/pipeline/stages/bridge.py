"""Products whose single-word fields fill context attributes of other names.

`products.Product.capture` and `apply` read and set a product's fields by
name on the run context. A product named under the single-word rule
(`notes`, `labels`) still has to fill the context attribute the stages not
yet converted read (`cleaning_notes`, `ordinal_labels`); `Bridge.names`
maps one to the other. The bridge goes with the context (phase 4, 1.E).
"""

from collections.abc import Mapping
import dataclasses
from typing import ClassVar
from typing import Self

from scmle.pipeline import products


@dataclasses.dataclass(frozen=True)
class Bridge(products.Derived):
    """A derived product whose fields reach renamed context attributes.

    Attributes:
        names: The context attribute of each renamed field, by field; a
            field it does not list keeps its own name.
    """

    names: ClassVar[Mapping[str, str]] = {}

    @classmethod
    def capture(cls, board: object) -> Self:
        """Reads the product from the run context after its stage ran.

        Args:
            board: The run context; it has an attribute per field, under
                the name `names` gives.

        Returns:
            The product.
        """
        return cls(
            **{
                field.name: getattr(
                    board, cls.names.get(field.name, field.name)
                )
                for field in dataclasses.fields(cls)
            }
        )

    def apply(self, board: object) -> None:
        """Sets the product's fields on the run context, renamed by `names`.

        Args:
            board: The run context.
        """
        for field in dataclasses.fields(self):
            setattr(
                board,
                self.names.get(field.name, field.name),
                getattr(self, field.name),
            )
