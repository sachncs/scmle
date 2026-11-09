"""Local explanations: what moved one prediction away from a baseline."""

from collections.abc import Callable
from collections.abc import Sequence

import numpy
import pandas

MAX_BACKGROUND = 50


class OcclusionExplainer:
    """Attributes a prediction to features by occluding them.

    A feature is occluded by replacing it with background values.

    The contribution of a feature is the prediction minus the average
    prediction obtained when that feature alone is replaced by the values of
    each background row. Contributions are not additive across features;
    they answer "how much does this feature's actual value matter here?".
    """

    def __init__(
        self,
        predict: Callable[[pandas.DataFrame], numpy.ndarray],
        features: Sequence[str],
    ) -> None:
        """Binds the explainer to a scoring function.

        Args:
            predict: Maps a feature frame to one score per row (for example
                a positive-class probability, or a regression value).
            features: Columns that may be replaced.
        """
        self.predict = predict
        self.features = list(features)

    def explain(
        self, rows: pandas.DataFrame, background: pandas.DataFrame
    ) -> pandas.DataFrame:
        """Explains each row.

        Args:
            rows: Rows to explain.
            background: Rows describing "typical" input; at most
                `MAX_BACKGROUND` are used.

        Returns:
            One row per input row: a `prediction` column and one
            contribution column per feature.
        """
        reference = background.head(MAX_BACKGROUND).reset_index(drop=True)
        count, width = len(rows), len(reference)
        scores = numpy.asarray(self.predict(rows), dtype=float)
        table = {"prediction": scores}
        repeated = rows.loc[rows.index.repeat(width)].reset_index(drop=True)
        for feature in self.features:
            replaced = repeated.copy()
            replaced[feature] = numpy.tile(reference[feature].to_numpy(), count)
            baseline = (
                numpy.asarray(self.predict(replaced), dtype=float)
                .reshape(count, width)
                .mean(axis=1)
            )
            table[feature] = scores - baseline
        return pandas.DataFrame(table, index=rows.index)
