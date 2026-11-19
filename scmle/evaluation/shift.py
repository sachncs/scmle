"""Adversarial validation: can a model tell training rows from new rows?

If a classifier separates the development rows from the evaluation rows
better than chance, the two differ in distribution, and a score measured on
one says less about the other. The report also names the features that differ
most so the cause can be inspected.
"""

from typing import Any

import numpy
import pandas
from sklearn import base
from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection

from scmle.evaluation import diagnostics
from scmle.evaluation import shift_weights

FOLDS = 5
MIN_PRESENT_ROWS = 2
BINARY_CLASSES = 2
MAX_ROWS = 4000
MODERATE_AUC = 0.60
STRONG_AUC = 0.75
TOP_FEATURES = 5
# Cost model, fitted by non-negative least squares on the 31 runs described
# at `diagnostics.MARGIN`: `ADVERSARY_SECONDS + ADVERSARY_CELL_SECONDS * m *
# p` for m sampled rows (both sets) and p columns; measured over fitted
# 0.60..1.46.
ADVERSARY_SECONDS = 0.18
ADVERSARY_CELL_SECONDS = 2.93e-6


class Adversary(diagnostics.Diagnostic):
    """Separability of development rows from evaluation rows."""

    name = "adversarial_validation"

    def estimate(self, inputs: diagnostics.DiagnosticInput) -> float:
        """Predicts the seconds of the cross-validated separability.

        Args:
            inputs: Evaluation data, with the reference rows attached.

        Returns:
            The fitted cost on the capped sample, times the margin.
        """
        reference = inputs.reference_features
        rows = min(len(inputs.features), MAX_ROWS) + (
            0 if reference is None else min(len(reference), MAX_ROWS)
        )
        columns = len(inputs.feature_plan.sourced)
        return diagnostics.MARGIN * (
            ADVERSARY_SECONDS + ADVERSARY_CELL_SECONDS * rows * columns
        )

    def fold(
        self,
        model: Any,
        encoded: pandas.DataFrame,
        labels: numpy.ndarray,
        train: numpy.ndarray,
        test: numpy.ndarray,
    ) -> numpy.ndarray:
        """Fits on one fold's training rows and scores its held-out rows.

        Args:
            model: Unfitted classifier, cloned before fitting.
            encoded: Numeric features of both sets.
            labels: 1 for evaluation rows, 0 for reference rows.
            train: Positions to fit on.
            test: Positions to score.

        Returns:
            The probability that each held-out row is an evaluation row.
        """
        fitted: Any = base.clone(model)
        fitted.fit(encoded.iloc[train], labels[train])
        return numpy.asarray(fitted.predict_proba(encoded.iloc[test]))[:, 1]

    def compute(self, inputs: diagnostics.DiagnosticInput) -> dict[str, Any]:
        """Cross-validates a classifier that predicts which set a row is from.

        Args:
            inputs: Evaluation data, with the development rows attached as
                `reference_features`.

        Returns:
            A mapping `{"available": False, "reason": "no reference rows"}`
            when `inputs` has no reference rows. Otherwise a mapping with
            `available` true, the out-of-fold separability `auc` (0.5 means
            indistinguishable), its `verdict`, the `odds` summary, the
            `rank` features and the sampled `rows` counts for the
            reference and evaluation sets. `skipped` when the deadline
            leaves no time for the next fold (folds are checked one by one,
            each predicted to take as long as the longest so far).
        """
        reference = inputs.reference_features
        if reference is None:
            return {"available": False, "reason": "no reference rows"}
        if not inputs.affords(self.estimate(inputs) / FOLDS):
            return diagnostics.skipped(
                "the deadline leaves no time for the first fold"
            )
        columns = list(inputs.feature_plan.sourced)
        rng = numpy.random.default_rng(inputs.seed)
        old = self.sample(reference[columns], rng)
        new = self.sample(inputs.features[columns], rng)
        frame = pandas.concat([old, new], ignore_index=True)
        labels = numpy.r_[numpy.zeros(len(old)), numpy.ones(len(new))]
        encoded = self.encode(frame, inputs)
        model = ensemble.HistGradientBoostingClassifier(
            max_iter=60, random_state=inputs.seed
        )
        splitter = model_selection.StratifiedKFold(
            FOLDS, shuffle=True, random_state=inputs.seed
        )
        scores = numpy.zeros(len(labels))
        longest = self.estimate(inputs) / FOLDS
        for done, (train, test) in enumerate(splitter.split(encoded, labels)):
            if done and not inputs.affords(diagnostics.MARGIN * longest):
                return diagnostics.skipped(
                    f"{done} of {FOLDS} folds ran; the next takes up to "
                    f"{longest:.2f} s and the deadline came first"
                )
            began = diagnostics.clock()
            scores[test] = self.fold(model, encoded, labels, train, test)
            took = diagnostics.clock() - began
            longest = took if done == 0 else max(longest, took)
        auc = float(metrics.roc_auc_score(labels, scores))
        return {
            "available": True,
            "auc": auc,
            "verdict": self.verdict(auc),
            "odds": self.odds(scores, len(old)),
            "most_shifted": self.rank(old, new, inputs),
            "rows": {"reference": len(old), "evaluation": len(new)},
        }

    @staticmethod
    def odds(scores: numpy.ndarray, reference_rows: int) -> dict[str, Any]:
        """Returns the odds that weighted conformal would use.

        The out-of-fold probability `p` that a row is an evaluation row
        gives the odds `p / (1 - p)`, corrected for the sample sizes and
        clipped at `shift_weights.DEFAULT_CLIP`, the ratio estimate of
        `scmle.evaluation.shift_weights.weigh` (which uses a
        logistic discriminator fitted on all rows; this one is the boosted
        classifier of the report, scored out of fold). Descriptive.

        Args:
            scores: Out-of-fold probability that each row is an evaluation
                row; the reference rows come first, then the evaluation rows.
            reference_rows: Number of reference rows at the start of `scores`.

        Returns:
            The clip, the size correction, summaries (mean, 95th percentile,
            maximum, clipped share, effective size) of the odds on the
            reference and on the evaluation rows, whether weighted conformal
            is `possible` or `refused` by the reference effective size, the
            minimum effective size used for that decision, and a note.
        """
        new_rows = len(scores) - reference_rows
        clip = shift_weights.DEFAULT_CLIP
        raw = (
            scores
            / numpy.maximum(1.0 - scores, 1e-12)
            * reference_rows
            / new_rows
        )
        ratio = numpy.minimum(raw, clip)

        def describe(
            values: numpy.ndarray, unclipped: numpy.ndarray
        ) -> dict[str, Any]:
            return {
                "mean": float(values.mean()),
                "p95": float(numpy.quantile(values, 0.95)),
                "max": float(values.max()),
                "clipped_share": float((unclipped > clip).mean()),
                "effective_size": shift_weights.effective_sample_size(values),
            }

        reference = describe(ratio[:reference_rows], raw[:reference_rows])
        verdict = (
            "possible"
            if reference["effective_size"] >= shift_weights.MIN_EFFECTIVE_SIZE
            else "refused"
        )
        return {
            "clip": clip,
            "size_correction": reference_rows / new_rows,
            "reference": reference,
            "evaluation": describe(
                ratio[reference_rows:], raw[reference_rows:]
            ),
            "weighted_conformal": verdict,
            "minimum_effective_size": shift_weights.MIN_EFFECTIVE_SIZE,
            "note": "odds of an out-of-fold discriminator, corrected for the "
            "sample sizes and clipped; descriptive, on the capped sample; "
            "weights for a bundle are estimated by tilt",
        }

    @staticmethod
    def sample(
        frame: pandas.DataFrame, rng: numpy.random.Generator
    ) -> pandas.DataFrame:
        """Caps the rows used, keeping runtime bounded.

        Args:
            frame: Rows to sample from.
            rng: Source of the random choice.

        Returns:
            `frame` with a fresh index when it has at most `MAX_ROWS` rows,
            otherwise `MAX_ROWS` randomly chosen rows in original order.
        """
        if len(frame) <= MAX_ROWS:
            return frame.reset_index(drop=True)
        chosen = rng.choice(len(frame), MAX_ROWS, replace=False)
        return frame.iloc[numpy.sort(chosen)].reset_index(drop=True)

    @staticmethod
    def encode(
        frame: pandas.DataFrame, inputs: diagnostics.DiagnosticInput
    ) -> pandas.DataFrame:
        """Returns a numeric view: numbers as is, categories as codes.

        Args:
            frame: Rows holding the feature columns.
            inputs: The diagnostic inputs naming the numeric and
                categorical columns.

        Returns:
            A copy of `frame` with categorical columns replaced by category
            codes, and numeric and datetime columns converted to numbers.
            Numeric features the pipeline derives itself (calendar parts,
            say) are not in `frame`; their source columns are.
        """
        encoded = frame.copy()
        for column in inputs.feature_plan.categorical:
            encoded[column] = encoded[column].astype("category").cat.codes
        for column in encoded.columns:
            dated = pandas.api.types.is_datetime64_any_dtype(encoded[column])
            if dated or column in inputs.feature_plan.numeric:
                encoded[column] = pandas.to_numeric(encoded[column])
        return encoded

    @staticmethod
    def verdict(auc: float) -> str:
        """Names the degree of shift.

        Args:
            auc: Separability AUC.

        Returns:
            `strong shift` from `STRONG_AUC`, `moderate shift` from
            `MODERATE_AUC`, otherwise `no meaningful shift`.
        """
        if auc >= STRONG_AUC:
            return "strong shift"
        if auc >= MODERATE_AUC:
            return "moderate shift"
        return "no meaningful shift"

    @staticmethod
    def rank(
        old: pandas.DataFrame,
        new: pandas.DataFrame,
        inputs: diagnostics.DiagnosticInput,
    ) -> list[dict[str, Any]]:
        """Ranks features by how separable the two sets are on that feature.

        Numeric features use `2 * |AUC - 0.5|` of the single feature as a
        classifier of old versus new rows; categorical features use the
        total-variation distance between level frequencies. Both lie in
        [0, 1] and are comparable as a ranking. A numeric feature is left out
        when it has fewer than two non-missing values or when its non-missing
        values come from only one of the two sets.

        Args:
            old: Sampled reference rows.
            new: Sampled evaluation rows.
            inputs: The diagnostic inputs naming the numeric and
                categorical columns.

        Returns:
            Up to `TOP_FEATURES` mappings with keys `feature` and
            `distance`, largest distance first.
        """
        found = []
        sourced = inputs.feature_plan.sourced
        for column in (c for c in inputs.feature_plan.numeric if c in sourced):
            values = pandas.concat(
                [pandas.to_numeric(old[column]), pandas.to_numeric(new[column])]
            )
            labels = numpy.r_[numpy.zeros(len(old)), numpy.ones(len(new))]
            present = values.notna().to_numpy()
            if (
                present.sum() < MIN_PRESENT_ROWS
                or len(set(labels[present])) < BINARY_CLASSES
            ):
                continue
            auc = metrics.roc_auc_score(
                labels[present], values.to_numpy()[present]
            )
            found.append((column, abs(float(auc) - 0.5) * 2))
        for column in inputs.feature_plan.categorical:
            before = old[column].value_counts(normalize=True)
            after = new[column].value_counts(normalize=True)
            levels = before.index.union(after.index)
            distance = 0.5 * float(
                (
                    before.reindex(levels, fill_value=0.0)
                    - after.reindex(levels, fill_value=0.0)
                )
                .abs()
                .sum()
            )
            found.append((column, distance))
        found.sort(key=lambda item: -item[1])
        return [
            {"feature": name, "distance": float(distance)}
            for name, distance in found[:TOP_FEATURES]
        ]
