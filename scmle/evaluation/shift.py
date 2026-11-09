"""Adversarial validation: can a model tell training rows from new rows?

If a classifier separates the development rows from the evaluation rows
better than chance, the two differ in distribution, and a score measured on
one says less about the other. The report also names the features that differ
most so the cause can be inspected.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn import metrics as sk_metrics
from sklearn import model_selection

from scmle.evaluation import diagnostics
from scmle.evaluation import shift_weights

FOLDS = 5
MAX_ROWS = 4000
MODERATE_AUC = 0.60
STRONG_AUC = 0.75
TOP_FEATURES = 5


class AdversarialValidation(diagnostics.Diagnostic):
    """Separability of development rows from evaluation rows."""

    name = "adversarial_validation"

    def compute(self, inputs: diagnostics.DiagnosticInput) -> dict[str, Any]:
        """Cross-validates a classifier that predicts which set a row is from.

        Args:
            inputs: Evaluation data, with the development rows attached as
                `reference_features`.

        Returns:
            The separability AUC (0.5 means indistinguishable), a verdict,
            and the features whose distributions differ most.
        """
        reference = inputs.reference_features
        if reference is None:
            return {"available": False, "reason": "no reference rows"}
        columns = list(inputs.feature_plan.input_columns)
        rng = np.random.default_rng(inputs.seed)
        old = self.sample(reference[columns], rng)
        new = self.sample(inputs.features[columns], rng)
        frame = pd.concat([old, new], ignore_index=True)
        labels = np.r_[np.zeros(len(old)), np.ones(len(new))]
        encoded = self.encode(frame, inputs)
        scores = model_selection.cross_val_predict(
            ensemble.HistGradientBoostingClassifier(
                max_iter=60, random_state=inputs.seed
            ),
            encoded,
            labels,
            cv=model_selection.StratifiedKFold(
                FOLDS, shuffle=True, random_state=inputs.seed
            ),
            method="predict_proba",
        )[:, 1]
        auc = float(sk_metrics.roc_auc_score(labels, scores))
        return {
            "available": True,
            "auc": auc,
            "verdict": self.verdict(auc),
            "odds": self.odds(scores, len(old)),
            "most_shifted": self.most_shifted(old, new, inputs),
            "rows": {"reference": len(old), "evaluation": len(new)},
        }

    @staticmethod
    def odds(scores: np.ndarray, reference_rows: int) -> dict[str, Any]:
        """The odds that weighted conformal would use, from the same scores.

        The out-of-fold probability `p` that a row is an evaluation row
        gives the odds `p / (1 - p)`, corrected for the sample sizes and
        clipped at `shift_weights.DEFAULT_CLIP`, the ratio estimate of
        `scmle.evaluation.shift_weights.estimate_weights` (which uses a
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
            is `possible` or `refused` by the reference effective size, and a
            note.
        """
        new_rows = len(scores) - reference_rows
        clip = shift_weights.DEFAULT_CLIP
        raw = (
            scores / np.maximum(1.0 - scores, 1e-12) * reference_rows / new_rows
        )
        ratio = np.minimum(raw, clip)

        def describe(
            values: np.ndarray, unclipped: np.ndarray
        ) -> dict[str, Any]:
            return {
                "mean": float(values.mean()),
                "p95": float(np.quantile(values, 0.95)),
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
            "weights for a bundle are estimated by attach_shift_weights",
        }

    @staticmethod
    def sample(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
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
        return frame.iloc[np.sort(chosen)].reset_index(drop=True)

    @staticmethod
    def encode(
        frame: pd.DataFrame, inputs: diagnostics.DiagnosticInput
    ) -> pd.DataFrame:
        """Numeric view of the rows: numbers as is, categories as codes.

        Args:
            frame: Rows holding the feature columns.
            inputs: The diagnostic inputs naming the numeric and
                categorical columns.

        Returns:
            A copy of `frame` with categorical columns replaced by category
            codes and numeric columns converted to numbers.
        """
        encoded = frame.copy()
        for column in inputs.feature_plan.categorical:
            encoded[column] = encoded[column].astype("category").cat.codes
        for column in inputs.feature_plan.numeric:
            encoded[column] = pd.to_numeric(encoded[column])
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
    def most_shifted(
        old: pd.DataFrame,
        new: pd.DataFrame,
        inputs: diagnostics.DiagnosticInput,
    ) -> list[dict[str, Any]]:
        """Ranks features by how separable the two sets are on that feature.

        Numeric features use the single-feature AUC distance from 0.5;
        categorical features use the total-variation distance between level
        frequencies. Both lie in [0, 1] and are comparable as a ranking.

        Args:
            old: Sampled reference rows.
            new: Sampled evaluation rows.
            inputs: The diagnostic inputs naming the numeric and
                categorical columns.

        Returns:
            Up to `TOP_FEATURES` entries of feature name and distance,
            largest distance first.
        """
        found = []
        for column in inputs.feature_plan.numeric:
            values = pd.concat(
                [pd.to_numeric(old[column]), pd.to_numeric(new[column])]
            )
            labels = np.r_[np.zeros(len(old)), np.ones(len(new))]
            present = values.notna().to_numpy()
            if present.sum() < 2 or len(set(labels[present])) < 2:
                continue
            auc = sk_metrics.roc_auc_score(
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
