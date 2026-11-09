"""The prediction sets of a served bundle as input to the closed loop.

`Loop.observe` routes a batch by a score that grows with the size of the
conformal set and with novelty, and `Loop.mature` feeds the loop's coverage
monitor with whether each case's outcome was inside its set. This feed builds
both from a bundle: the set size and the out-of-distribution flag of every
served case go to `observe`, and when the outcomes arrive the cases whose
label was outside the set are counted as coverage violations, written to the
deployment ledger as a chained `observe_summary` row and handed to `mature`.

What it does not do: it gates nothing. A run of violations can make the
loop's coverage monitor escalate (`coverage_below_nominal`), exactly as it
already did for any caller of `mature`; it never retrains, and the policy's
`alarm_gates_retrain` is not read or changed. Classification only: sets of a
regression bundle have no size to route on. The monitor in the loop tests the
share of covered cases against `LoopPolicy.coverage_target`, so that target
should be the nominal level used here.
"""

from collections.abc import Mapping
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from scmle.contracts import errors
from scmle.governance import deployment_ledger
from scmle.governance import loop as loop_module

SOURCE = "uncertainty_feed"


class UncertaintyFeed:
    """Feeds one loop from one classification bundle.

    Attributes:
        loop: The closed loop that receives the batches and the violations.
        bundle: The classification bundle that produces the sets.
        nominal: Coverage level requested for the sets.
        sets: Prediction set of every served case, by case id.
    """

    def __init__(
        self, loop: loop_module.Loop, bundle: Any, nominal: float = 0.9
    ) -> None:
        """Binds the loop and the bundle.

        Args:
            loop: The closed loop that receives the batches.
            bundle: A classification bundle with conformal calibration.
            nominal: Coverage level requested for the prediction sets, in
                the open interval (0, 1).

        Raises:
            ValueError: If `nominal` is not strictly between 0 and 1.
        """
        if not 0.0 < nominal < 1.0:
            raise ValueError("nominal must lie strictly between 0 and 1")
        self.loop = loop
        self.bundle = bundle
        self.nominal = nominal
        self.sets: dict[str, list[Any]] = {}

    def observe(
        self,
        features: pd.DataFrame,
        predictions: Sequence[Any],
        decisions: Sequence[Any],
        case_ids: Sequence[str],
    ) -> Any:
        """Routes a served batch by its set sizes and novelty.

        The size of each set is at least 1, and a row is flagged novel when
        the bundle has a `novelty` detector that marks it out of
        distribution; without one no row is novel. The set of every row is
        stored by case id for `mature`. A bundle that cannot produce
        classification sets fails in `predict_set` (an `ArtifactError`).

        Args:
            features: The served rows.
            predictions: The served predictions, one per row.
            decisions: The decisions taken, one per row.
            case_ids: One identifier per row; the sets are remembered by it.

        Returns:
            The loop's outcome for the batch, as returned by `Loop.observe`.

        Raises:
            ValueError: If there is not one case id per row.
        """
        if len(case_ids) != len(features):
            raise ValueError("one case id per row")
        found = self.bundle.predict_set(features, self.nominal)
        sizes = [max(len(item), 1) for item in found]
        novelty = getattr(self.bundle, "novelty", None)
        novel = (
            np.asarray(
                novelty.score(features)["out_of_distribution"], dtype=bool
            )
            if novelty is not None
            else np.zeros(len(features), dtype=bool)
        )
        for identity, item in zip(case_ids, found, strict=True):
            self.sets[str(identity)] = list(item)
        frame = pd.DataFrame(
            {
                "case_id": [str(i) for i in case_ids],
                "set_size": sizes,
                "novel": novel,
            }
        )
        return self.loop.observe(frame, predictions, decisions)

    def mature(
        self,
        labels: Mapping[str, Any],
        losses: Mapping[str, float],
    ) -> int:
        """Counts coverage violations of arrived outcomes and hands them on.

        Appends an `observe_summary` row to the loop's ledger with the number
        of cases, the violations and `gates_retrain` false, then calls
        `Loop.mature` with the losses and the per-case coverage (1.0 when the
        label is in the stored set, else 0.0). The ledger row records the
        loop's latest batch index, which need not be the batch of these cases.

        Args:
            labels: True label by case id.
            losses: Loss of the deployed model by case id, passed to
                `Loop.mature` unchanged.

        Returns:
            The number of cases in `labels` whose label was outside the
            stored set.

        Raises:
            ScmleError: If a label belongs to a case this feed did not serve.
        """
        unknown = sorted(str(i) for i in labels if str(i) not in self.sets)
        if unknown:
            raise errors.ScmleError(
                f"labels for cases this feed did not serve: {unknown[:3]}"
            )
        covered = {
            str(i): float(label in self.sets[str(i)])
            for i, label in labels.items()
        }
        violations = sum(1 for v in covered.values() if v == 0.0)
        self.loop.ledger.commit(
            deployment_ledger.OBSERVE_SUMMARY,
            {
                "source": SOURCE,
                "batch": self.loop.batch - 1,
                "cases": len(covered),
                "violations": violations,
                "nominal": self.nominal,
                "gates_retrain": False,
            },
        )
        self.loop.mature(dict(losses), covered=covered)
        return violations
