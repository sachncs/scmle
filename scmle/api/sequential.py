"""Public project API for durable sequential causal estimation."""

import pathlib

import pandas

from scmle.artifacts import reproducibility
from scmle.contracts import errors
from scmle.data.readers import tabular
from scmle.evaluation import sequential

SEQUENTIAL_ESTIMATION_ARTIFACT = "sequential_estimation.json"


def tabulate(
    data: pandas.DataFrame | str | pathlib.Path,
) -> pandas.DataFrame:
    """Resolves an in-memory frame or supported tabular file into a copy.

    Args:
        data: A pandas DataFrame or a path accepted by the standard tabular
            dataset loader.

    Returns:
        A caller-independent DataFrame ready for sequential validation.

    Raises:
        TypeError: If ``data`` is not a DataFrame, string or path. An
            unsupported file suffix raises from the loader.
    """
    if isinstance(data, pandas.DataFrame):
        return data.copy(deep=True)
    if isinstance(data, (str, pathlib.Path)):
        return tabular.Loader().load(pathlib.Path(data))
    raise errors.KindError(
        "data must be a pandas DataFrame or a path to a supported tabular file"
    )


class Estimation:
    """Runs and durably verifies a sequential causal estimation project.

    The project owns the input declaration, sequential estimand, numerical
    policy, and deterministic artifact location.  A run is not exposed as
    complete until the report has been atomically written and read back through
    its digest-verifying envelope.

    Attributes:
        data: The input as given.
        estimand: The sustained-treatment causal declaration.
        policy: The numerical, support, and truncation policy.
        directory: Directory containing the report artifact.
        artifact: Path of the report artifact.
        result: The latest result run or loaded, or None.
    """

    def __init__(
        self,
        data: pandas.DataFrame | str | pathlib.Path,
        estimand: sequential.Estimand,
        policy: sequential.Config,
        *,
        directory: str | pathlib.Path = "runs",
    ) -> None:
        """Defines a sequential estimation project.

        Args:
            data: In-memory tabular data or a supported tabular file path.
            estimand: Explicit sustained-treatment causal declaration.
            policy: Explicit numerical, support, and truncation policy.
            directory: Directory containing the deterministic report artifact.

        Raises:
            TypeError: If the input, estimand, or policy is unsupported.
        """
        if not isinstance(data, (pandas.DataFrame, str, pathlib.Path)):
            raise errors.KindError(
                "data must be a pandas DataFrame or a path to a supported "
                "tabular file"
            )
        if not isinstance(estimand, sequential.Estimand):
            raise errors.KindError("estimand must be Estimand")
        if not isinstance(policy, sequential.Config):
            raise errors.KindError("policy must be Config")
        self.data = data
        self.estimand = estimand
        self.policy = policy
        self.directory = pathlib.Path(directory)
        self.artifact = self.directory / SEQUENTIAL_ESTIMATION_ARTIFACT
        self.result: sequential.Estimation | None = None

    def run(self) -> sequential.Estimation:
        """Estimates, atomically persists, and reads back the verified result.

        The report is written to `artifact`, replacing any earlier one.

        Returns:
            The result as read back from the persisted report.

        Raises:
            ValueError: If the persisted report's digest differs from the digest
                of the report that was written.
        """
        frame = tabulate(self.data)
        estimate = sequential.Estimator(self.estimand, self.policy).fit(frame)
        expected_digest = estimate.write(self.artifact)
        verified = sequential.Estimation.read(self.artifact)
        if verified.digest() != expected_digest:
            raise errors.InputError(
                "persisted sequential report failed verification"
            )
        self.result = verified
        return verified

    def recall(self) -> sequential.Estimation:
        """Loads and verifies the durable report associated with this project.

        A missing report, or one whose envelope or digest fails verification,
        raises from the reader.

        Returns:
            The persisted result, which is also kept as `result`.
        """
        self.result = sequential.Estimation.read(self.artifact)
        return self.result

    def verify(
        self, data: pandas.DataFrame | str | pathlib.Path | None = None
    ) -> reproducibility.Comparison:
        """Replays the project and verifies input, policy, and result identity.

        The persisted report must exist (it is loaded first).

        Args:
            data: Optional replacement input used for the replay.  When it is
                omitted, the project's original input is resolved again.

        Returns:
            An exact verdict when the persisted and replayed evidence agree;
            otherwise a different verdict with explicit reasons.
        """
        persisted = self.recall()
        replay_input = self.data if data is None else data
        frame = tabulate(replay_input)
        reasons: list[str] = []
        observed_fingerprint = sequential.imprint(frame)
        if observed_fingerprint != persisted.digests["frame"]:
            reasons.append("input fingerprint differs from persisted evidence")
        if persisted.estimand.to_dict() != self.estimand.to_dict():
            reasons.append(
                "estimand declaration differs from persisted evidence"
            )
        if persisted.policy.to_dict() != self.policy.to_dict():
            reasons.append("estimation policy differs from persisted evidence")
        replayed = sequential.Estimator(self.estimand, self.policy).fit(frame)
        if replayed.digest() != persisted.digest():
            reasons.append(
                "replayed result digest differs from persisted evidence"
            )
        if reasons:
            return reproducibility.Comparison(
                level=reproducibility.DIFFERENT,
                reasons=reasons,
                discrepancy=float("nan"),
                compared=1,
                notes=[],
            )
        return reproducibility.Comparison(
            level=reproducibility.EXACT,
            reasons=[],
            discrepancy=0.0,
            compared=1,
            notes=["input identity, policy, and digest-bound result verified"],
        )
