"""Public project API for durable sequential causal estimation."""

import pathlib
from typing import TypeAlias

import pandas as pd

from scmle.artifacts import reproducibility
from scmle.data.readers import tabular
from scmle.evaluation import sequential

TabularInput: TypeAlias = pd.DataFrame | str | pathlib.Path
SEQUENTIAL_ESTIMATION_ARTIFACT = "sequential_estimation.json"


def resolve_tabular_input(data: TabularInput) -> pd.DataFrame:
    """Resolves an in-memory frame or supported tabular file into a copy.

    Args:
        data: A pandas DataFrame or a path accepted by the standard tabular
            dataset loader.

    Returns:
        A caller-independent DataFrame ready for sequential validation.

    Raises:
        TypeError: If ``data`` is not a supported tabular input.
    """
    if isinstance(data, pd.DataFrame):
        return data.copy(deep=True)
    if isinstance(data, (str, pathlib.Path)):
        return tabular.DatasetLoader().load(pathlib.Path(data))
    raise TypeError(
        "data must be a pandas DataFrame or a path to a supported tabular file"
    )


class SequentialEstimationProject:
    """Runs and durably verifies a sequential causal estimation project.

    The project owns the input declaration, sequential estimand, numerical
    policy, and deterministic artifact location.  A run is not exposed as
    complete until the report has been atomically written and read back through
    its digest-verifying envelope.

    Attributes:
        data: The input as given.
        estimand: The sustained-treatment causal declaration.
        policy: The numerical, support, and truncation policy.
        output_dir: Directory containing the report artifact.
        artifact_path: Path of the report artifact.
        result: The latest result run or loaded, or None.
    """

    def __init__(
        self,
        data: TabularInput,
        estimand: sequential.SequentialEstimand,
        policy: sequential.SequentialEstimationPolicy,
        *,
        output_dir: str | pathlib.Path = "runs",
    ) -> None:
        """Defines a sequential estimation project.

        Args:
            data: In-memory tabular data or a supported tabular file path.
            estimand: Explicit sustained-treatment causal declaration.
            policy: Explicit numerical, support, and truncation policy.
            output_dir: Directory containing the deterministic report artifact.

        Raises:
            TypeError: If the input, estimand, or policy is unsupported.
        """
        if not isinstance(data, (pd.DataFrame, str, pathlib.Path)):
            raise TypeError(
                "data must be a pandas DataFrame or a path to a supported "
                "tabular file"
            )
        if not isinstance(estimand, sequential.SequentialEstimand):
            raise TypeError("estimand must be SequentialEstimand")
        if not isinstance(policy, sequential.SequentialEstimationPolicy):
            raise TypeError("policy must be SequentialEstimationPolicy")
        self.data = data
        self.estimand = estimand
        self.policy = policy
        self.output_dir = pathlib.Path(output_dir)
        self.artifact_path = self.output_dir / SEQUENTIAL_ESTIMATION_ARTIFACT
        self.result: sequential.SequentialEstimationResult | None = None

    def run(self) -> sequential.SequentialEstimationResult:
        """Estimates, atomically persists, and reads back the verified result.

        Returns:
            The result as read back from the persisted report.

        Raises:
            ValueError: If the persisted report's digest differs from the digest
                of the report that was written.
        """
        frame = resolve_tabular_input(self.data)
        estimate = sequential.SequentialIPWEstimator(
            self.estimand, self.policy
        ).fit(frame)
        expected_digest = estimate.write(self.artifact_path)
        verified = sequential.SequentialEstimationResult.read(
            self.artifact_path
        )
        if verified.digest() != expected_digest:
            raise ValueError("persisted sequential report failed verification")
        self.result = verified
        return verified

    def load_result(self) -> sequential.SequentialEstimationResult:
        """Loads and verifies the durable report associated with this project.

        Returns:
            The persisted result, which is also kept as `result`.
        """
        self.result = sequential.SequentialEstimationResult.read(
            self.artifact_path
        )
        return self.result

    def verify_replay(
        self, data: TabularInput | None = None
    ) -> reproducibility.ReplayVerdict:
        """Replay the project and verify input, policy, and result identity.

        Args:
            data: Optional replacement input used for the replay.  When it is
                omitted, the project's original input is resolved again.

        Returns:
            An exact verdict when the persisted and replayed evidence agree;
            otherwise a different verdict with explicit reasons.
        """
        persisted = self.load_result()
        replay_input = self.data if data is None else data
        frame = resolve_tabular_input(replay_input)
        reasons: list[str] = []
        observed_fingerprint = sequential.fingerprint_frame(frame)
        if observed_fingerprint != persisted.raw_input_fingerprint:
            reasons.append("input fingerprint differs from persisted evidence")
        if persisted.estimand.to_dict() != self.estimand.to_dict():
            reasons.append(
                "estimand declaration differs from persisted evidence"
            )
        if persisted.policy.to_dict() != self.policy.to_dict():
            reasons.append("estimation policy differs from persisted evidence")
        replayed = sequential.SequentialIPWEstimator(
            self.estimand, self.policy
        ).fit(frame)
        if replayed.digest() != persisted.digest():
            reasons.append(
                "replayed result digest differs from persisted evidence"
            )
        if reasons:
            return reproducibility.ReplayVerdict(
                level=reproducibility.DIFFERENT,
                reasons=reasons,
                max_abs_difference=float("nan"),
                trials_compared=1,
                notes=[],
            )
        return reproducibility.ReplayVerdict(
            level=reproducibility.EXACT,
            reasons=[],
            max_abs_difference=0.0,
            trials_compared=1,
            notes=["input identity, policy, and digest-bound result verified"],
        )
