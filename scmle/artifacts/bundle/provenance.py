"""`Manifest`, the provenance and compatibility record of a bundle."""

from collections.abc import Mapping
import dataclasses
import platform
from typing import Any
from typing import ClassVar

import pandas
import sklearn

import scmle
from scmle.artifacts.bundle import environment
from scmle.artifacts.bundle import layout
from scmle.contracts import records


def truncate(version: str) -> list[str]:
    """Returns the `[major, minor]` components of a version string.

    Args:
        version: A dotted version string.

    Returns:
        The first two dot-separated parts as strings (fewer if the version has
        fewer).
    """
    return version.split(".")[:2]


@dataclasses.dataclass(frozen=True, slots=True)
class Manifest:
    """Provenance and compatibility metadata of a bundle.

    Attributes:
        model_name: Selected model family.
        model_version: Version label of this bundle.
        metric: Optimized metric.
        format: Bundle format version.
        scmle_version: Framework version that built it.
        python: Python version that built it.
        sklearn: scikit-learn version that built it.
        pandas: pandas version that built it.
        files: SHA-256 per stored file.
        model_params: Hyperparameter overrides of the selected model.
        strategy: Data strategy of the selected candidate.
        decision_threshold: Optimized binary decision threshold, if any.
        libraries: Versions of the installed libraries in `TRACKED_LIBRARIES`
            (numpy, scipy, joblib, lightgbm, xgboost, catboost, imblearn).
        calibration: Description of the probability calibration applied at
            inference, empty if none.
        environment: Every distribution installed when the bundle was built
            (filled by `Bundle.create`, empty otherwise), by name, so the
            environment can be rebuilt.
        platform: Operating system and architecture that built it.
        seed: Random seed of the run that built it, if known.
        configuration: Hash of the run configuration that built it.
        dataset: Hash of the training data that built it.
        split: Fingerprint of the train and test split it was fitted on.
        commit: Git commit of the scmle source that built it; empty when
            scmle ran from an installed wheel, whose version names the code.
        threads: Thread count per math library loaded when it was built.
        blas: BLAS implementation in use when it was built.
        KEYS: Version-1 keys to the written ones (`model`, `version`,
            `scmle`, `params`, `threshold`); `downgrade` restores version 1.
    """

    KEYS: ClassVar[records.Migration] = records.Migration(
        renames={
            "model_name": "model",
            "model_version": "version",
            "scmle_version": "scmle",
            "model_params": "params",
            "decision_threshold": "threshold",
        }
    )

    model_name: str
    model_version: str
    metric: str
    format: int = layout.BUNDLE_FORMAT
    scmle_version: str = scmle.__version__
    python: str = dataclasses.field(default_factory=platform.python_version)
    sklearn: str = sklearn.__version__
    pandas: str = pandas.__version__
    files: dict[str, str] = dataclasses.field(default_factory=dict)
    model_params: dict[str, Any] = dataclasses.field(default_factory=dict)
    strategy: str = "default"
    decision_threshold: float | None = None
    libraries: dict[str, str] = dataclasses.field(
        default_factory=environment.detect
    )
    calibration: str = ""
    environment: dict[str, str] = dataclasses.field(default_factory=dict)
    platform: str = ""
    seed: int | None = None
    configuration: str = ""
    dataset: str = ""
    split: str = ""
    commit: str = ""
    threads: dict[str, int] = dataclasses.field(default_factory=dict)
    blas: str = ""

    def environment_lock(self) -> str:
        """Renders the build environment as a pinned requirements listing.

        Returns:
            Header comment lines (Python version, platform, framework version)
            followed by one `name==version` line per installed distribution.
        """
        header = [
            "# scmle environment lock",
            f"# python {self.python}",
            f"# platform {self.platform or 'unknown'}",
            f"# scmle {self.scmle_version}",
        ]
        pins = [
            f"{name}=={version}" for name, version in self.environment.items()
        ]
        return "\n".join([*header, *pins]) + "\n"

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-serializable mapping.

        Returns:
            A dict of all fields under their written keys.
        """
        return self.KEYS.upgrade(dataclasses.asdict(self))

    @classmethod
    def read(cls, stored: Mapping[str, Any]) -> "Manifest":
        """Reads a stored manifest in either layout.

        Args:
            stored: A `model_manifest.json` record, with the written keys or
                with the version-1 field names.

        Returns:
            The manifest.

        Raises:
            InputError: If the record mixes an old key with its new one.
        """
        fields = {field.name for field in dataclasses.fields(cls)}
        layout = dict(stored)
        if set(layout) - fields:
            layout = cls.KEYS.downgrade(layout)
        return cls(**layout)

    def incompatibilities(self) -> list[str]:
        """Lists dependency mismatches with the running environment.

        Returns:
            One message per difference in the minor version of scikit-learn,
            Python, pandas or a library in `libraries` that is installed now;
            empty if all agree.
        """
        problems = []
        if truncate(self.sklearn) != truncate(sklearn.__version__):
            problems.append(
                f"scikit-learn {self.sklearn} (installed {sklearn.__version__})"
            )
        running = platform.python_version()
        if truncate(self.python) != truncate(running):
            problems.append(f"python {self.python} (running {running})")
        if truncate(self.pandas) != truncate(pandas.__version__):
            problems.append(
                f"pandas {self.pandas} (installed {pandas.__version__})"
            )
        current = environment.detect()
        for library, built in sorted(self.libraries.items()):
            now = current.get(library)
            if now is not None and truncate(built) != truncate(now):
                problems.append(f"{library} {built} (installed {now})")
        return problems
