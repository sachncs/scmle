"""Model bundle: a validated, portable inference pipeline with schema and checksums.

SECURITY: bundles are serialized with joblib (pickle). Loading executes code, so only load
bundles from sources you trust. Checksums detect corruption/tampering after save, not malice
at the source.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn

import mml
from mml.contracts.data import FeaturePlan
from mml.contracts.errors import ArtifactError, SchemaError
from mml.data.readers import read_table

BUNDLE_FORMAT = 1
MODEL_FILE = "pipeline.joblib"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class SchemaReport:
    missing: tuple[str, ...]
    extra: tuple[str, ...]
    missing_value_counts: dict[str, int]
    unseen_categories: dict[str, list[str]]

    @property
    def valid(self) -> bool:
        return not self.missing


class ModelBundle:
    def __init__(self, pipeline: Any, schema: dict[str, Any], info: dict[str, Any]) -> None:
        self.pipeline = pipeline
        self.schema = schema
        self.info = info

    # ---- construction -------------------------------------------------
    @classmethod
    def create(
        cls,
        pipeline: Any,
        plan: FeaturePlan,
        dev_X: pd.DataFrame,
        target: str,
        task: str,
        metric: str,
        model_name: str,
        classes: list | None,
        model_version: str = "v1",
    ) -> ModelBundle:
        categories = {
            c: sorted(map(str, dev_X[c].dropna().unique()))[:200] for c in plan.categorical
        }
        schema = {
            "target": target,
            "task": task,
            "numeric": list(plan.numeric),
            "categorical": list(plan.categorical),
            "dtypes": {c: str(dev_X[c].dtype) for c in plan.features},
            "known_categories": categories,
            "classes": [c.item() if hasattr(c, "item") else c for c in classes]
            if classes
            else None,
        }
        info = {
            "format": BUNDLE_FORMAT,
            "model_name": model_name,
            "model_version": model_version,
            "metric": metric,
            "magicml_version": mml.__version__,
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "pandas": pd.__version__,
        }
        return cls(pipeline, schema, info)

    # ---- persistence --------------------------------------------------
    def save(self, directory: str | Path) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        model_path = d / MODEL_FILE
        joblib.dump(self.pipeline, model_path)
        (d / "schema.json").write_text(json.dumps(self.schema, indent=2, default=str))
        manifest = {**self.info, "files": {MODEL_FILE: _sha256(model_path)}}
        (d / "model_manifest.json").write_text(json.dumps(manifest, indent=2))
        return d

    @classmethod
    def load(cls, directory: str | Path, *, allow_version_mismatch: bool = False) -> ModelBundle:
        d = Path(directory)
        try:
            manifest = json.loads((d / "model_manifest.json").read_text())
            schema = json.loads((d / "schema.json").read_text())
        except FileNotFoundError as exc:
            raise ArtifactError(f"not a model bundle (missing {exc.filename})") from exc
        if manifest.get("format") != BUNDLE_FORMAT:
            raise ArtifactError(f"unsupported bundle format {manifest.get('format')}")
        model_path = d / MODEL_FILE
        if not model_path.exists():
            raise ArtifactError(f"missing {MODEL_FILE}")
        if _sha256(model_path) != manifest["files"][MODEL_FILE]:
            raise ArtifactError("checksum mismatch: model file was modified or corrupted")
        mismatches = []
        if manifest["sklearn"].split(".")[:2] != sklearn.__version__.split(".")[:2]:
            mismatches.append(
                f"scikit-learn {manifest['sklearn']} (installed {sklearn.__version__})"
            )
        if manifest["python"].split(".")[:2] != platform.python_version().split(".")[:2]:
            mismatches.append(f"python {manifest['python']} (running {sys.version.split()[0]})")
        if mismatches and not allow_version_mismatch:
            raise ArtifactError(
                "bundle was built with different dependencies: " + "; ".join(mismatches),
                {"mismatches": mismatches},
            )
        return cls(joblib.load(model_path), schema, manifest)

    # ---- inference ----------------------------------------------------
    @property
    def features(self) -> list[str]:
        return self.schema["numeric"] + self.schema["categorical"]

    @property
    def classes(self) -> list | None:
        return self.schema.get("classes")

    def _frame(self, data: str | Path | pd.DataFrame) -> pd.DataFrame:
        return data.copy() if isinstance(data, pd.DataFrame) else read_table(data)

    def validate(self, data: str | Path | pd.DataFrame) -> SchemaReport:
        df = self._frame(data)
        missing = tuple(c for c in self.features if c not in df.columns)
        extra = tuple(
            c for c in df.columns if c not in self.features and c != self.schema["target"]
        )
        present = [c for c in self.features if c in df.columns]
        counts = {c: int(df[c].isna().sum()) for c in present if df[c].isna().any()}
        unseen = {}
        for c in self.schema["categorical"]:
            if c in df.columns:
                known = set(self.schema["known_categories"][c])
                new = sorted(set(df[c].dropna().astype(str)) - known)
                if new:
                    unseen[c] = new[:20]
        return SchemaReport(missing, extra, counts, unseen)

    def _prepare(self, data: str | Path | pd.DataFrame) -> pd.DataFrame:
        df = self._frame(data)
        report = self.validate(df)
        if not report.valid:
            raise SchemaError(
                f"input is missing required columns: {list(report.missing)}",
                {"missing": list(report.missing)},
            )
        bad = []
        for c in self.schema["numeric"]:
            coerced = pd.to_numeric(df[c], errors="coerce")
            if (coerced.isna() & df[c].notna()).any():
                bad.append(c)
        if bad:
            raise SchemaError(f"non-numeric values in numeric columns: {bad}", {"columns": bad})
        return df[self.features]

    def predict(self, data: str | Path | pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict(self._prepare(data))

    def predict_proba(self, data: str | Path | pd.DataFrame) -> np.ndarray:
        if not hasattr(self.pipeline, "predict_proba") or self.schema["task"] == "regression":
            raise ArtifactError("this model does not provide probabilities")
        return self.pipeline.predict_proba(self._prepare(data))
