"""Predict default risk from prepared, 60-feature application records.

Input is a JSON object for one applicant or a JSON list for a batch. Each record
must contain every feature in the model manifest; categorical and numeric values
may be null. Raw application and history tables are not accepted here.
"""

import argparse
import json
import logging
import math
import sys
from collections.abc import Mapping, Sequence
from functools import wraps
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from .config import ARTIFACT_DIR, MLFLOW_TRACKING_URI, MODEL_ALIAS, MODEL_NAME
from .features import prepare_final_catboost_frame
from .logging_conf import configure_logging

logger = logging.getLogger(__name__)


def timed(func):
    """Log execution time even when the wrapped call raises."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        started = perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            logger.info(
                "Execution timed: function=%s latency_ms=%.2f",
                func.__name__,
                (perf_counter() - started) * 1000,
            )

    return wrapper


class Predictor:
    """A loaded model and its matching inference contract."""

    def __init__(self, model: CatBoostClassifier, manifest: dict, version: str | None):
        self.model = model
        self.manifest = manifest
        self.version = version
        columns = manifest.get("feature_columns")
        categorical = manifest.get("categorical_columns")
        if (
            not isinstance(columns, list)
            or not columns
            or len(columns) != len(set(columns))
        ):
            raise ValueError("Manifest requires unique feature_columns")
        if not isinstance(categorical, list) or not set(categorical) <= set(columns):
            raise ValueError("Manifest has invalid categorical_columns")
        if list(model.feature_names_) != columns:
            raise ValueError("Model feature names differ from the manifest")
        if manifest.get("positive_class") != 1:
            raise ValueError("Expected positive class 1 in the manifest")
        threshold = manifest.get("decision_threshold")
        if threshold is not None and (
            not isinstance(threshold, (int, float))
            or not math.isfinite(threshold)
            or not 0 <= threshold <= 1
        ):
            raise ValueError("Invalid decision threshold in the manifest")
        self.threshold = threshold

    @classmethod
    def load(
        cls, artifact_dir: Path = ARTIFACT_DIR, source: str = "registry"
    ) -> "Predictor":
        """Prefer the registered alias; use the local bundle if it is unavailable."""
        if source == "local":
            return cls.from_artifacts(artifact_dir)
        if source != "registry":
            raise ValueError(f"Unknown model source: {source}")
        import mlflow

        try:
            return cls.from_registry()
        except (mlflow.exceptions.MlflowException, ValueError, OSError) as exc:
            logger.warning(
                "Registry model load failed; trying local artifacts: reason=%s", exc
            )
            return cls.from_artifacts(artifact_dir)

    @classmethod
    def from_registry(cls) -> "Predictor":
        """Pin the alias to one version and load its model and run manifest."""
        import mlflow
        from mlflow import MlflowClient

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        version = MlflowClient().get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
        if not version.run_id:
            raise ValueError("Registered version has no run ID for its manifest")
        model = mlflow.catboost.load_model(f"models:/{MODEL_NAME}/{version.version}")
        manifest = mlflow.artifacts.load_dict(
            f"runs:/{version.run_id}/inference/manifest.json"
        )
        predictor = cls(model, manifest, str(version.version))
        logger.info(
            "Model loaded: source=registry model=%s alias=%s version=%s",
            MODEL_NAME,
            MODEL_ALIAS,
            version.version,
        )
        return predictor

    @classmethod
    def from_artifacts(cls, artifact_dir: Path = ARTIFACT_DIR) -> "Predictor":
        artifact_dir = Path(artifact_dir)
        try:
            manifest = json.loads(
                (artifact_dir / "manifest.json").read_text(encoding="utf-8")
            )
            model = CatBoostClassifier()
            model.load_model(str(artifact_dir / "model.cbm"))
            registration_path = artifact_dir / "registration.json"
            registration = (
                json.loads(registration_path.read_text(encoding="utf-8"))
                if registration_path.exists()
                else {}
            )
            predictor = cls(model, manifest, registration.get("model_version"))
        except Exception:
            logger.exception("Model load failure: artifact_dir=%s", artifact_dir)
            raise
        logger.info(
            "Model loaded: source=local artifact_dir=%s version=%s",
            artifact_dir,
            predictor.version,
        )
        return predictor

    def predict_batch(self, records: Sequence[Mapping[str, object]]) -> list[dict]:
        """Return one result per record, preserving input order."""
        if not records:
            raise ValueError("Prediction batch must contain at least one record")
        columns = self.manifest["feature_columns"]
        for index, record in enumerate(records):
            if not isinstance(record, Mapping):
                raise TypeError(f"Record {index} must be a JSON object")
            missing = set(columns) - record.keys()
            extra = record.keys() - set(columns)
            if missing or extra:
                raise ValueError(
                    f"Record {index} has missing features {sorted(missing)} and unexpected features {sorted(extra)}"
                )

        started = perf_counter()
        frame = pd.DataFrame.from_records(records, columns=columns)
        categorical = set(self.manifest["categorical_columns"])
        for column in columns:
            if column in categorical:
                frame[column] = (
                    frame[column].astype("string").fillna("__MISSING__").astype(object)
                )
            else:
                try:
                    frame[column] = pd.to_numeric(frame[column], errors="raise").astype(
                        "float64"
                    )
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Feature {column} must be numeric or null"
                    ) from exc
        prepared = prepare_final_catboost_frame(frame, self.manifest)
        probabilities = np.asarray(
            self.model.predict_proba(prepared)[:, 1], dtype=float
        )
        if (
            not np.isfinite(probabilities).all()
            or ((probabilities < 0) | (probabilities > 1)).any()
        ):
            raise ValueError("Model returned invalid probabilities")
        results = [
            {
                "default_probability": float(probability),
                "review_flag": (
                    bool(probability >= self.threshold)
                    if self.threshold is not None
                    else None
                ),
                "decision_threshold": self.threshold,
                "model_version": self.version,
            }
            for probability in probabilities
        ]
        logger.info(
            "Prediction served: count=%d latency_ms=%.2f model_version=%s",
            len(results),
            (perf_counter() - started) * 1000,
            self.version,
        )
        return results

    @timed
    def predict_one(self, record: Mapping[str, object]) -> dict:
        """Return a single applicant's prediction."""
        return self.predict_batch([record])[0]


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Predict from engineered feature JSON")
    parser.add_argument(
        "--input", type=Path, required=True, help="JSON object or list of objects"
    )
    parser.add_argument("--artifacts", type=Path, default=ARTIFACT_DIR)
    parser.add_argument("--source", choices=("registry", "local"), default="registry")
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    predictor = Predictor.load(args.artifacts, source=args.source)
    if isinstance(payload, dict):
        output = predictor.predict_one(payload)
    elif isinstance(payload, list):
        output = predictor.predict_batch(payload)
    else:
        parser.error("Input JSON must be an object or a list of objects")
    sys.stdout.write(json.dumps(output, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
