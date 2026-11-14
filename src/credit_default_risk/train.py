"""Fit the selected CatBoost model, evaluate, save, and register it.

Run with ``python -m credit_default_risk.train`` after preparing features.
Set MLFLOW_TRACKING_URI to override the local MLflow server address.
"""

import json
import logging
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from mlflow import MlflowClient
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from .config import (
    FINAL_SELECTED_COLUMNS,
    ID_COL,
    PROCESSED_TEST_PATH,
    PROCESSED_TRAIN_PATH,
    PROJECT_ROOT,
    PROPERTY_SUFFIX,
    RANDOM_STATE,
    TARGET_COL,
    TEST_SIZE,
    WINNING_RUN_ID,
    MODEL_NAME,
    MLFLOW_TRACKING_URI,
    ARTIFACT_DIR,
    DEFAULT_PARAMS
)
from .features import prepare_catboost_frame
from .helper import select_final_features
from .logging_conf import configure_logging

logger = logging.getLogger(__name__)

def load_winning_params(run_id: str = WINNING_RUN_ID) -> tuple[dict, str]:
    """Reconstruct final fit settings from the selected finished tuning run."""
    try:
        run = mlflow.get_run(run_id)
        if run.info.status != "FINISHED" or run.data.tags.get("model_family") != "catboost":
            raise ValueError("Winning run is not a finished CatBoost run")
        p, m = run.data.params, run.data.metrics
        params = {
            "iterations": int(m["best_iteration"]) + 1,
            "learning_rate": float(p["learning_rate"]),
            "depth": int(p["depth"]),
            "l2_leaf_reg": float(p["l2_leaf_reg"]),
            "random_strength": float(p["random_strength"]),
            "auto_class_weights": None if p.get("auto_class_weights") in (None, "None") else p["auto_class_weights"],
            "loss_function": p["loss_function"],
            "eval_metric": p["eval_metric"],
            "random_seed": int(p["random_seed"]),
            "verbose": False,
        }
        logger.info("Winning parameters loaded: run_id=%s", run_id)
        return params, "winning_run"
    except (KeyError, ValueError, mlflow.exceptions.MlflowException) as exc:
        logger.warning("Winning run unavailable or invalid; using default parameters: run_id=%s reason=%s", run_id, exc)
        return DEFAULT_PARAMS.copy(), "defaults"


def load_split(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if {ID_COL, TARGET_COL} - set(frame.columns):
        raise ValueError(f"Missing ID or target in {path}")
    if frame[ID_COL].isna().any() or frame[ID_COL].duplicated().any():
        raise ValueError(f"Invalid application IDs in {path}")
    if frame[TARGET_COL].isna().any() or not frame[TARGET_COL].isin([0, 1]).all():
        raise ValueError(f"Target must be binary in {path}")
    return frame


def main() -> tuple[str, str]:
    configure_logging()
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or MLFLOW_TRACKING_URI
    mlflow.set_tracking_uri(tracking_uri)
    logger.info("MLflow tracking store: uri=%s", tracking_uri)
    experiment = mlflow.set_experiment("credit-default-risk")
    if experiment.artifact_location.startswith("mlflow-artifacts:") and not tracking_uri.startswith(("http://", "https://")):
        raise RuntimeError(
            "This experiment uses mlflow-artifacts:/ and requires an HTTP MLflow "
            "tracking server for artifact uploads. Set MLFLOW_TRACKING_URI to its URL."
        )

    train = load_split(PROCESSED_TRAIN_PATH)
    test = load_split(PROCESSED_TEST_PATH)
    if train[ID_COL].isin(test[ID_COL]).any():
        raise ValueError("Processed train and test IDs overlap")
    X_train = prepare_catboost_frame(select_final_features(train, FINAL_SELECTED_COLUMNS))
    X_test = prepare_catboost_frame(select_final_features(test, FINAL_SELECTED_COLUMNS))
    if X_train.columns.tolist() != X_test.columns.tolist() or not X_train.dtypes.equals(X_test.dtypes):
        raise ValueError("Train and test model schemas differ")
    categorical = X_train.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    params, source = load_winning_params()
    logger.info("Training started: rows=%d test_rows=%d features=%d parameter_source=%s", len(train), len(test), X_train.shape[1], source)

    model = CatBoostClassifier(**params)
    model.fit(X_train, train[TARGET_COL], cat_features=categorical)
    probabilities = model.predict_proba(X_test)[:, 1]
    if not np.isfinite(probabilities).all():
        raise ValueError("Model returned non-finite probabilities")
    metrics = {
        "test_average_precision": float(average_precision_score(test[TARGET_COL], probabilities)),
        "test_roc_auc": float(roc_auc_score(test[TARGET_COL], probabilities)),
        "test_brier_score": float(brier_score_loss(test[TARGET_COL], probabilities)),
        "test_log_loss": float(log_loss(test[TARGET_COL], probabilities)),
        "test_default_rate": float(test[TARGET_COL].mean()),
        "test_mean_predicted_risk": float(probabilities.mean()),
    }
    manifest = {
        "model_family": "CatBoost",
        "source_mlflow_run_id": WINNING_RUN_ID if source == "winning_run" else None,
        "parameter_source": source,
        "feature_columns": X_train.columns.tolist(),
        "categorical_columns": categorical,
        "property_suffix": PROPERTY_SUFFIX,
        "parameters": params,
        "positive_class": 1,
        "probability_output": "predict_proba(X)[:, 1]",
        "decision_threshold": None,
        "threshold_note": "A threshold must be selected on validation data for this training split.",
        "test_size": TEST_SIZE,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(ARTIFACT_DIR / "model.cbm"))
    with (ARTIFACT_DIR / "model.pkl").open("wb") as stream:
        pickle.dump(model, stream, protocol=pickle.HIGHEST_PROTOCOL)
    (ARTIFACT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (ARTIFACT_DIR / "evaluation.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    with mlflow.start_run(run_name="catboost-final-packaged") as run:
        mlflow.set_tags({"model_family": "catboost", "stage": "final_test", "parameter_source": source})
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_dict(manifest, "inference/manifest.json")
        mlflow.log_dict(metrics, "evaluation/report.json")
        mlflow.log_artifact(str(ARTIFACT_DIR / "model.pkl"), artifact_path="serialization")
        model_info = mlflow.catboost.log_model(model, name="model", input_example=X_train.head(5))
        run_id = run.info.run_id

    version = mlflow.register_model(model_uri=model_info.model_uri, name=MODEL_NAME)
    MlflowClient().set_model_version_tag(MODEL_NAME, version.version, "parameter_source", source)
    registration = {"run_id": run_id, "model_name": MODEL_NAME, "model_version": str(version.version), "model_uri": model_info.model_uri}
    (ARTIFACT_DIR / "registration.json").write_text(json.dumps(registration, indent=2), encoding="utf-8")
    logger.info("Model registered: run_id=%s model=%s version=%s", run_id, MODEL_NAME, version.version)
    logger.info("Artifacts saved: path=%s", ARTIFACT_DIR)
    return run_id, str(version.version)


if __name__ == "__main__":
    main()
