from types import SimpleNamespace

import pandas as pd
import pytest

from credit_default_risk import train
from credit_default_risk.config import ID_COL, TARGET_COL


def test_winning_run_parameters(monkeypatch):
    run = SimpleNamespace(
        info=SimpleNamespace(status="FINISHED"),
        data=SimpleNamespace(
            tags={"model_family": "catboost"},
            params={
                "learning_rate": "0.05",
                "depth": "5",
                "l2_leaf_reg": "9.5",
                "random_strength": "1.2",
                "auto_class_weights": "None",
                "loss_function": "Logloss",
                "eval_metric": "AUC",
                "random_seed": "42",
            },
            metrics={"best_iteration": 1493},
        ),
    )
    monkeypatch.setattr(train.mlflow, "get_run", lambda run_id: run)
    params, source = train.load_winning_params("winner")
    assert source == "winning_run"
    assert params["iterations"] == 1494
    assert params["auto_class_weights"] is None


def test_missing_winning_run_uses_defaults(monkeypatch):
    def missing(run_id):
        raise train.mlflow.exceptions.MlflowException("not found")

    monkeypatch.setattr(train.mlflow, "get_run", missing)
    params, source = train.load_winning_params("missing")
    assert source == "defaults"
    assert params == train.DEFAULT_PARAMS


@pytest.mark.parametrize(
    "ids,targets", [([1, 1], [0, 1]), ([1, 2], [0, 2]), ([1, 2], [0, None])]
)
def test_load_split_rejects_invalid_rows(tmp_path, ids, targets):
    path = tmp_path / "split.parquet"
    pd.DataFrame({ID_COL: ids, TARGET_COL: targets}).to_parquet(path)
    with pytest.raises(ValueError):
        train.load_split(path)
