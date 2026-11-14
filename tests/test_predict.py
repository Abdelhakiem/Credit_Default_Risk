import json
from unittest.mock import Mock

import numpy as np
import pytest

from credit_default_risk.predict import Predictor


@pytest.fixture
def predictor():
    model = Mock()
    model.feature_names_ = ["kind", "amount"]
    model.predict_proba.side_effect = lambda frame: np.array(
        [[1 - value, value] for value in (frame["amount"].fillna(0) / 10)]
    )
    manifest = {
        "feature_columns": ["kind", "amount"],
        "categorical_columns": ["kind"],
        "positive_class": 1,
        "decision_threshold": 0.5,
    }
    return Predictor(model, manifest, "7")


def test_single_prediction_prepares_values_and_threshold(predictor):
    result = predictor.predict_one({"amount": 8, "kind": None})
    frame = predictor.model.predict_proba.call_args.args[0]
    assert list(frame.columns) == ["kind", "amount"]
    assert frame.loc[0, "kind"] == "__MISSING__"
    assert result == {
        "default_probability": 0.8,
        "review_flag": True,
        "decision_threshold": 0.5,
        "model_version": "7",
    }


def test_batch_preserves_order(predictor):
    results = predictor.predict_batch(
        [
            {"kind": "A", "amount": 2},
            {"kind": "B", "amount": 8},
        ]
    )
    assert [result["default_probability"] for result in results] == [0.2, 0.8]
    assert [result["review_flag"] for result in results] == [False, True]


@pytest.mark.parametrize(
    "record",
    [
        {"kind": "A"},
        {"kind": "A", "amount": 2, "extra": 1},
        {"kind": "A", "amount": "bad"},
    ],
)
def test_rejects_invalid_features(predictor, record):
    with pytest.raises(ValueError):
        predictor.predict_one(record)


def test_rejects_model_manifest_mismatch(predictor):
    predictor.model.feature_names_ = ["amount", "kind"]
    with pytest.raises(ValueError, match="differ"):
        Predictor(predictor.model, predictor.manifest, "7")


def test_registry_failure_uses_local_bundle(monkeypatch, predictor):
    def unavailable(cls):
        raise ValueError("registry unavailable")

    monkeypatch.setattr(Predictor, "from_registry", classmethod(unavailable))
    local = Mock(return_value=predictor)
    monkeypatch.setattr(Predictor, "from_artifacts", local)
    assert Predictor.load() is predictor
    local.assert_called_once()


def test_local_bundle_loads_model_and_manifest(monkeypatch, tmp_path, predictor):
    (tmp_path / "manifest.json").write_text(json.dumps(predictor.manifest))
    (tmp_path / "model.cbm").write_bytes(b"model")
    (tmp_path / "registration.json").write_text('{"model_version":"7"}')
    fake_model = Mock()
    fake_model.feature_names_ = predictor.manifest["feature_columns"]
    monkeypatch.setattr(
        "credit_default_risk.predict.CatBoostClassifier", lambda: fake_model
    )
    result = Predictor.from_artifacts(tmp_path)
    assert result.version == "7"
    fake_model.load_model.assert_called_once_with(str(tmp_path / "model.cbm"))
