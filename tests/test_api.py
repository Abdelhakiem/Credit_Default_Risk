from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from credit_default_risk.api.main import app
from credit_default_risk.predict import Predictor


@pytest.fixture
def fake_predictor(monkeypatch):
    predictor = Mock(spec=Predictor)
    predictor.version = "7"
    predictor.threshold = 0.5
    predictor.manifest = {"feature_columns": ["amount"]}
    predictor.predict_one.return_value = {
        "default_probability": 0.8,
        "review_flag": True,
        "decision_threshold": 0.5,
        "model_version": "7",
    }
    predictor.predict_batch.return_value = [predictor.predict_one.return_value]
    loader = Mock(return_value=predictor)
    monkeypatch.setattr(Predictor, "load", loader)
    return predictor, loader


def test_health_loads_model_once(fake_predictor):
    _, loader = fake_predictor
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/health").json() == {"status": "ready"}
        assert client.get("/metadata").json()["feature_count"] == 1
    loader.assert_called_once_with()


def test_predict_happy_path_and_response_schema(fake_predictor):
    predictor, _ = fake_predictor
    with TestClient(app) as client:
        response = client.post("/predict", json={"features": {"amount": 8}})
    assert response.status_code == 200
    assert set(response.json()) == {
        "default_probability",
        "review_flag",
        "decision_threshold",
        "model_version",
    }
    assert response.json()["default_probability"] == 0.8
    assert response.headers["X-Request-ID"]
    predictor.predict_one.assert_called_once_with({"amount": 8})


def test_batch_prediction(fake_predictor):
    predictor, _ = fake_predictor
    with TestClient(app) as client:
        response = client.post("/predict/batch", json=[{"features": {"amount": 8}}])
    assert response.status_code == 200
    assert len(response.json()) == 1
    predictor.predict_batch.assert_called_once_with([{"amount": 8}])


@pytest.mark.parametrize(
    "payload", [{}, {"features": {}}, {"features": {"amount": []}}]
)
def test_invalid_payload_has_clean_422(fake_predictor, payload):
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert "detail" in response.json()
    assert "input" not in str(response.json())


def test_health_unavailable_when_model_fails_to_load(monkeypatch):
    monkeypatch.setattr(
        Predictor, "load", Mock(side_effect=RuntimeError("unavailable"))
    )
    with TestClient(app) as client:
        assert client.get("/health").status_code == 503
        assert (
            client.post("/predict", json={"features": {"amount": 8}}).status_code == 503
        )
