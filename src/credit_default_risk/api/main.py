"""FastAPI app serving the packaged CatBoost model."""

import logging
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from ..config import MODEL_NAME
from ..logging_conf import configure_logging, reset_correlation_id, set_correlation_id
from ..predict import Predictor
from .schemas import PredictionRequest, PredictionResponse

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once per server process at startup."""
    configure_logging()
    app.state.predictor = None
    try:
        app.state.predictor = Predictor.load()
    except Exception:
        logger.exception("Model load failure during API startup")
    yield
    app.state.predictor = None


app = FastAPI(title="Credit Default Risk API", lifespan=lifespan)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = str(uuid4())
    token = set_correlation_id(request_id)
    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        reset_correlation_id(token)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    """Return concise validation errors without echoing feature values."""
    errors = [
        {"location": error["loc"], "message": error["msg"], "type": error["type"]}
        for error in exc.errors()
    ]
    logger.error(
        "Request validation rejected: path=%s errors=%s", request.url.path, errors
    )
    return JSONResponse(status_code=422, content={"detail": errors})


def loaded_predictor(request: Request) -> Predictor:
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return predictor


@app.get("/health")
def health(request: Request):
    loaded_predictor(request)
    return {"status": "ready"}


@app.get("/metadata")
def metadata(request: Request):
    predictor = loaded_predictor(request)
    return {
        "model_name": MODEL_NAME,
        "model_version": predictor.version,
        "feature_count": len(predictor.manifest["feature_columns"]),
        "decision_threshold": predictor.threshold,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest, http_request: Request):
    predictor = loaded_predictor(http_request)
    try:
        return predictor.predict_one(request.features)
    except (TypeError, ValueError) as exc:
        logger.error("Prediction input rejected: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/predict/batch", response_model=list[PredictionResponse])
def predict_batch(requests: list[PredictionRequest], http_request: Request):
    predictor = loaded_predictor(http_request)
    try:
        return predictor.predict_batch([item.features for item in requests])
    except (TypeError, ValueError) as exc:
        logger.error("Batch prediction input rejected: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
