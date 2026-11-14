"""HTTP request and response shapes for model inference."""

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """One record containing the model's engineered features."""

    features: dict[str, str | float | int | None] = Field(min_length=1)


class PredictionResponse(BaseModel):
    """Default probability and optional review decision."""

    default_probability: float = Field(ge=0, le=1)
    review_flag: bool | None
    decision_threshold: float | None
    model_version: str | None
