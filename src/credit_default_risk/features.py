"""Build shared feature-engineered train/test tables from the interim splits.

Run with ``python -m credit_default_risk.features`` from an installed environment.
Model-specific categorical preparation is available for individual frames.
"""

import logging
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from .config import (
    ID_COL,
    INTERIM_DIR,
    PROCESSED_DIR,
    PROPERTY_SUFFIX,
    TARGET_COL,
    TEST_FILE,
    TRAIN_FILE,
)
from .helper import (
    add_application_ratios,
    add_time_features,
    select_final_features,
    select_property_suffix,
)
from .logging_conf import configure_logging

logger = logging.getLogger(__name__)
REQUIRED_SOURCE_COLUMNS = frozenset(
    {
        "DAYS_EMPLOYED",
        "DAYS_BIRTH",
        "AMT_CREDIT",
        "AMT_INCOME_TOTAL",
        "AMT_ANNUITY",
        "AMT_GOODS_PRICE",
        "CNT_FAM_MEMBERS",
    }
)


def engineer_features(
    df: pd.DataFrame, property_suffix: str | None = PROPERTY_SUFFIX
) -> pd.DataFrame:
    """Add application features while retaining the target and application ID."""
    missing = sorted(REQUIRED_SOURCE_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"Feature engineering requires columns: {missing}")
    if property_suffix not in {"AVG", "MEDI", "MODE", None}:
        raise ValueError(f"Unsupported property suffix: {property_suffix}")
    result = add_time_features(df)
    result = add_application_ratios(result)
    if property_suffix is not None:
        result = select_property_suffix(result, suffix=property_suffix)
    return result


def prepare_catboost_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return CatBoost predictors with categorical values as strings."""
    result = df.drop(columns=[ID_COL, TARGET_COL], errors="ignore").copy()
    categorical_cols = result.select_dtypes(
        include=["object", "string", "category"]
    ).columns

    for col in categorical_cols:
        result[col] = result[col].astype("string").fillna("__MISSING__").astype(object)

    # CatBoost accepts NaN in numeric columns, including former nullable integers.
    for col in result.select_dtypes(include="number").columns:
        if isinstance(result[col].dtype, pd.api.extensions.ExtensionDtype):
            result[col] = result[col].astype("float64")

    return result


def prepare_final_catboost_frame(
    df: pd.DataFrame, manifest: Mapping[str, object]
) -> pd.DataFrame:
    """Select and prepare the exact features recorded with a trained model."""
    columns = manifest.get("feature_columns")
    categorical = manifest.get("categorical_columns")
    if (
        not isinstance(columns, list)
        or not columns
        or not all(isinstance(column, str) for column in columns)
        or len(columns) != len(set(columns))
    ):
        raise ValueError("Manifest must contain unique feature_columns")
    if (
        not isinstance(categorical, list)
        or not all(isinstance(column, str) for column in categorical)
        or not set(categorical) <= set(columns)
    ):
        raise ValueError("Manifest has invalid categorical_columns")

    selected = select_final_features(df, selected_columns=columns)
    for column in categorical:
        selected[column] = (
            selected[column].astype("string").fillna("__MISSING__").astype(object)
        )
    for column in set(columns) - set(categorical):
        selected[column] = pd.to_numeric(selected[column], errors="raise").astype(
            "float64"
        )
    prepared = prepare_catboost_frame(selected)
    actual_categorical = set(
        prepared.select_dtypes(include=["object", "string", "category"]).columns
    )
    if actual_categorical != set(categorical):
        raise ValueError("Prepared categorical columns differ from model manifest")
    return prepared


def prepare_lightgbm_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return LightGBM predictors with pandas categorical columns."""
    result = df.drop(columns=[ID_COL, TARGET_COL], errors="ignore").copy()
    categorical_cols = result.select_dtypes(
        include=["object", "string", "category"]
    ).columns

    for col in categorical_cols:
        result[col] = result[col].astype("category")
        if result[col].cat.ordered:
            result[col] = result[col].cat.as_unordered()

    return result


def build_feature_splits(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    property_suffix: str | None = PROPERTY_SUFFIX,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Transform both splits identically and check their feature schema."""
    for name, frame in (("train", train_df), ("test", test_df)):
        missing = {ID_COL, TARGET_COL} - set(frame.columns)
        if missing:
            raise ValueError(f"{name} is missing columns: {sorted(missing)}")
        if frame[ID_COL].isna().any() or frame[ID_COL].duplicated().any():
            raise ValueError(f"{name} must have unique, non-null {ID_COL} values")

    if train_df[ID_COL].isin(test_df[ID_COL]).any():
        raise ValueError("Train and test contain overlapping application IDs")

    train_features = engineer_features(train_df, property_suffix)
    test_features = engineer_features(test_df, property_suffix)

    if train_features.columns.tolist() != test_features.columns.tolist():
        raise ValueError("Train and test feature columns differ")
    mismatched = [
        column
        for column in train_features.columns
        if train_features[column].dtype != test_features[column].dtype
    ]
    if mismatched:
        raise ValueError(f"Train and test feature dtypes differ: {mismatched}")
    return train_features, test_features


def main(
    input_dir: Path = INTERIM_DIR,
    output_dir: Path = PROCESSED_DIR,
    property_suffix: str | None = PROPERTY_SUFFIX,
) -> tuple[Path, Path]:
    """Read interim splits, engineer shared features, and save Parquet splits."""
    configure_logging()
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    train_df = pd.read_parquet(input_dir / TRAIN_FILE)
    test_df = pd.read_parquet(input_dir / TEST_FILE)
    train_features, test_features = build_feature_splits(
        train_df, test_df, property_suffix
    )
    logger.info("Feature engineering: property_suffix=%s", property_suffix)

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / TRAIN_FILE
    test_path = output_dir / TEST_FILE
    train_features.to_parquet(train_path, index=False)
    test_features.to_parquet(test_path, index=False)
    logger.info(
        "Train features saved: rows=%d columns=%d path=%s",
        *train_features.shape,
        train_path,
    )
    logger.info(
        "Test features saved: rows=%d columns=%d path=%s",
        *test_features.shape,
        test_path,
    )
    return train_path, test_path


if __name__ == "__main__":
    main()
