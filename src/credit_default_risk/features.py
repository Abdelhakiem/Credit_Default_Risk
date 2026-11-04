"""Build shared feature-engineered train/test tables from the interim splits.

Run with ``python -m credit_default_risk.features`` from an installed environment.
Model-specific categorical preparation belongs in the modeling stage.
"""

from pathlib import Path

import pandas as pd

from .config import (
    ID_COL,
    INTERIM_DIR,
    PROCESSED_DIR,
    TARGET_COL,
    TEST_FILE,
    TRAIN_FILE,
)
from .helper import add_application_ratios, add_time_features, select_property_suffix


def engineer_features(
    df: pd.DataFrame, property_suffix: str | None = "MEDI"
) -> pd.DataFrame:
    """Add application features while retaining the target and application ID."""
    result = add_time_features(df)
    result = add_application_ratios(result)
    if property_suffix is not None:
        result = select_property_suffix(result, suffix=property_suffix)
    return result


def build_feature_splits(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    property_suffix: str | None = "MEDI",
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
    return train_features, test_features


def main(
    input_dir: Path = INTERIM_DIR,
    output_dir: Path = PROCESSED_DIR,
    property_suffix: str | None = "MEDI",
) -> tuple[Path, Path]:
    """Read interim splits, engineer shared features, and save Parquet splits."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    train_df = pd.read_parquet(input_dir / TRAIN_FILE)
    test_df = pd.read_parquet(input_dir / TEST_FILE)
    train_features, test_features = build_feature_splits(
        train_df, test_df, property_suffix
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / TRAIN_FILE
    test_path = output_dir / TEST_FILE
    train_features.to_parquet(train_path, index=False)
    test_features.to_parquet(test_path, index=False)
    print(f"Train: {train_features.shape} -> {train_path}")
    print(f"Test: {test_features.shape} -> {test_path}")
    return train_path, test_path


if __name__ == "__main__":
    main()
