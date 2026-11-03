"""Prepare application data and save stratified train/test Parquet splits.

Run with ``python -m credit_default_risk.data`` from an installed environment.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (
    HISTORY_COLUMNS,
    ID_COL,
    INTERIM_DIR,
    RANDOM_STATE,
    RAW_DIR,
    RAW_FILES,
    TARGET_COL,
    TEST_SIZE,
    TRAIN_FILE,
    TEST_FILE
)
from .helper import (
    add_document_flag_sum,
    aggregate_bureau,
    aggregate_credit_cards,
    aggregate_installments,
    aggregate_previous_applications,
    convert_application_dtypes,
    fill_missing_history_counts,
    join_history_features,
)


def read_raw_table(name: str, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Read a raw CSV, selecting only needed columns for history tables."""
    path = Path(raw_dir) / RAW_FILES[name]
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(
        path,
        usecols=HISTORY_COLUMNS.get(name),
        low_memory=False,
    )


def prepare_applications(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Clean applications and join four historical aggregate feature tables."""
    applications = read_raw_table("applications", raw_dir)
    if applications[ID_COL].isna().any() or applications[ID_COL].duplicated().any():
        raise ValueError(f"{ID_COL} must be present and unique in applications.")
    if applications[TARGET_COL].isna().any():
        raise ValueError(f"{TARGET_COL} contains unlabeled application rows.")

    prepared = add_document_flag_sum(convert_application_dtypes(applications))
    del applications

    aggregators = (
        ("bureau", aggregate_bureau),
        ("previous_application", aggregate_previous_applications),
        ("credit_card_balance", aggregate_credit_cards),
        ("installments_payments", aggregate_installments),
    )
    for name, aggregate in aggregators:
        raw_history = read_raw_table(name, raw_dir)
        features = aggregate(raw_history)
        del raw_history
        prepared = join_history_features(prepared, features)
        del features

    return fill_missing_history_counts(prepared)


def split_train_test(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Make the notebook's stratified 80/20 split on TARGET."""
    return train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[TARGET_COL],
    )


def save_train_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path = INTERIM_DIR,
) -> tuple[Path, Path]:
    """Save both prepared splits as Parquet, without DataFrame indexes."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / TRAIN_FILE
    test_path = output_dir / TEST_FILE
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)
    return train_path, test_path


def main() -> None:
    prepared = prepare_applications()
    train_df, test_df = split_train_test(prepared)
    train_path, test_path = save_train_test(train_df, test_df)
    print(f"Train: {train_df.shape} -> {train_path}")
    print(f"Test: {test_df.shape} -> {test_path}")


if __name__ == "__main__":
    main()
