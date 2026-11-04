"""Application cleaning and one-row-per-application history features."""

import re

import numpy as np
import pandas as pd

from .config import (
    FINAL_SELECTED_COLUMNS,
    ID_COL,
    TARGET_COL,
    ZERO_WHEN_NO_HISTORY,
)


def select_final_features(
    df: pd.DataFrame,
    selected_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Return the configured model features in their saved column order.

    Extra input columns such as SK_ID_CURR and TARGET are excluded. Raise an
    error if any selected feature is missing from the input frame.
    """
    columns = FINAL_SELECTED_COLUMNS if selected_columns is None else selected_columns
    if not columns:
        raise ValueError("selected_columns must contain at least one feature")
    if ID_COL in columns or TARGET_COL in columns:
        raise ValueError(f"Selected features must not include {ID_COL} or {TARGET_COL}")

    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Input frame is missing selected features: {missing}")

    return df.loc[:, columns].copy()


def safe_divide(numerator, denominator):
    denominator = denominator.where(denominator > 0)
    return numerator.div(denominator).replace([np.inf, -np.inf], np.nan)

def add_time_features(df):
    df = df.copy()

    employed_days = df["DAYS_EMPLOYED"].replace(365243, np.nan)
    df["DAYS_EMPLOYED"] = employed_days

    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365.25
    df["EMPLOYMENT_YEARS"] = -employed_days / 365.25

    df["AGE_BIN"] = pd.cut(
        df["AGE_YEARS"],
        bins=[0, 25, 35, 45, 55, 65, np.inf],
        labels=["<25", "25-34", "35-44", "45-54", "55-64", "65+"],
        right=False,
    )

    df["EMPLOYMENT_BIN"] = pd.cut(
        df["EMPLOYMENT_YEARS"],
        bins=[0, 1, 3, 5, 10, 20, np.inf],
        labels=["<1", "1-2", "3-4", "5-9", "10-19", "20+"],
        right=False,
    )

    return df
def add_application_ratios(df):
    df = df.copy()

    df["CREDIT_TO_INCOME_RATIO"] = safe_divide(
        df["AMT_CREDIT"], df["AMT_INCOME_TOTAL"]
    )
    df["ANNUITY_TO_INCOME_RATIO"] = safe_divide(
        df["AMT_ANNUITY"], df["AMT_INCOME_TOTAL"]
    )
    df["GOODS_TO_CREDIT_RATIO"] = safe_divide(
        df["AMT_GOODS_PRICE"], df["AMT_CREDIT"]
    )
    df["CREDIT_TO_ANNUITY_RATIO"] = safe_divide(
        df["AMT_CREDIT"], df["AMT_ANNUITY"]
    )
    df["INCOME_PER_FAMILY_MEMBER"] = safe_divide(
        df["AMT_INCOME_TOTAL"], df["CNT_FAM_MEMBERS"]
    )

    return df
def select_property_suffix(df, suffix="MEDI"):
    df = df.copy()
    property_suffixes = ("AVG", "MEDI", "MODE")

    families = {
        col.rsplit("_", 1)[0]
        for col in df.columns
        if col.endswith(tuple(f"_{s}" for s in property_suffixes))
        and all(
            f"{col.rsplit('_', 1)[0]}_{s}" in df.columns
            for s in property_suffixes
        )
    }

    drop_cols = [
        f"{family}_{s}"
        for family in families
        for s in property_suffixes
        if s != suffix
    ]

    return df.drop(columns=drop_cols)

def _map_binary(series: pd.Series, mapping: dict[str, int], column: str) -> pd.Series:
    values = series.astype("string").str.strip().str.upper()
    unexpected = values.dropna().loc[~values.dropna().isin(mapping)].unique()
    if len(unexpected):
        raise ValueError(f"{column} has unexpected values: {unexpected.tolist()}")
    return values.map(mapping).astype("Int8")


def convert_application_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Convert ownership/emergency flags and replace invalid gender with null."""
    result = df.copy()
    for column in ("FLAG_OWN_REALTY", "FLAG_OWN_CAR"):
        result[column] = _map_binary(result[column], {"Y": 1, "N": 0}, column)

    column = "EMERGENCYSTATE_MODE"
    result[column] = _map_binary(result[column], {"YES": 1, "NO": 0}, column)

    gender = result["CODE_GENDER"].astype("string").str.strip().str.upper()
    result["CODE_GENDER"] = gender.mask(gender.eq("XNA"), pd.NA)
    return result


def add_document_flag_sum(
    df: pd.DataFrame, new_column: str = "NUM_DOCUMENTS"
) -> pd.DataFrame:
    """Count document flags per row and remove the original flag columns."""
    document_columns = [
        column for column in df.columns
        if re.fullmatch(r"FLAG_DOCUMENT_\d+", column)
    ]
    if not document_columns:
        raise ValueError("No FLAG_DOCUMENT_<number> columns found.")

    values = df[document_columns].to_numpy(
        dtype=np.float32, na_value=np.nan, copy=False
    )
    if np.isnan(values).any():
        sums = np.nansum(values, axis=1)
        sums[np.isnan(values).all(axis=1)] = np.nan
    else:
        sums = values.sum(axis=1)

    result = df.drop(columns=document_columns).copy()
    result[new_column] = sums
    return result


def aggregate_bureau(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize external bureau credits by current application."""
    return (
        df.groupby(ID_COL)
        .agg(
            BUREAU_NUM_CREDITS=("SK_ID_BUREAU", "nunique"),
            BUREAU_ACTIVE_CREDIT_COUNT=(
                "CREDIT_ACTIVE", lambda values: values.eq("Active").sum()
            ),
            BUREAU_MAX_DAYS_OVERDUE=("CREDIT_DAY_OVERDUE", "max"),
        )
        .reset_index()
    )


def aggregate_previous_applications(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize previous Home Credit applications by current application."""
    approved = df["NAME_CONTRACT_STATUS"].eq("Approved")
    inputs = pd.DataFrame({
        ID_COL: df[ID_COL],
        "SK_ID_PREV": df["SK_ID_PREV"],
        "IS_APPROVED": approved.astype("int8"),
        "APPROVED_REQUESTED_SUM": df["AMT_APPLICATION"].where(approved, 0),
        "APPROVED_GRANTED_SUM": df["AMT_CREDIT"].where(approved, 0),
        "APPROVED_REQUESTED_MEAN": df["AMT_APPLICATION"].where(approved),
        "APPROVED_GRANTED_MEAN": df["AMT_CREDIT"].where(approved),
        "APPROVED_REQUESTED_TO_GRANTED_RATIO": (
            df["AMT_APPLICATION"]
            .div(df["AMT_CREDIT"].where(df["AMT_CREDIT"].gt(0)))
            .where(approved)
        ),
    })
    features = (
        inputs.groupby(ID_COL)
        .agg(
            PREV_APPLICATION_COUNT=("SK_ID_PREV", "nunique"),
            PREV_APPROVED_COUNT=("IS_APPROVED", "sum"),
            PREV_APPROVED_REQUESTED_AMT_SUM=("APPROVED_REQUESTED_SUM", "sum"),
            PREV_APPROVED_GRANTED_AMT_SUM=("APPROVED_GRANTED_SUM", "sum"),
            PREV_APPROVED_REQUESTED_AMT_MEAN=("APPROVED_REQUESTED_MEAN", "mean"),
            PREV_APPROVED_GRANTED_AMT_MEAN=("APPROVED_GRANTED_MEAN", "mean"),
            PREV_APPROVED_REQUESTED_TO_GRANTED_RATIO_MEAN=(
                "APPROVED_REQUESTED_TO_GRANTED_RATIO", "mean"
            ),
        )
        .reset_index()
    )
    features["PREV_APPROVAL_RATE"] = (
        features["PREV_APPROVED_COUNT"] / features["PREV_APPLICATION_COUNT"]
    )
    return features


def aggregate_credit_cards(df: pd.DataFrame) -> pd.DataFrame:
    """Average monthly card balance and utilization by application."""
    positive_limit = df["AMT_CREDIT_LIMIT_ACTUAL"].where(
        df["AMT_CREDIT_LIMIT_ACTUAL"].gt(0)
    )
    inputs = pd.DataFrame({
        ID_COL: df[ID_COL],
        "UTILIZATION": df["AMT_BALANCE"].div(positive_limit),
        "BALANCE": df["AMT_BALANCE"],
    })
    return (
        inputs.groupby(ID_COL)
        .agg(
            CREDIT_CARD_AVG_UTILIZATION=("UTILIZATION", "mean"),
            CREDIT_CARD_AVG_BALANCE=("BALANCE", "mean"),
        )
        .reset_index()
    )


def aggregate_installments(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize late installments and paid-to-due amounts by application."""
    ids = df[ID_COL]
    has_due_date = df["DAYS_INSTALMENT"].notna()
    late_payment = has_due_date & (
        df["DAYS_ENTRY_PAYMENT"].isna()
        | df["DAYS_ENTRY_PAYMENT"].gt(df["DAYS_INSTALMENT"])
    )
    late_count = late_payment.groupby(ids).sum()
    due_count = has_due_date.groupby(ids).sum()
    late_rate = late_count.div(due_count.where(due_count.gt(0)))

    paid_total = df["AMT_PAYMENT"].fillna(0).groupby(ids).sum()
    installment_total = df["AMT_INSTALMENT"].groupby(ids).sum()
    payment_ratio = paid_total.div(installment_total.where(installment_total.ne(0)))

    return pd.concat(
        [
            late_rate.rename("INSTALLMENT_LATE_PAYMENT_RATE"),
            payment_ratio.rename("INSTALLMENT_PAYMENT_TO_DUE_RATIO"),
        ],
        axis=1,
    ).rename_axis(ID_COL).reset_index()


def join_history_features(
    applications: pd.DataFrame, features: pd.DataFrame
) -> pd.DataFrame:
    """Left-join a unique aggregate table without changing application count."""
    if features[ID_COL].duplicated().any():
        raise ValueError("Aggregate features contain duplicate SK_ID_CURR values.")
    result = applications.merge(features, on=ID_COL, how="left", validate="one_to_one")
    if len(result) != len(applications):
        raise RuntimeError("A history join changed the application row count.")
    return result


def fill_missing_history_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Represent absent history as zero for counts and approved amount totals."""
    result = df.copy()
    result[ZERO_WHEN_NO_HISTORY] = result[ZERO_WHEN_NO_HISTORY].fillna(0)
    return result
