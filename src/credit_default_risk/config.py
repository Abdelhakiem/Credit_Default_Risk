"""Paths and settings for the application data preparation workflow."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

ID_COL = "SK_ID_CURR"
TARGET_COL = "TARGET"
TEST_SIZE = 0.20
RANDOM_STATE = 42

RAW_FILES = {
    "applications": "applications.csv",
    "bureau": "bureau.csv",
    "previous_application": "previous_application.csv",
    "credit_card_balance": "credit_card_balance.csv",
    "installments_payments": "installments_payments.csv",
}

TRAIN_FILE = "train.parquet"
TEST_FILE = "test.parquet"

# The history tables are large. Read only columns used by the current features.
HISTORY_COLUMNS = {
    "bureau": [ID_COL, "SK_ID_BUREAU", "CREDIT_ACTIVE", "CREDIT_DAY_OVERDUE"],
    "previous_application": [
        ID_COL, "SK_ID_PREV", "NAME_CONTRACT_STATUS", "AMT_APPLICATION", "AMT_CREDIT"
    ],
    "credit_card_balance": [ID_COL, "AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL"],
    "installments_payments": [
        ID_COL, "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT", "AMT_INSTALMENT", "AMT_PAYMENT"
    ],
}

ZERO_WHEN_NO_HISTORY = [
    "BUREAU_NUM_CREDITS",
    "BUREAU_ACTIVE_CREDIT_COUNT",
    "PREV_APPLICATION_COUNT",
    "PREV_APPROVED_COUNT",
    "PREV_APPROVED_REQUESTED_AMT_SUM",
    "PREV_APPROVED_GRANTED_AMT_SUM",
]
