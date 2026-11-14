"""Paths and settings for the application data preparation workflow."""

from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings read from environment variables at construction."""

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    project_root: Path = Path(__file__).resolve().parents[2]
    raw_dir: Path | None = None
    interim_dir: Path | None = None
    processed_dir: Path | None = None
    artifact_dir: Path | None = None
    test_size: float = 0.15
    random_state: int = 42
    property_suffix: str = "MEDI"
    winning_run_id: str = "ce08758483ff4f21bc1499ecfca12115"
    model_name: str = "credit-default-risk-catboost"
    model_alias: str = "champion"
    mlflow_tracking_uri: str = "http://127.0.0.1:5000"
    log_level: str = "INFO"

    @model_validator(mode="after")
    def resolve_paths(self) -> "Settings":
        self.project_root = self.project_root.expanduser().resolve()
        for field, relative in (
            ("raw_dir", "data/raw"),
            ("interim_dir", "data/interim"),
            ("processed_dir", "data/processed"),
            ("artifact_dir", "artifacts/catboost_final"),
        ):
            path = getattr(self, field) or self.project_root / relative
            setattr(self, field, path.expanduser().resolve())
        return self


settings = Settings()
PROJECT_ROOT = settings.project_root
RAW_DIR = settings.raw_dir
INTERIM_DIR = settings.interim_dir

ID_COL = "SK_ID_CURR"
TARGET_COL = "TARGET"
TEST_SIZE = settings.test_size
RANDOM_STATE = settings.random_state
PROPERTY_SUFFIX = settings.property_suffix


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
        ID_COL,
        "SK_ID_PREV",
        "NAME_CONTRACT_STATUS",
        "AMT_APPLICATION",
        "AMT_CREDIT",
    ],
    "credit_card_balance": [ID_COL, "AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL"],
    "installments_payments": [
        ID_COL,
        "DAYS_INSTALMENT",
        "DAYS_ENTRY_PAYMENT",
        "AMT_INSTALMENT",
        "AMT_PAYMENT",
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

PROCESSED_DIR = settings.processed_dir
PROCESSED_TRAIN_PATH = PROCESSED_DIR / TRAIN_FILE
PROCESSED_TEST_PATH = PROCESSED_DIR / TEST_FILE

# CatBoost-selected feature set from notebooks/04_feature_selection.ipynb.
# This reflects the evaluated reduced model before the remaining correlated-pair review.
FINAL_SELECTED_COLUMNS = [
    "NAME_CONTRACT_TYPE",
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "CNT_CHILDREN",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "NAME_TYPE_SUITE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "REGION_POPULATION_RELATIVE",
    "DAYS_REGISTRATION",
    "DAYS_ID_PUBLISH",
    "OWN_CAR_AGE",
    "FLAG_WORK_PHONE",
    "FLAG_CONT_MOBILE",
    "FLAG_PHONE",
    "FLAG_EMAIL",
    "OCCUPATION_TYPE",
    "CNT_FAM_MEMBERS",
    "REGION_RATING_CLIENT_W_CITY",
    "REG_REGION_NOT_WORK_REGION",
    "REG_CITY_NOT_LIVE_CITY",
    "LIVE_CITY_NOT_WORK_CITY",
    "ORGANIZATION_TYPE",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "APARTMENTS_MEDI",
    "BASEMENTAREA_MEDI",
    "COMMONAREA_MEDI",
    "FLOORSMAX_MEDI",
    "FLOORSMIN_MEDI",
    "NONLIVINGAPARTMENTS_MEDI",
    "DEF_30_CNT_SOCIAL_CIRCLE",
    "OBS_60_CNT_SOCIAL_CIRCLE",
    "AMT_REQ_CREDIT_BUREAU_HOUR",
    "AMT_REQ_CREDIT_BUREAU_WEEK",
    "AMT_REQ_CREDIT_BUREAU_MON",
    "AMT_REQ_CREDIT_BUREAU_QRT",
    "AMT_REQ_CREDIT_BUREAU_YEAR",
    "NUM_DOCUMENTS",
    "BUREAU_NUM_CREDITS",
    "BUREAU_ACTIVE_CREDIT_COUNT",
    "PREV_APPROVED_COUNT",
    "PREV_APPROVED_REQUESTED_AMT_SUM",
    "PREV_APPROVED_GRANTED_AMT_SUM",
    "PREV_APPROVED_REQUESTED_AMT_MEAN",
    "PREV_APPROVED_REQUESTED_TO_GRANTED_RATIO_MEAN",
    "PREV_APPROVAL_RATE",
    "CREDIT_CARD_AVG_UTILIZATION",
    "CREDIT_CARD_AVG_BALANCE",
    "INSTALLMENT_LATE_PAYMENT_RATE",
    "AGE_YEARS",
    "EMPLOYMENT_YEARS",
    "CREDIT_TO_INCOME_RATIO",
    "ANNUITY_TO_INCOME_RATIO",
    "GOODS_TO_CREDIT_RATIO",
    "CREDIT_TO_ANNUITY_RATIO",
]


WINNING_RUN_ID = settings.winning_run_id
MODEL_NAME = settings.model_name
MODEL_ALIAS = settings.model_alias
MLFLOW_TRACKING_URI = settings.mlflow_tracking_uri
ARTIFACT_DIR = settings.artifact_dir
DEFAULT_PARAMS = {
    "iterations": 1000,
    "learning_rate": 0.03,
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "random_strength": 1.0,
    "auto_class_weights": None,
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "random_seed": RANDOM_STATE,
    "verbose": False,
}
