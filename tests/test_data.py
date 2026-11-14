import pandas as pd
import pytest

from credit_default_risk import data
from credit_default_risk.config import ID_COL, TARGET_COL


def test_prepare_applications_joins_history_and_fills_absent_counts(tmp_path):
    applications = pd.DataFrame(
        {
            ID_COL: [1, 2],
            TARGET_COL: [0, 1],
            "FLAG_OWN_REALTY": ["Y", "N"],
            "FLAG_OWN_CAR": ["N", "Y"],
            "EMERGENCYSTATE_MODE": ["NO", "YES"],
            "CODE_GENDER": ["F", "XNA"],
            "FLAG_DOCUMENT_2": [1, 0],
            "FLAG_DOCUMENT_3": [0, 1],
        }
    )
    tables = {
        "applications": applications,
        "bureau": pd.DataFrame(
            {
                ID_COL: [1],
                "SK_ID_BUREAU": [10],
                "CREDIT_ACTIVE": ["Active"],
                "CREDIT_DAY_OVERDUE": [0],
            }
        ),
        "previous_application": pd.DataFrame(
            {
                ID_COL: [1],
                "SK_ID_PREV": [20],
                "NAME_CONTRACT_STATUS": ["Approved"],
                "AMT_APPLICATION": [80],
                "AMT_CREDIT": [100],
            }
        ),
        "credit_card_balance": pd.DataFrame(
            {ID_COL: [1], "AMT_BALANCE": [20], "AMT_CREDIT_LIMIT_ACTUAL": [100]}
        ),
        "installments_payments": pd.DataFrame(
            {
                ID_COL: [1],
                "DAYS_INSTALMENT": [-10],
                "DAYS_ENTRY_PAYMENT": [-9],
                "AMT_INSTALMENT": [10],
                "AMT_PAYMENT": [10],
            }
        ),
    }
    for name, frame in tables.items():
        frame.to_csv(tmp_path / data.RAW_FILES[name], index=False)
    result = data.prepare_applications(tmp_path)
    assert len(result) == 2
    assert result.loc[0, "BUREAU_NUM_CREDITS"] == 1
    assert result.loc[1, "BUREAU_NUM_CREDITS"] == 0
    assert result.loc[0, "INSTALLMENT_LATE_PAYMENT_RATE"] == 1
    assert result.loc[0, "NUM_DOCUMENTS"] == 1
    assert pd.isna(result.loc[1, "CODE_GENDER"])


def test_split_is_reproducible_disjoint_and_saved(tmp_path):
    frame = pd.DataFrame({ID_COL: range(20), TARGET_COL: [0, 1] * 10})
    train, test = data.split_train_test(frame, test_size=0.2, random_state=42)
    train_again, test_again = data.split_train_test(
        frame, test_size=0.2, random_state=42
    )
    assert train[ID_COL].tolist() == train_again[ID_COL].tolist()
    assert test[ID_COL].tolist() == test_again[ID_COL].tolist()
    assert not set(train[ID_COL]) & set(test[ID_COL])
    paths = data.save_train_test(train, test, tmp_path)
    assert [len(pd.read_parquet(path)) for path in paths] == [16, 4]


def test_read_raw_table_reports_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.read_raw_table("applications", tmp_path)
