import numpy as np
import pandas as pd
import pytest

from credit_default_risk.features import (
    build_feature_splits,
    engineer_features,
    prepare_final_catboost_frame,
)


@pytest.fixture
def application():
    return pd.DataFrame(
        {
            "SK_ID_CURR": [1],
            "TARGET": [0],
            "DAYS_EMPLOYED": [365243],
            "DAYS_BIRTH": [-3652.5],
            "AMT_CREDIT": [100.0],
            "AMT_INCOME_TOTAL": [50.0],
            "AMT_ANNUITY": [10.0],
            "AMT_GOODS_PRICE": [80.0],
            "CNT_FAM_MEMBERS": [2.0],
            "APARTMENTS_AVG": [0.1],
            "APARTMENTS_MEDI": [0.2],
            "APARTMENTS_MODE": [0.3],
        }
    )


@pytest.mark.parametrize(
    "missing", ["DAYS_BIRTH", "DAYS_EMPLOYED", "AMT_CREDIT", "CNT_FAM_MEMBERS"]
)
def test_engineering_rejects_missing_source(application, missing):
    with pytest.raises(ValueError, match=missing):
        engineer_features(application.drop(columns=missing))


@pytest.mark.parametrize(
    "suffix,kept",
    [
        ("MEDI", "APARTMENTS_MEDI"),
        ("AVG", "APARTMENTS_AVG"),
        ("MODE", "APARTMENTS_MODE"),
    ],
)
def test_property_suffix_selects_one_family(application, suffix, kept):
    result = engineer_features(application, property_suffix=suffix)
    assert [name for name in result if name.startswith("APARTMENTS_")] == [kept]
    assert {"SK_ID_CURR", "TARGET"} <= set(result)


@pytest.mark.parametrize("income,expected", [(50.0, 2.0), (0.0, None)])
def test_ratio_handles_zero_denominator(application, income, expected):
    application["AMT_INCOME_TOTAL"] = income
    result = engineer_features(application)
    value = result.loc[0, "CREDIT_TO_INCOME_RATIO"]
    assert np.isnan(value) if expected is None else value == expected
    assert np.isnan(result.loc[0, "EMPLOYMENT_YEARS"])
    assert result.loc[0, "AGE_YEARS"] == 10.0


def test_final_frame_uses_manifest_order_and_missing_category():
    source = pd.DataFrame(
        {"SK_ID_CURR": [1], "TARGET": [0], "amount": [pd.NA], "kind": [None]}
    )
    manifest = {"feature_columns": ["kind", "amount"], "categorical_columns": ["kind"]}
    result = prepare_final_catboost_frame(source, manifest)
    assert list(result.columns) == ["kind", "amount"]
    assert result.loc[0, "kind"] == "__MISSING__"
    assert pd.isna(result.loc[0, "amount"])


def test_feature_splits_reject_overlapping_ids(application):
    with pytest.raises(ValueError, match="overlapping"):
        build_feature_splits(application, application.copy())
