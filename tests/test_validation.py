import pandas as pd

from src.validation import validate_dataset


def base_df():
    return pd.DataFrame(
        {
            "month": ["2026-07", "2026-08"],
            "district": ["A", "A"],
            "anc_coverage": [80, 90],
            "institutional_delivery": [85, 86],
            "immunization": [88, 89],
            "high_risk_cases": [10, 12],
        }
    )


def test_valid_csv_with_all_required_columns():
    result = validate_dataset(base_df())
    assert result.is_valid
    assert len(result.analysis) == 2
    assert result.analysis.loc[0, "month"] == "2026-07"


def test_missing_required_column():
    df = base_df().drop(columns=["immunization"])
    result = validate_dataset(df)
    assert not result.is_valid
    assert "immunization" in result.missing_columns


def test_invalid_date_month_is_reported_and_excluded():
    df = base_df()
    df.loc[0, "month"] = "not-a-month"
    result = validate_dataset(df)
    assert result.invalid_month_rows == [2]
    assert result.rows_dropped == 1
    assert len(result.analysis) == 1


def test_invalid_numeric_value_becomes_missing():
    df = base_df()
    df.loc[0, "anc_coverage"] = "bad"
    result = validate_dataset(df)
    assert result.invalid_numeric_counts["anc_coverage"] == 1
    assert result.analysis["anc_coverage"].isna().sum() == 1


def test_missing_values_are_counted():
    df = base_df()
    df.loc[0, "immunization"] = None
    result = validate_dataset(df)
    assert result.missing_counts["immunization"] == 1


def test_duplicate_district_month_rows_reported_and_excluded():
    df = base_df()
    df.loc[1, "month"] = "2026-07"
    result = validate_dataset(df)
    assert result.duplicate_count == 2
    assert result.rows_dropped == 2
    assert result.analysis.empty
