import pandas as pd

from src.analytics import detect_correlations, detect_outliers, detect_trends


def trend_df():
    return pd.DataFrame(
        {
            "month": ["2026-08", "2026-07", "2026-07", "2026-08"],
            "district": ["A", "A", "B", "B"],
            "anc_coverage": [120, 100, 50, 45],
            "institutional_delivery": [1, 1, 1, 1],
            "immunization": [1, 1, 1, 1],
            "high_risk_cases": [1, 1, 1, 1],
        }
    )


def test_positive_change_above_threshold():
    trends, _ = detect_trends(trend_df(), ["anc_coverage"], threshold=10)
    row = trends[(trends["entity"] == "A")].iloc[0]
    assert row["change"] == 20
    assert row["severity"] == "High"


def test_negative_change_above_threshold():
    trends, _ = detect_trends(trend_df(), ["anc_coverage"], threshold=10)
    row = trends[(trends["entity"] == "B")].iloc[0]
    assert row["change"] == -5
    assert row["severity"] == "Medium"


def test_change_below_threshold_not_flagged():
    df = trend_df()
    df.loc[df["district"] == "B", "anc_coverage"] = [50, 48]
    trends, _ = detect_trends(df, ["anc_coverage"], threshold=10)
    assert "B" not in trends.get("entity", pd.Series(dtype=str)).tolist()


def test_previous_value_zero_does_not_divide_by_zero():
    df = pd.DataFrame({"month": ["2026-07", "2026-08"], "district": ["A", "A"], "anc_coverage": [0, 10]})
    trends, warnings = detect_trends(df, ["anc_coverage"], threshold=10)
    assert trends.empty
    assert any("previous available value was zero" in warning for warning in warnings)


def test_districts_are_not_mixed_and_months_sort_chronologically():
    trends, _ = detect_trends(trend_df(), ["anc_coverage"], threshold=10)
    assert set(trends["entity"]) == {"A", "B"}
    assert trends.loc[trends["entity"] == "A", "prev_value"].iloc[0] == 100


def outlier_df():
    return pd.DataFrame(
        {
            "month": [f"2026-{month:02d}" for month in range(1, 9)],
            "district": list("ABCDEFGH"),
            "anc_coverage": [1, 10, 11, 12, 13, 14, 15, 30],
        }
    )


def test_known_low_and_high_outliers():
    outliers, _ = detect_outliers(outlier_df(), ["anc_coverage"], multiplier=1.0)
    assert set(outliers["value"]) == {1.0, 30.0}


def test_missing_values_ignored_in_outliers():
    df = outlier_df()
    df.loc[0, "anc_coverage"] = None
    outliers, _ = detect_outliers(df, ["anc_coverage"], multiplier=1.0)
    assert 30.0 in outliers["value"].tolist()


def test_constant_indicator_iqr_zero_handled():
    df = outlier_df()
    df["anc_coverage"] = 5
    outliers, warnings = detect_outliers(df, ["anc_coverage"], multiplier=1.5)
    assert outliers.empty
    assert any("IQR 0" in warning for warning in warnings)


def test_multiplier_changes_outlier_detection():
    df = outlier_df()
    strict, _ = detect_outliers(df, ["anc_coverage"], multiplier=1.0)
    loose, _ = detect_outliers(df, ["anc_coverage"], multiplier=3.0)
    assert len(strict) > len(loose)


def corr_df():
    return pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": [2, 4, 6, 8, 10],
            "c": [10, 8, 6, 4, 2],
            "d": [1, 1, 2, 1, 2],
        }
    )


def test_strong_positive_and_negative_correlation():
    rows, matrix, _ = detect_correlations(corr_df(), ["a", "b", "c"], threshold=0.9)
    pairs = {(row.indicator, row.compared_indicator) for row in rows.itertuples()}
    assert ("a", "b") in pairs
    assert ("a", "c") in pairs
    assert matrix.loc["a", "b"] == 1


def test_weak_correlation_not_flagged_and_no_duplicate_pairs():
    rows, _, _ = detect_correlations(corr_df(), ["a", "b", "d"], threshold=0.9)
    pairs = [(row.indicator, row.compared_indicator) for row in rows.itertuples()]
    assert ("a", "d") not in pairs
    assert len(pairs) == len(set(tuple(sorted(pair)) for pair in pairs))


def test_missing_values_handled_for_correlation():
    df = corr_df()
    df.loc[0, "b"] = None
    rows, _, _ = detect_correlations(df, ["a", "b"], threshold=0.9)
    assert len(rows) == 1


def test_insufficient_sample_size_handled():
    df = corr_df().head(2)
    rows, _, warnings = detect_correlations(df, ["a", "b"], threshold=0.9)
    assert rows.empty
    assert any("paired observations" in warning for warning in warnings)
