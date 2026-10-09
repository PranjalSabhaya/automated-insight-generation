import pandas as pd

from src.insight_generator import INSIGHT_COLUMNS, generate_insights


def test_insight_ids_unique_and_sequential():
    df = pd.DataFrame(
        {
            "month": ["2026-07", "2026-08", "2026-07", "2026-08"],
            "district": ["A", "A", "B", "B"],
            "anc_coverage": [100, 130, 80, 60],
            "institutional_delivery": [1, 2, 3, 4],
        }
    )
    insights, _, _ = generate_insights(df, ["anc_coverage", "institutional_delivery"], trend_threshold=10)
    assert insights["insight_id"].is_unique
    assert insights["insight_id"].tolist() == [f"INS-{idx:04d}" for idx in range(1, len(insights) + 1)]


def test_required_fields_exist_and_severity_values_are_valid():
    df = pd.DataFrame({"month": ["2026-07", "2026-08"], "district": ["A", "A"], "anc_coverage": [100, 130]})
    insights, _, _ = generate_insights(df, ["anc_coverage"], trend_threshold=10)
    assert list(insights.columns) == INSIGHT_COLUMNS
    assert set(insights["severity"]).issubset({"Low", "Medium", "High"})


def test_explanation_is_dynamic_from_computed_values():
    df = pd.DataFrame({"month": ["2026-07", "2026-08"], "district": ["North", "North"], "anc_coverage": [50, 75]})
    insights, _, _ = generate_insights(df, ["anc_coverage"], trend_threshold=10)
    explanation = insights.iloc[0]["explanation"]
    assert "North" in explanation
    assert "50.0%" in explanation
    assert "50 to 75" in explanation


def test_no_district_specific_narrative_is_hardcoded():
    df = pd.DataFrame({"month": ["2026-07", "2026-08"], "district": ["CustomTown", "CustomTown"], "anc_coverage": [10, 20]})
    insights, _, _ = generate_insights(df, ["anc_coverage"], trend_threshold=10)
    assert "CustomTown" in insights.iloc[0]["explanation"]
    assert "Ahmedabad" not in insights.iloc[0]["explanation"]
