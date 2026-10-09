import pandas as pd

from .analytics import detect_correlations, detect_outliers, detect_trends


INSIGHT_COLUMNS = [
    "insight_id",
    "type",
    "indicator",
    "entity",
    "period",
    "metric",
    "change",
    "severity",
    "explanation",
    "value",
    "prev_value",
    "change_pct",
    "lower_bound",
    "upper_bound",
    "correlation",
    "compared_indicator",
]


def generate_insights(
    df: pd.DataFrame,
    indicators: list[str],
    trend_threshold: float = 10.0,
    iqr_multiplier: float = 1.5,
    correlation_threshold: float = 0.70,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    trend_rows, trend_warnings = detect_trends(df, indicators, trend_threshold)
    outlier_rows, outlier_warnings = detect_outliers(df, indicators, iqr_multiplier)
    correlation_rows, correlation_matrix, correlation_warnings = detect_correlations(
        df, indicators, correlation_threshold
    )

    insight_frames = [frame for frame in [trend_rows, outlier_rows, correlation_rows] if not frame.empty]
    if insight_frames:
        insights = pd.concat(insight_frames, ignore_index=True)
    else:
        insights = pd.DataFrame(columns=[column for column in INSIGHT_COLUMNS if column != "insight_id"])

    for column in INSIGHT_COLUMNS:
        if column == "insight_id":
            continue
        if column not in insights.columns:
            insights[column] = pd.NA

    insights = insights[[column for column in INSIGHT_COLUMNS if column != "insight_id"]]
    insights.insert(0, "insight_id", [f"INS-{idx:04d}" for idx in range(1, len(insights) + 1)])

    warnings = trend_warnings + outlier_warnings + correlation_warnings
    return insights[INSIGHT_COLUMNS], correlation_matrix, warnings
