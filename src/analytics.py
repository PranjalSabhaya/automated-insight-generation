from __future__ import annotations

import numpy as np
import pandas as pd

from .validation import INDICATORS


def humanize_indicator(indicator: str) -> str:
    return indicator.replace("_", " ").title()


def trend_severity(change_pct: float, threshold: float) -> str:
    change = abs(change_pct)

    if change >= 2 * threshold:
        return "High"
    elif change >= threshold:
        return "Medium"
    return "Low"


def outlier_severity(value: float, lower_bound: float, upper_bound: float, iqr: float) -> str:
    if value < lower_bound:
        distance = lower_bound - value
    else:
        distance = value - upper_bound

    ratio = distance / iqr

    if ratio >= 1:
        return "High"
    elif ratio >= 0.5:
        return "Medium"
    return "Low"


def correlation_severity(correlation: float) -> str:
    value = abs(correlation)

    if value >= 0.90:
        return "High"
    elif value >= 0.70:
        return "Medium"
    return "Low"


def detect_trends(df: pd.DataFrame, indicators: list[str], threshold: float = 10.0) -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict] = []
    warnings: list[str] = []

    if df.empty:
        return _frame(rows), ["No rows are available for trend detection."]

    for district, district_df in df.groupby("district", dropna=True):
        district_df = district_df.sort_values("month")
        for indicator in indicators:
            if indicator not in district_df.columns:
                continue
            values = district_df[["month", indicator]].dropna(subset=[indicator]).reset_index(drop=True)
            if len(values) < 2:
                continue
            for idx in range(1, len(values)):
                previous = float(values.loc[idx - 1, indicator])
                current = float(values.loc[idx, indicator])
                change = current - previous
                if previous == 0:
                    warnings.append(
                        f"Skipped percentage trend flag for {district} {humanize_indicator(indicator)} in "
                        f"{values.loc[idx, 'month']} because the previous available value was zero."
                    )
                    continue
                change_pct = (change / abs(previous)) * 100
                if abs(change_pct) >= threshold:
                    direction = "increased" if change > 0 else "decreased"
                    rows.append(
                        {
                            "type": "trend",
                            "indicator": indicator,
                            "entity": district,
                            "period": str(values.loc[idx, "month"]),
                            "metric": current,
                            "value": current,
                            "prev_value": previous,
                            "change": change,
                            "change_pct": change_pct,
                            "lower_bound": np.nan,
                            "upper_bound": np.nan,
                            "correlation": np.nan,
                            "compared_indicator": np.nan,
                            "severity": trend_severity(change_pct, threshold),
                            "explanation": (
                                f"{humanize_indicator(indicator)} in {district} {direction} by "
                                f"{abs(change_pct):.1f}% ({previous:g} to {current:g}) compared with the "
                                "previous available month."
                            ),
                        }
                    )

    if not rows:
        warnings.append("No month-over-month trends met the configured threshold.")
    return _frame(rows), warnings


def detect_outliers(df: pd.DataFrame, indicators: list[str], multiplier: float = 1.5) -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict] = []
    warnings: list[str] = []

    for indicator in indicators:
        if indicator not in df.columns:
            continue
        values = df[indicator].dropna()
        if len(values) < 4:
            warnings.append(f"{humanize_indicator(indicator)} has fewer than 4 values, so IQR outlier detection was skipped.")
            continue
        q1 = float(values.quantile(0.25))
        q3 = float(values.quantile(0.75))
        iqr = q3 - q1
        if iqr == 0:
            warnings.append(f"{humanize_indicator(indicator)} has IQR 0, so outlier detection was skipped.")
            continue
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr
        for _, row in df.loc[df[indicator].notna()].iterrows():
            value = float(row[indicator])
            if value < lower_bound or value > upper_bound:
                rows.append(
                    {
                        "type": "outlier",
                        "indicator": indicator,
                        "entity": row["district"],
                        "period": str(row["month"]),
                        "metric": value,
                        "value": value,
                        "prev_value": np.nan,
                        "change": np.nan,
                        "change_pct": np.nan,
                        "lower_bound": lower_bound,
                        "upper_bound": upper_bound,
                        "correlation": np.nan,
                        "compared_indicator": np.nan,
                        "severity": outlier_severity(value, lower_bound, upper_bound, iqr),
                        "explanation": (
                            f"{humanize_indicator(indicator)} in {row['district']} was {value:g}, outside the "
                            f"expected IQR range of {lower_bound:.2f} to {upper_bound:.2f}; review this observation."
                        ),
                    }
                )

    if not rows:
        warnings.append("No IQR outliers were found for the selected indicators.")
    return _frame(rows), warnings


def calculate_correlation_matrix(df: pd.DataFrame, indicators: list[str], min_periods: int = 3) -> pd.DataFrame:
    available = [indicator for indicator in indicators if indicator in df.columns]
    if not available:
        return pd.DataFrame()
    return df[available].corr(method="pearson", min_periods=min_periods)


def detect_correlations(
    df: pd.DataFrame,
    indicators: list[str],
    threshold: float = 0.70,
    min_periods: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    matrix = calculate_correlation_matrix(df, indicators, min_periods=min_periods)
    rows: list[dict] = []
    warnings: list[str] = []

    if matrix.empty or len(matrix.columns) < 2:
        return _frame(rows), matrix, ["Select at least two indicators to calculate correlations."]

    for i, first in enumerate(matrix.columns):
        for second in matrix.columns[i + 1 :]:
            paired_count = int(df[[first, second]].dropna().shape[0])
            if paired_count < min_periods:
                warnings.append(
                    f"{humanize_indicator(first)} and {humanize_indicator(second)} have only {paired_count} paired observations; correlation was skipped."
                )
                continue
            correlation = matrix.loc[first, second]
            if pd.notna(correlation) and abs(float(correlation)) >= threshold:
                rows.append(
                    {
                        "type": "correlation",
                        "indicator": first,
                        "entity": "All selected districts",
                        "period": "All available months",
                        "metric": float(correlation),
                        "value": np.nan,
                        "prev_value": np.nan,
                        "change": np.nan,
                        "change_pct": np.nan,
                        "lower_bound": np.nan,
                        "upper_bound": np.nan,
                        "correlation": float(correlation),
                        "compared_indicator": second,
                        "severity": correlation_severity(float(correlation)),
                        "explanation": (
                            f"{humanize_indicator(first)} and {humanize_indicator(second)} have a Pearson correlation "
                            f"of {correlation:.2f} in the analyzed data. Correlation does not establish causation."
                        ),
                    }
                )

    if not rows:
        warnings.append("No indicator pairs met the configured correlation threshold.")
    return _frame(rows), matrix, warnings


def _frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)
