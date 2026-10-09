from dataclasses import dataclass

import pandas as pd


REQUIRED_COLUMNS = [
    "month",
    "district",
    "anc_coverage",
    "institutional_delivery",
    "immunization",
    "high_risk_cases",
]

INDICATORS = [
    "anc_coverage",
    "institutional_delivery",
    "immunization",
    "high_risk_cases",
]


@dataclass
class ValidationResult:
    raw: pd.DataFrame
    normalized: pd.DataFrame
    analysis: pd.DataFrame
    missing_columns: list[str]
    invalid_month_rows: list[int]
    invalid_district_rows: list[int]
    invalid_numeric_counts: dict[str, int]
    missing_counts: dict[str, int]
    duplicate_count: int
    duplicate_rows: pd.DataFrame
    rows_dropped: int
    excluded_rows: pd.DataFrame
    info: pd.DataFrame
    errors: list[str]
    warnings: list[str]

    @property
    def is_valid(self) -> bool:
        return not self.errors


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized.columns = [str(col).strip().lower() for col in normalized.columns]
    return normalized


def validate_dataset(raw_df: pd.DataFrame) -> ValidationResult:
    """Normalize and validate uploaded healthcare data.

    Invalid month/district rows and duplicate district-month rows are excluded from
    the returned analysis frame, but are retained in the validation report.
    """
    raw = raw_df.copy()
    df = normalize_columns(raw_df)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    errors: list[str] = []
    warnings: list[str] = []

    if missing_columns:
        errors.append("Missing required columns: " + ", ".join(missing_columns))
        return ValidationResult(
            raw=raw,
            normalized=df,
            analysis=pd.DataFrame(columns=REQUIRED_COLUMNS),
            missing_columns=missing_columns,
            invalid_month_rows=[],
            invalid_district_rows=[],
            invalid_numeric_counts={indicator: 0 for indicator in INDICATORS},
            missing_counts={column: 0 for column in REQUIRED_COLUMNS},
            duplicate_count=0,
            duplicate_rows=pd.DataFrame(columns=REQUIRED_COLUMNS),
            rows_dropped=0,
            excluded_rows=pd.DataFrame(columns=list(df.columns) + ["exclusion_reason"]),
            info=_build_info(df),
            errors=errors,
            warnings=warnings,
        )

    df = df[REQUIRED_COLUMNS].copy()
    df["_source_row"] = df.index + 2

    original_month_missing = df["month"].isna()
    month_text = df["month"].astype("string").str.strip()
    parsed_month = pd.to_datetime(month_text, format="%Y-%m", errors="coerce")
    remaining = parsed_month.isna() & month_text.notna()
    if remaining.any():
        parsed_month.loc[remaining] = pd.to_datetime(month_text.loc[remaining], format="%Y-%m-%d", errors="coerce")
    invalid_month_mask = parsed_month.isna()
    df["month"] = parsed_month.dt.to_period("M").astype("string")
    df.loc[invalid_month_mask, "month"] = pd.NA

    df["district"] = df["district"].astype("string").str.strip()
    invalid_district_mask = df["district"].isna() | (df["district"] == "")

    invalid_numeric_counts: dict[str, int] = {}
    for indicator in INDICATORS:
        before = df[indicator]
        converted = pd.to_numeric(before, errors="coerce")
        invalid_numeric = before.notna() & converted.isna()
        invalid_numeric_counts[indicator] = int(invalid_numeric.sum())
        df[indicator] = converted

    missing_counts = {column: int(df[column].isna().sum()) for column in REQUIRED_COLUMNS}

    duplicate_mask = df.duplicated(subset=["district", "month"], keep=False) & df["district"].notna() & df["month"].notna()
    duplicate_rows = df.loc[duplicate_mask, REQUIRED_COLUMNS + ["_source_row"]].copy()
    duplicate_count = int(duplicate_mask.sum())

    exclusion_reasons: list[str] = []
    for idx, row in df.iterrows():
        reasons: list[str] = []
        if invalid_month_mask.loc[idx]:
            if original_month_missing.loc[idx]:
                reasons.append("missing month")
            else:
                reasons.append("invalid month")
        if invalid_district_mask.loc[idx]:
            reasons.append("missing district")
        if duplicate_mask.loc[idx]:
            reasons.append("duplicate district-month")
        exclusion_reasons.append("; ".join(reasons))

    df["_exclusion_reason"] = exclusion_reasons
    analysis = df.loc[df["_exclusion_reason"] == "", REQUIRED_COLUMNS + ["_source_row"]].copy()
    analysis = analysis.sort_values(["district", "month", "_source_row"]).reset_index(drop=True)

    excluded_rows = df.loc[df["_exclusion_reason"] != "", REQUIRED_COLUMNS + ["_source_row", "_exclusion_reason"]].copy()
    excluded_rows = excluded_rows.rename(columns={"_exclusion_reason": "exclusion_reason"})

    rows_dropped = int(len(excluded_rows))
    if rows_dropped:
        warnings.append(f"{rows_dropped} row(s) were excluded from analysis. See the excluded rows table for details.")
    if duplicate_count:
        warnings.append(f"{duplicate_count} duplicate district-month row(s) were found and excluded from analysis.")
    invalid_numeric_total = sum(invalid_numeric_counts.values())
    if invalid_numeric_total:
        warnings.append(f"{invalid_numeric_total} invalid numeric value(s) were converted to missing values.")

    return ValidationResult(
        raw=raw,
        normalized=df.drop(columns=["_exclusion_reason"]),
        analysis=analysis,
        missing_columns=missing_columns,
        invalid_month_rows=df.loc[invalid_month_mask, "_source_row"].astype(int).tolist(),
        invalid_district_rows=df.loc[invalid_district_mask, "_source_row"].astype(int).tolist(),
        invalid_numeric_counts=invalid_numeric_counts,
        missing_counts=missing_counts,
        duplicate_count=duplicate_count,
        duplicate_rows=duplicate_rows,
        rows_dropped=rows_dropped,
        excluded_rows=excluded_rows,
        info=_build_info(df[REQUIRED_COLUMNS]),
        errors=errors,
        warnings=warnings,
    )


def _build_info(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "column": list(df.columns),
            "non_null_count": [int(df[column].notna().sum()) for column in df.columns],
            "dtype": [str(df[column].dtype) for column in df.columns],
            "missing_count": [int(df[column].isna().sum()) for column in df.columns],
        }
    )
