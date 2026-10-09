# Automated Insight Generation

Automated Insight Generation is a Streamlit dashboard that analyzes monthly district-level healthcare CSV data and produces deterministic, data-driven insights. It identifies significant month-over-month trends, IQR outliers, Pearson correlations, and exportable insight tables without using an LLM, external API, or separate backend.

## Features

- CSV upload with the assignment sample data loaded by default
- Required-column validation, normalized column names, date/month parsing, duplicate district-month reporting, and missing-value reporting
- Live filters for district, month, and indicator
- Configurable trend threshold, IQR multiplier, and correlation threshold
- KPI cards for rows analyzed, district count, total insights, and high-severity insights
- Insight table with severity/type filters and CSV download
- Severity bar chart, Pearson correlation heatmap/table, and district-level line chart
- Export downloads for `insights.csv` and `correlation_matrix.csv`
- Unit tests for validation, analytics, and insight generation logic

## Tech Stack

- Python
- Pandas
- NumPy
- Streamlit
- Plotly
- Pytest

## Repository Structure

```text
.
|-- app.py
|-- requirements.txt
|-- README.md
|-- .gitignore
|-- data/
|   `-- sample_healthcare.csv
|-- src/
|   |-- __init__.py
|   |-- analytics.py
|   |-- data_loader.py
|   |-- insight_generator.py
|   `-- validation.py
`-- tests/
    |-- test_analytics.py
    |-- test_insight_generator.py
    `-- test_validation.py
```

## Required CSV Schema

Each row represents one `(district, month)` pair.

| Column | Meaning | Example |
|---|---|---|
| `month` | Reporting month or date | `2026-07` or `2026-07-01` |
| `district` | District name | `Ahmedabad` |
| `anc_coverage` | ANC coverage percentage | `85` |
| `institutional_delivery` | Institutional delivery percentage | `91` |
| `immunization` | Immunization percentage | `93` |
| `high_risk_cases` | High-risk cases raw count | `10` |

Column names are trimmed and lowercased. Months are normalized to `YYYY-MM`. Indicator values are converted to numeric values, and invalid numeric entries become missing values with a warning.

## Setup

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Algorithms

### Trend Detection

For each district and selected indicator, observations are sorted chronologically. Each value is compared with the previous available month for the same district and indicator:

```text
change = current_value - previous_value
change_pct = ((current_value - previous_value) / abs(previous_value)) * 100
```

If the previous value is zero, percentage change is skipped to avoid division by zero. A trend is flagged when `abs(change_pct)` is at least the configured threshold. The default threshold is `10%`.

### IQR Outlier Detection

For each selected indicator, the dashboard computes quartiles across the currently filtered dataset:

```text
Q1 = 25th percentile
Q3 = 75th percentile
IQR = Q3 - Q1
lower_bound = Q1 - multiplier * IQR
upper_bound = Q3 + multiplier * IQR
```

Values below the lower bound or above the upper bound are flagged as outliers. The default multiplier is `1.5`. Missing values are ignored. Indicators with fewer than four values or `IQR == 0` are skipped with a warning.

### Pearson Correlation

The app calculates a Pearson correlation matrix for selected numerical indicators using at least three paired observations. Unique indicator pairs are flagged when `abs(correlation)` is at least the configured threshold. The default threshold is `0.70`. Correlation does not establish causation.

## Severity Rules

- Trend `High`: absolute percentage change is at least twice the selected trend threshold.
- Trend `Medium`: absolute percentage change meets the selected trend threshold but is less than twice the threshold.
- Outlier `High`: distance beyond the relevant IQR bound is at least `1.0 * IQR`.
- Outlier `Medium`: value is outside the IQR bounds, but distance beyond the bound is less than `1.0 * IQR`.
- Correlation `High`: absolute Pearson correlation is at least `0.90`.
- Correlation `Medium`: absolute Pearson correlation meets the selected threshold but is below `0.90`.
- `Low` is reserved for possible informational insight types and is included in charts even when no Low insights are generated.

## Output Files

`insights.csv` contains:

```text
insight_id,type,indicator,entity,period,metric,change,severity,explanation,value,prev_value,change_pct,lower_bound,upper_bound,correlation,compared_indicator
```

`correlation_matrix.csv` contains the square Pandas correlation matrix with selected indicators as both rows and columns.

## Example Output

The bundled sample can flag trends such as large month-over-month ANC changes and observations outside the IQR range. All insight text is generated from computed districts, months, indicators, values, bounds, and correlations.

## Limitations

- The bundled sample has only two months and six districts, so correlations are unstable and should be treated as exploratory.
- The app does not claim medical causation or statistical significance.
- Duplicate district-month rows are excluded from analysis after being reported, because silently aggregating them would hide data quality issues.
- Missing indicator values are allowed, but analyses that lack enough valid data are skipped with a warning.

## Evaluator Demo Flow

1. Run `streamlit run app.py`.
2. Show that the sample dataset loads without an upload.
3. Open Data Validation and show first rows, types, missing values, duplicate reporting, and excluded rows.
4. Change the trend threshold and IQR multiplier to show the insight table updating.
5. Filter by district, month, and indicator.
6. Show the severity bar chart, correlation heatmap/table, and district line chart.
7. Download `insights.csv` and `correlation_matrix.csv`.
8. Upload a CSV with a missing required column to show the clear validation error.
