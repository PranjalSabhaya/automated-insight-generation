import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_loader import load_csv, load_sample_data
from src.insight_generator import generate_insights
from src.validation import INDICATORS, REQUIRED_COLUMNS, validate_dataset


st.set_page_config(page_title="Automated Insight Generation", page_icon="AI", layout="wide")


THEMES = {
    "Dark": {
        "background": "#0f172a",
        "secondary_background": "#111827",
        "card": "#1f2937",
        "text": "#e5e7eb",
        "muted": "#9ca3af",
        "border": "#374151",
        "accent": "#38bdf8",
        "plotly_template": "plotly_dark",
    },
    "Light": {
        "background": "#f8fafc",
        "secondary_background": "#ffffff",
        "card": "#ffffff",
        "text": "#111827",
        "muted": "#4b5563",
        "border": "#d1d5db",
        "accent": "#2563eb",
        "plotly_template": "plotly_white",
    },
}


def apply_theme(theme_name: str) -> dict:
    theme = THEMES[theme_name]
    st.markdown(
        f"""
        <style>
            :root {{
                --app-background: {theme["background"]};
                --app-secondary-background: {theme["secondary_background"]};
                --app-card: {theme["card"]};
                --app-text: {theme["text"]};
                --app-muted: {theme["muted"]};
                --app-border: {theme["border"]};
                --app-accent: {theme["accent"]};
            }}

            .stApp {{
                background-color: var(--app-background);
                color: var(--app-text);
            }}

            [data-testid="stSidebar"],
            [data-testid="stSidebarContent"] {{
                background-color: var(--app-secondary-background);
            }}

            h1, h2, h3, h4, h5, h6,
            p, label, span, div,
            [data-testid="stMarkdownContainer"],
            [data-testid="stMetricValue"],
            [data-testid="stMetricLabel"] {{
                color: var(--app-text);
            }}

            [data-testid="stCaptionContainer"],
            .stCaptionContainer {{
                color: var(--app-muted);
            }}

            [data-testid="stMetric"],
            [data-testid="stExpander"],
            div[data-testid="stDataFrame"],
            div[data-testid="stPlotlyChart"] {{
                background-color: var(--app-card);
                border: 1px solid var(--app-border);
                border-radius: 8px;
            }}

            .stButton > button,
            .stDownloadButton > button {{
                background-color: var(--app-accent);
                border-color: var(--app-accent);
                color: #ffffff;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    px.defaults.template = theme["plotly_template"]
    return theme


def format_indicator(indicator: str) -> str:
    return indicator.replace("_", " ").title()


if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "Dark"

with st.sidebar:
    st.header("Appearance")
    light_mode = st.toggle("Light mode", value=st.session_state.theme_mode == "Light")
    st.session_state.theme_mode = "Light" if light_mode else "Dark"

apply_theme(st.session_state.theme_mode)

st.title("Automated Insight Generation")
st.caption(
    "A deterministic analytics dashboard for district-level monthly healthcare CSVs. "
    "It detects month-over-month trends, IQR outliers, and Pearson correlations without LLM/API calls."
)

with st.sidebar:
    st.header("Data Upload")
    uploaded_file = st.file_uploader("Upload healthcare CSV", type=["csv"])
    with st.expander("Required schema", expanded=False):
        st.write(pd.DataFrame({"column": REQUIRED_COLUMNS}))

    st.header("Analysis Controls")
    trend_threshold = st.slider("Trend threshold (%)", min_value=1, max_value=100, value=10, step=1)
    iqr_multiplier = st.slider("IQR multiplier", min_value=1.0, max_value=3.0, value=1.5, step=0.25)
    correlation_threshold = st.slider(
        "Minimum absolute correlation", min_value=0.10, max_value=1.00, value=0.70, step=0.05
    )


try:
    raw_df = load_csv(uploaded_file) if uploaded_file is not None else load_sample_data()
except Exception as exc:
    st.error(f"Could not read the CSV: {exc}")
    st.stop()

validation = validate_dataset(raw_df)

if validation.errors:
    st.error(validation.errors[0])
    with st.expander("Detected columns", expanded=True):
        st.dataframe(pd.DataFrame({"column": list(validation.normalized.columns)}), use_container_width=True)
    st.stop()

for warning in validation.warnings:
    st.warning(warning)

df = validation.analysis.copy()
if df.empty:
    st.error("No valid rows are available for analysis after validation.")
    st.stop()

with st.sidebar:
    st.header("Live Filters")
    all_districts = sorted(df["district"].dropna().unique().tolist())
    all_months = sorted(df["month"].dropna().unique().tolist())
    selected_districts = st.multiselect("District", all_districts, default=all_districts)
    selected_months = st.multiselect("Month", all_months, default=all_months)
    selected_indicators = st.multiselect(
        "Indicator",
        INDICATORS,
        default=INDICATORS,
        format_func=format_indicator,
    )

if not selected_indicators:
    st.warning("Select at least one indicator to run the analysis.")
    st.stop()

filtered = df[
    df["district"].isin(selected_districts) &
    df["month"].isin(selected_months)
].copy()

with st.expander("Data Validation", expanded=True):
    st.subheader("First five rows")
    st.dataframe(validation.normalized[REQUIRED_COLUMNS].head(), use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Column names and types")
        st.dataframe(validation.info, use_container_width=True, hide_index=True)
    with right:
        st.subheader("Missing values")
        missing_df = pd.DataFrame(
            {"column": list(validation.missing_counts.keys()), "missing_count": list(validation.missing_counts.values())}
        )
        st.dataframe(missing_df, use_container_width=True, hide_index=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Duplicate district-month rows", validation.duplicate_count)
    c2.metric("Rows excluded from analysis", validation.rows_dropped)
    c3.metric("Invalid numeric values", sum(validation.invalid_numeric_counts.values()))

    if not validation.duplicate_rows.empty:
        st.subheader("Duplicate rows")
        st.dataframe(validation.duplicate_rows, use_container_width=True, hide_index=True)
    if not validation.excluded_rows.empty:
        st.subheader("Excluded rows")
        st.dataframe(validation.excluded_rows, use_container_width=True, hide_index=True)

if filtered.empty:
    st.warning("The current filters leave no rows to analyze.")
    st.stop()

insights, correlation_matrix, analysis_warnings = generate_insights(
    filtered,
    selected_indicators,
    trend_threshold=trend_threshold,
    iqr_multiplier=iqr_multiplier,
    correlation_threshold=correlation_threshold,
)

if len(filtered) < 4:
    st.warning("The filtered dataset is very small. Outliers and correlations may be skipped or unstable.")
for warning in analysis_warnings:
    st.info(warning)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Rows analyzed", f"{len(filtered):,}")
kpi2.metric("Districts", filtered["district"].nunique())
kpi3.metric("Total insights", len(insights))
kpi4.metric("High severity", int((insights["severity"] == "High").sum()) if not insights.empty else 0)

st.subheader("Generated Insights")
all_types = ["trend", "outlier", "correlation", "threshold_breach"]
severity_filter = st.multiselect("Filter by severity", ["Low", "Medium", "High"], default=["Low", "Medium", "High"])
type_filter = st.multiselect("Filter by insight type", all_types, default=all_types)

shown_insights = insights[insights["severity"].isin(severity_filter) & insights["type"].isin(type_filter)].copy()
st.dataframe(shown_insights, use_container_width=True, hide_index=True)
st.download_button(
    "Download insights.csv",
    data=shown_insights.to_csv(index=False).encode("utf-8"),
    file_name="insights.csv",
    mime="text/csv",
)

left, right = st.columns(2)
with left:
    st.subheader("Insights by Severity")
    severity_counts = (
        insights["severity"].value_counts().reindex(["Low", "Medium", "High"], fill_value=0).rename_axis("severity")
    )
    severity_df = severity_counts.reset_index(name="count")
    fig = px.bar(
        severity_df,
        x="severity",
        y="count",
        category_orders={"severity": ["Low", "Medium", "High"]},
        labels={"severity": "Severity", "count": "Insight count"},
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Pearson Correlation Heatmap")
    st.caption("Correlation does not establish causation; small samples can produce unstable coefficients.")
    if correlation_matrix.empty or correlation_matrix.dropna(how="all").empty:
        st.info("Not enough numerical data to calculate correlations.")
    else:
        fig = px.imshow(
            correlation_matrix,
            text_auto=".2f",
            zmin=-1,
            zmax=1,
            color_continuous_scale="RdBu_r",
            aspect="auto",
        )
        st.plotly_chart(fig, use_container_width=True)

st.subheader("Correlation Matrix")
st.dataframe(correlation_matrix, use_container_width=True)
st.download_button(
    "Download correlation_matrix.csv",
    data=correlation_matrix.to_csv().encode("utf-8"),
    file_name="correlation_matrix.csv",
    mime="text/csv",
)

st.subheader("District-Level Indicator Trend")
line_indicator = st.selectbox("Indicator for line chart", selected_indicators, format_func=format_indicator)
line_data = filtered.dropna(subset=[line_indicator]).sort_values("month")
if line_data.empty:
    st.info("No non-missing values are available for the selected line chart indicator.")
else:
    fig = px.line(
        line_data,
        x="month",
        y=line_indicator,
        color="district",
        markers=True,
        labels={
            "month": "Month",
            line_indicator: format_indicator(line_indicator),
            "district": "District",
        },
    )
    st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Limitation: the bundled sample has only two months and six districts. Treat correlation findings as exploratory, "
    "not statistically significant or medically causal."
)
