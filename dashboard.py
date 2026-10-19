import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import time
import os
import shutil

# ============================================================
# PUBLIC HEALTH INTELLIGENCE DASHBOARD
# Put this file beside:
#   year_summary.csv
#   state_summary.csv
#   disease_summary.csv
#   cleaned_health_data.csv (optional, for Data Explorer)
# ============================================================

st.set_page_config(
    page_title="HealthScope | Public Health",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent

# Fill these in with the details from your actual dataset source.
# Do not publish the site with a guessed or unverified source URL.
DATASET_SOURCE_NAME = ""  # Example: official dataset title
DATASET_SOURCE_URL = ""   # Full URL to the original dataset, if publicly available
DATASET_REPORTING_PERIOD = ""  # Example: 2010–2024 (verify against your data)
DATASET_LAST_UPDATED = ""      # Date the source data was last refreshed, if known
DATASET_CASE_DEFINITION = ""    # Explain what a case/count means in this dataset
DATASET_GEOGRAPHIC_COVERAGE = "" # Regions or countries represented in the data
DATASET_LICENSE = ""             # Verified reuse/license terms, if known


@st.cache_resource(show_spinner=False)
def get_spark_session():
    """Create and reuse one Spark session across Streamlit reruns.

    Do not call spark.stop() after each benchmark: Streamlit reruns this file
    after interactions, and stopping a cached session can leave a stale Spark
    context that raises errors such as NoneType.setCallSite.
    """
    from pyspark.sql import SparkSession

    return (
        SparkSession.builder
        .master("local[*]")
        .appName("HealthScopeBenchmark")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )


PAGES = [
    "Overview",
    "Yearly Trends",
    "Regional Analysis",
    "Disease Analysis",
    "Anomaly Review",
    "PySpark Benchmark",
    "Data Explorer",
]

PAGE_COLORS = {
    PAGES[0]: {"accent": "#2DD4BF", "soft": "#103B3B"},
    PAGES[1]: {"accent": "#60A5FA", "soft": "#172D50"},
    PAGES[2]: {"accent": "#34D399", "soft": "#103B2E"},
    PAGES[3]: {"accent": "#C084FC", "soft": "#30204A"},
    PAGES[4]: {"accent": "#FBBF24", "soft": "#443516"},
    PAGES[5]: {"accent": "#38BDF8", "soft": "#15344A"},
    PAGES[6]: {"accent": "#FB923C", "soft": "#492B1D"},
}

# --------------------------- UTILITIES ---------------------------

def find_column(df, candidates):
    """Return the real column name matching one of the candidate names."""
    if df is None or df.empty:
        return None
    lookup = {str(col).strip().lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


@st.cache_data
def read_csv_file(filename):
    """Load a CSV and return both its contents and an actionable status message."""
    path = BASE_DIR / filename
    if not path.exists():
        return pd.DataFrame(), {"file": filename, "status": "Missing", "details": f"File not found at {path}."}
    if path.stat().st_size == 0:
        return pd.DataFrame(), {"file": filename, "status": "Empty", "details": "The file exists but contains no data."}
    try:
        frame = pd.read_csv(path)
        if frame.empty:
            return frame, {"file": filename, "status": "Empty", "details": "The CSV loaded, but it has no data rows."}
        return frame, {"file": filename, "status": "Loaded", "details": f"Loaded {len(frame):,} rows and {len(frame.columns):,} columns."}
    except Exception as exc:
        # Keep the dashboard running, but preserve the reason so users can fix the file.
        return pd.DataFrame(), {"file": filename, "status": "Read error", "details": f"{type(exc).__name__}: {exc}"}


@st.cache_data
def load_all_data():
    yearly, yearly_status = read_csv_file("year_summary.csv")
    states, states_status = read_csv_file("state_summary.csv")
    diseases, diseases_status = read_csv_file("disease_summary.csv")
    raw, raw_status = read_csv_file("cleaned_health_data.csv")
    file_status = [yearly_status, states_status, diseases_status, raw_status]

    # Make yearly summary column names predictable.
    y_year = find_column(yearly, ["year", "Year", "YEAR"])
    y_cases = find_column(
        yearly, ["Total_Cases", "total_cases", "Cases", "case_count", "Total Cases"]
    )
    if y_year and y_cases:
        yearly = yearly.rename(columns={y_year: "year", y_cases: "Total_Cases"})
        yearly["year"] = pd.to_numeric(yearly["year"], errors="coerce")
        yearly["Total_Cases"] = pd.to_numeric(
            yearly["Total_Cases"], errors="coerce"
        )
        yearly = yearly.dropna(subset=["year", "Total_Cases"])
        yearly["year"] = yearly["year"].astype(int)
        yearly = yearly.groupby("year", as_index=False)["Total_Cases"].sum()
        yearly = yearly.sort_values("year")

    # Make state summary column names predictable.
    s_state = find_column(
        states, ["state_ut", "State/UT", "state", "State", "state_name", "State_Name"]
    )
    s_cases = find_column(
        states, ["Total_Cases", "total_cases", "Cases", "case_count", "Total Cases"]
    )
    if s_state and s_cases:
        states = states.rename(
            columns={s_state: "state_ut", s_cases: "Total_Cases"}
        )
        states["Total_Cases"] = pd.to_numeric(
            states["Total_Cases"], errors="coerce"
        )
        states = states.dropna(subset=["state_ut", "Total_Cases"])
        states = states.groupby("state_ut", as_index=False)["Total_Cases"].sum()
        states = states.sort_values("Total_Cases", ascending=False)

    # Make disease summary column names predictable.
    d_disease = find_column(
        diseases, ["Disease", "disease", "disease_name", "Disease_Name", "Disease Name"]
    )
    d_cases = find_column(
        diseases, ["Total_Cases", "total_cases", "Cases", "case_count", "Total Cases"]
    )
    if d_disease and d_cases:
        diseases = diseases.rename(
            columns={d_disease: "Disease", d_cases: "Total_Cases"}
        )
        diseases["Total_Cases"] = pd.to_numeric(
            diseases["Total_Cases"], errors="coerce"
        )
        diseases = diseases.dropna(subset=["Disease", "Total_Cases"])
        diseases = diseases.groupby("Disease", as_index=False).agg(
            {"Total_Cases": "sum"}
        )
        diseases = diseases.sort_values("Total_Cases", ascending=False)

    return yearly, states, diseases, raw, file_status


def show_hero(title, subtitle, accent, eyebrow="PUBLIC HEALTH INTELLIGENCE"):
    st.markdown(
        f"""
        <div class="hero" style="border-left: 6px solid {accent};">
            <div class="eyebrow">{eyebrow}</div>
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_missing_file(filename):
    st.error(f"Required file not found or empty: `{filename}`")
    st.write(
        "Check that the CSV file is in the same folder as `dashboard.py`, "
        "and that it contains data with the expected columns."
    )


def fmt_num(value):
    try:
        return f"{float(value):,.0f}"
    except (ValueError, TypeError):
        return "N/A"


def chart_layout(accent, height=None):
    layout = dict(
        paper_bgcolor="#111C30",
        plot_bgcolor="#111C30",
        font=dict(color="#E2E8F0", family="Arial"),
        margin=dict(l=20, r=25, t=65, b=35),
        title=dict(x=0.02, xanchor="left"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hoverlabel=dict(bgcolor="#0B1220", font_color="#F8FAFC"),
    )
    if height:
        layout["height"] = height
    return layout


def add_page_navigation(current_page):
    """Previous/Next controls for moving between analysis pages."""
    idx = PAGES.index(current_page)
    st.divider()
    left, center, right = st.columns([1.4, 2, 1.4])

    with left:
        if idx > 0:
            st.button(
                "← Previous page",
                key=f"prev_{idx}",
                use_container_width=True,
                on_click=set_page,
                args=(PAGES[idx - 1],),
            )
    with center:
        st.markdown(
            f"<div class='page-counter'>PAGE {idx + 1} OF {len(PAGES)}</div>",
            unsafe_allow_html=True,
        )
        st.progress((idx + 1) / len(PAGES))
    with right:
        if idx < len(PAGES) - 1:
            st.button(
                "Next page →",
                key=f"next_{idx}",
                type="primary",
                use_container_width=True,
                on_click=set_page,
                args=(PAGES[idx + 1],),
            )


def set_page(page_name):
    st.session_state["page_nav"] = page_name


# --------------------------- LOAD DATA ---------------------------

year_df, state_df, disease_df, raw_df, file_status = load_all_data()

# --------------------------- NAVIGATION ---------------------------

if "page_nav" not in st.session_state:
    st.session_state["page_nav"] = PAGES[0]

# Main website header with visible project name.
st.markdown(
    """
    <div class="site-header">
      <div class="site-brand-mark">+</div>
      <div>
        <div class="site-brand">HealthScope <span>PUBLIC HEALTH</span></div>
        <div class="site-tagline">Public Health Outbreak Monitoring &amp; Trend Analysis</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Button navigation: website-like tabs without radio bullets.
nav_cols = st.columns(len(PAGES), gap="small")
for i, page_name in enumerate(PAGES):
    with nav_cols[i]:
        if st.button(
            page_name,
            key=f"nav_button_{i}",
            type="primary" if st.session_state["page_nav"] == page_name else "secondary",
            use_container_width=True,
        ):
            st.session_state["page_nav"] = page_name
            st.rerun()

page = st.session_state["page_nav"]

page_index = PAGES.index(page)
theme = PAGE_COLORS[page]

# --------------------------- GLOBAL STYLE ---------------------------

st.markdown(
    f"""
    <style>
    :root {{
        --accent: {theme["accent"]};
        --accent-soft: {theme["soft"]};
    }}
    .stApp {{
        background: #0B1220;
        color: #F1F5F9;
    }}
    [data-testid="stSidebar"] {{
        display: none;
    }}
    [data-testid="collapsedControl"] {{
        display: none;
    }}
    .block-container {{
        max-width: 1450px;
        padding-top: 4.0rem;
        padding-bottom: 2.5rem;
    }}
    [data-testid="stAppViewContainer"] .main .block-container {{
        padding-top: 4.0rem;
    }}
    .site-header {{
        position: relative;
        overflow: visible;
        min-height: 88px;
        margin-top: 0.25rem;
    }}
    .hero {{
        position: relative;
        overflow: visible;
        margin-top: 1.2rem;
        margin-bottom: 1.8rem;
        padding: 28px 30px;
        border-radius: 18px;
        background: linear-gradient(115deg, #12263B 0%, #123A3A 100%);
        box-shadow: 0 12px 34px rgba(0, 0, 0, 0.16);
    }}
    .hero h1 {{
        color: #F8FAFC;
        font-size: clamp(1.8rem, 3vw, 2.8rem);
        line-height: 1.18;
        margin: 12px 0 14px 0;
        overflow-wrap: anywhere;
    }}
    .hero p {{
        color: #D6E4F2;
        font-size: 1rem;
        line-height: 1.65;
        margin-bottom: 0;
    }}
    .eyebrow {{
        color: var(--accent);
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 2px;
    }}
    .site-header {{
        display: flex;
        align-items: center;
        gap: 14px;
        background: linear-gradient(110deg, #101A2B, #14313A);
        border: 1px solid #243B4A;
        border-radius: 16px;
        padding: 18px 22px;
        margin-bottom: 14px;
    }}
    .site-brand-mark {{
        width: 44px;
        height: 44px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: #2DD4BF;
        color: #07131D;
        font-size: 28px;
        font-weight: 900;
        flex-shrink: 0;
    }}
    .site-brand {{
        color: #F8FAFC;
        font-size: 22px;
        font-weight: 850;
        letter-spacing: 1px;
        line-height: 1.3;
    }}
    .site-brand span {{
        color: #2DD4BF;
        font-size: 11px;
        letter-spacing: 1.5px;
        margin-left: 8px;
    }}
    .site-tagline {{
        color: #AFC1D4;
        font-size: 13px;
        margin-top: 4px;
    }}
    div[data-testid="stButton"] > button {{
        min-height: 44px;
        border-radius: 10px;
        border: 1px solid #2A3A52;
        background: #111C30;
        color: #DCE6F2;
        font-size: 13px;
        font-weight: 650;
        padding: 7px 8px;
        white-space: normal;
        transition: background 0.15s ease, border-color 0.15s ease;
    }}
    div[data-testid="stButton"] > button:hover {{
        border-color: #2DD4BF;
        color: #FFFFFF;
        background: #172B3D;
    }}
    div[data-testid="stButton"] > button[kind="primary"] {{
        background: #2DD4BF !important;
        color: #062A2A !important;
        border-color: #2DD4BF !important;
        font-weight: 800 !important;
    }}
    div[data-testid="stButton"] > button[kind="primary"]:hover {{
        background: #5EEAD4 !important;
        border-color: #5EEAD4 !important;
    }}
    .info-panel {{
        background: #111C30;
        border: 1px solid #26364D;
        border-radius: 13px;
        padding: 18px;
        height: 100%;
    }}
    .info-panel h3 {{
        margin-top: 0;
        color: var(--accent);
        font-size: 18px;
    }}
    .info-panel p, .info-panel li {{
        color: #D2DCE8;
        line-height: 1.65;
    }}
    .page-counter {{
        text-align: center;
        color: #B9C8DA;
        font-weight: 700;
        letter-spacing: 1px;
        padding-top: 5px;
    }}
    .stButton > button {{
        border-radius: 9px;
        min-height: 42px;
        font-weight: 650;
    }}
    h2, h3 {{
        color: #F1F5F9;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# PAGE 1: PROJECT OVERVIEW
# ============================================================

if page == PAGES[0]:
    show_hero(
        "Public Health Outbreak Monitoring",
        "Explore reported disease cases, identify changing trends, compare regions, and review unusual patterns.",
        theme["accent"],
        "HEALTH OVERVIEW",
    )

    total_cases = year_df["Total_Cases"].sum() if not year_df.empty else None
    year_count = year_df["year"].nunique() if not year_df.empty else 0
    region_count = state_df["state_ut"].nunique() if not state_df.empty else 0
    disease_count = disease_df["Disease"].nunique() if not disease_df.empty else 0

    st.markdown("### At a glance")
    k1, k2, k3, k4 = st.columns(4, gap="medium")
    k1.metric("Reported cases in yearly summary", fmt_num(total_cases) if total_cases is not None else "N/A")
    k2.metric("Years covered", f"{year_count:,}")
    k3.metric("Regions covered", f"{region_count:,}")
    k4.metric("Disease categories", f"{disease_count:,}")

    st.markdown("### Health trends and regional patterns")
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("#### Year-wise reported cases")
        if not year_df.empty:
            fig = px.line(
                year_df, x="year", y="Total_Cases", markers=True,
                labels={"year": "Year", "Total_Cases": "Reported cases"},
            )
            fig.update_traces(line=dict(width=3), marker=dict(size=7))
            fig.update_layout(**chart_layout(theme["accent"], 320))
            st.plotly_chart(fig, use_container_width=True, key="overview_year_trend")
        else:
            st.info("Yearly trend will appear when year_summary.csv has valid data.")

    with right:
        st.markdown("#### Regions with highest reported totals")
        if not state_df.empty:
            top_regions = state_df.nlargest(7, "Total_Cases").sort_values("Total_Cases")
            fig = px.bar(
                top_regions, x="Total_Cases", y="state_ut", orientation="h",
                labels={"state_ut": "Region", "Total_Cases": "Reported cases"},
            )
            fig.update_layout(**chart_layout(theme["accent"], 320), showlegend=False)
            st.plotly_chart(fig, use_container_width=True, key="overview_top_regions")
        else:
            st.info("Regional comparison will appear when state_summary.csv has valid data.")

    lower_left, lower_right = st.columns(2, gap="large")
    with lower_left:
        st.markdown("#### Most-reported disease categories")
        if not disease_df.empty:
            top_diseases = disease_df.nlargest(7, "Total_Cases").sort_values("Total_Cases")
            fig = px.bar(
                top_diseases, x="Total_Cases", y="Disease", orientation="h",
                labels={"Disease": "Disease", "Total_Cases": "Reported cases"},
            )
            fig.update_layout(**chart_layout(theme["accent"], 320), showlegend=False)
            st.plotly_chart(fig, use_container_width=True, key="overview_top_diseases")
        else:
            st.info("Disease comparison will appear when disease_summary.csv has valid data.")

    with lower_right:
        st.markdown("#### Key observations")
        if not year_df.empty:
            highest_year = year_df.loc[year_df["Total_Cases"].idxmax()]
            lowest_year = year_df.loc[year_df["Total_Cases"].idxmin()]
            st.markdown(
                f"- **Highest year:** {int(highest_year['year'])} — {fmt_num(highest_year['Total_Cases'])} cases\n"
                f"- **Lowest year:** {int(lowest_year['year'])} — {fmt_num(lowest_year['Total_Cases'])} cases"
            )
            if len(year_df) >= 2:
                first_value = float(year_df.iloc[0]["Total_Cases"])
                last_value = float(year_df.iloc[-1]["Total_Cases"])
                if first_value:
                    change = ((last_value - first_value) / abs(first_value)) * 100
                    direction = "increased" if change > 0 else "decreased" if change < 0 else "remained unchanged"
                    st.write(
                        f"Across {int(year_df.iloc[0]['year'])}–{int(year_df.iloc[-1]['year'])}, "
                        f"reported cases {direction} by {abs(change):,.1f}%."
                    )
        else:
            st.write("Key observations will be generated from the yearly summary.")
        if not state_df.empty:
            st.write(f"**Top region:** {state_df.iloc[0]['state_ut']} ({fmt_num(state_df.iloc[0]['Total_Cases'])} cases)")
        if not disease_df.empty:
            st.write(f"**Top disease category:** {disease_df.iloc[0]['Disease']} ({fmt_num(disease_df.iloc[0]['Total_Cases'])} cases)")

    st.caption(
        "This dashboard summarizes reported records. Unusual values are prompts for further review, "
        "not confirmed outbreaks or medical advice. Reporting coverage and population size may differ."
    )

# ============================================================
# PAGE 2: YEAR-WISE ANALYSIS
# ============================================================

elif page == PAGES[1]:
    show_hero(
        "Year-wise Trend Analysis",
        "Track changes in reported case totals and identify the highest and "
        "lowest years in the selected period.",
        theme["accent"],
        "TIME SERIES ANALYSIS",
    )

    if year_df.empty:
        show_missing_file("year_summary.csv")
    else:
        years = sorted(year_df["year"].unique().tolist())
        selected_range = st.slider(
            "Choose years to analyse",
            min_value=int(min(years)),
            max_value=int(max(years)),
            value=(int(min(years)), int(max(years))),
            step=1,
        )
        data = year_df[
            year_df["year"].between(selected_range[0], selected_range[1])
        ].sort_values("year").copy()

        if data.empty:
            st.warning("No data is available for this year range.")
        else:
            peak = data.loc[data["Total_Cases"].idxmax()]
            low = data.loc[data["Total_Cases"].idxmin()]

            a, b, c = st.columns(3)
            a.metric("Cases in selected period", fmt_num(data["Total_Cases"].sum()))
            b.metric("Highest year", str(int(peak["year"])), f"{fmt_num(peak['Total_Cases'])} cases")
            c.metric("Lowest year", str(int(low["year"])), f"{fmt_num(low['Total_Cases'])} cases")

            data["Year-over-year change (%)"] = (
                data["Total_Cases"].pct_change() * 100
            )

            # Rolling average smooths year-to-year fluctuations for easier visual interpretation.
            data["3-year moving average"] = (
                data["Total_Cases"].rolling(window=3, min_periods=1).mean()
            )

            fig = px.line(
                data,
                x="year",
                y="Total_Cases",
                markers=True,
                title="Annual reported cases with 3-year trend smoothing",
                labels={"year": "Year", "Total_Cases": "Reported cases"},
                hover_data={"year": True, "Total_Cases": ":,.0f"},
            )
            fig.update_traces(
                name="Annual cases",
                line=dict(color=theme["accent"], width=3),
                marker=dict(size=8),
            )
            fig.add_scatter(
                x=data["year"],
                y=data["3-year moving average"],
                mode="lines",
                name="3-year moving average",
                line=dict(color="#FBBF24", width=3, dash="dash"),
                hovertemplate="Year %{x}<br>Moving average %{y:,.0f}<extra></extra>",
            )
            fig.update_layout(**chart_layout(theme["accent"], 450))
            fig.update_xaxes(type="linear", dtick=1, tickformat="d")
            fig.update_yaxes(rangemode="tozero")
            st.plotly_chart(fig, use_container_width=True)

            # Year-over-year change chart highlights growth and decline separately.
            change_data = data[["year", "Total_Cases"]].copy()
            change_data["Change (%)"] = change_data["Total_Cases"].pct_change() * 100
            change_data = change_data.dropna(subset=["Change (%)"])
            if not change_data.empty:
                change_data["Direction"] = change_data["Change (%)"].apply(
                    lambda value: "Increase" if value >= 0 else "Decrease"
                )
                growth_fig = px.bar(
                    change_data,
                    x="year",
                    y="Change (%)",
                    color="Direction",
                    color_discrete_map={"Increase": "#34D399", "Decrease": "#FB7185"},
                    labels={"year": "Year", "Change (%)": "Change (%)"},
                    hover_data={"year": True, "Change (%)": ":.2f"},
                )
                growth_fig.add_hline(y=0, line_color="#94A3B8", line_width=1)
                # Merge layout settings into one dictionary. Passing margin twice
                # (once through chart_layout and once explicitly) raises a TypeError.
                growth_layout = chart_layout(theme["accent"], 360)
                growth_layout.update(
                    margin=dict(l=45, r=25, t=115, b=45),
                    title=dict(
                        text="Year-over-year change (%)",
                        x=0.02,
                        xanchor="left",
                        y=0.98,
                        yanchor="top",
                        font=dict(size=19),
                    ),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.04,
                        x=0.02,
                        xanchor="left",
                    ),
                )
                growth_fig.update_layout(**growth_layout)
                growth_fig.update_xaxes(type="linear", dtick=1, tickformat="d")
                st.plotly_chart(growth_fig, use_container_width=True)

            st.markdown("## How to interpret this chart")
            st.write(
                "Each point represents the total recorded cases for one year. "
                "A rising line means the recorded total increased; a falling "
                "line means it decreased. This alone does not explain why the "
                "change happened, and reporting coverage may differ between years."
            )

            st.markdown("## Year-by-year comparison")
            display = data.rename(
                columns={
                    "year": "Year",
                    "Total_Cases": "Reported cases",
                    "3-year moving average": "3-year moving average (smoothed)",
                }
            )
            st.dataframe(display, use_container_width=True, hide_index=True)

            st.download_button(
                "⬇️ Download selected year data",
                data=data.to_csv(index=False).encode("utf-8"),
                file_name="year_analysis.csv",
                mime="text/csv",
            )


# ============================================================
# PAGE 3: STATE-WISE ANALYSIS
# ============================================================

elif page == PAGES[2]:
    show_hero(
        "State-wise Analysis",
        "Compare state and union territory totals and explore the regional ranking.",
        theme["accent"],
        "REGIONAL COMPARISON",
    )

    if state_df.empty:
        show_missing_file("state_summary.csv")
    else:
        # Offer the full number of available regions as an option, not just a maximum of 20.
        region_count = len(state_df)
        region_options = sorted({n for n in [5, 10, 15, 20] if n < region_count} | {region_count})
        top_n = st.select_slider(
            "Number of regions shown",
            options=region_options,
            value=region_count,
            help=f"Defaults to all {region_count} regions. Choose a smaller number to focus on the highest totals.",
        )
        top = state_df.head(top_n).sort_values("Total_Cases", ascending=True)
        leader = state_df.iloc[0]
        total_state_cases = state_df["Total_Cases"].sum()

        a, b, c = st.columns(3)
        a.metric("Regions in summary", f"{len(state_df):,}")
        b.metric("Highest reported region", str(leader["state_ut"]))
        c.metric("Highest region cases", fmt_num(leader["Total_Cases"]))

        fig = px.bar(
            top,
            x="Total_Cases",
            y="state_ut",
            orientation="h",
            text="Total_Cases",
            title=f"Top {top_n} regions by reported cases",
            labels={"Total_Cases": "Reported cases", "state_ut": "State / UT"},
            hover_data={"Total_Cases": ":,.0f"},
        )
        fig.update_traces(
            marker_color=theme["accent"],
            texttemplate="%{text:,.0f}",
            textposition="outside",
            cliponaxis=False,
        )
        fig.update_layout(**chart_layout(theme["accent"], max(420, top_n * 38)))
        fig.update_xaxes(rangemode="tozero")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("## Regions with increasing reported cases")
        if not raw_df.empty:
            raw_state_col = find_column(raw_df, ["state_ut", "State/UT", "state", "State", "state_name", "State_Name"])
            raw_year_col = find_column(raw_df, ["year", "Year", "YEAR", "reporting_year"])
            raw_case_col = find_column(raw_df, ["Total_Cases", "total_cases", "Cases", "case_count", "confirmed_cases", "Count"])
            if raw_state_col and raw_year_col:
                regional_trend = raw_df.copy()
                regional_trend["_year"] = pd.to_numeric(regional_trend[raw_year_col], errors="coerce")
                if raw_case_col:
                    regional_trend["_cases"] = pd.to_numeric(regional_trend[raw_case_col], errors="coerce").fillna(0)
                    regional_trend = regional_trend.groupby([raw_state_col, "_year"], as_index=False)["_cases"].sum()
                else:
                    regional_trend = regional_trend.groupby([raw_state_col, "_year"]).size().rename("_cases").reset_index()
                regional_trend = regional_trend.dropna(subset=["_year"]).sort_values("_year")
                years_seen = sorted(regional_trend["_year"].unique())
                if len(years_seen) >= 2:
                    first_year, last_year = years_seen[0], years_seen[-1]
                    first = regional_trend[regional_trend["_year"] == first_year][[raw_state_col, "_cases"]].rename(columns={"_cases": "First period"})
                    last = regional_trend[regional_trend["_year"] == last_year][[raw_state_col, "_cases"]].rename(columns={"_cases": "Last period"})
                    changes = first.merge(last, on=raw_state_col, how="outer").fillna(0)
                    changes["Change"] = changes["Last period"] - changes["First period"]
                    changes["Change (%)"] = changes.apply(
                        lambda r: (r["Change"] / r["First period"] * 100) if r["First period"] else None,
                        axis=1,
                    )
                    changes = changes.sort_values("Change", ascending=False)
                    st.caption(
                        f"Comparison: {int(first_year)} to {int(last_year)}. "
                        "This compares the first and last available years, not necessarily adjacent years."
                    )
                    view_mode = st.selectbox(
                        "Regions to display",
                        options=["All states/UTs", "Increasing only"],
                        index=0,
                        key="regional_change_view_mode",
                        help="All states/UTs includes regions with increases, decreases, or no change."
                    )
                    if view_mode == "Increasing only":
                        display_changes = changes[changes["Change"] > 0].copy()
                    else:
                        display_changes = changes.copy()
                    if display_changes.empty:
                        st.info("No regions have an increase in reported cases over the selected comparison period.")
                    else:
                        display_changes = display_changes.rename(columns={raw_state_col: "State / UT"})
                        display_changes["Change (%)"] = display_changes["Change (%)"].round(2)
                        st.dataframe(
                            display_changes,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "First period": st.column_config.NumberColumn(format="%,.0f"),
                                "Last period": st.column_config.NumberColumn(format="%,.0f"),
                                "Change": st.column_config.NumberColumn(format="%,.0f"),
                                "Change (%)": st.column_config.NumberColumn(format="%.2f%%"),
                            },
                        )
                else:
                    st.info("At least two years of state-level records are required to compare increasing regions.")
            else:
                st.info("Increasing-case regions require state/UT and year columns in cleaned_health_data.csv.")
        else:
            st.info("Add cleaned_health_data.csv with state/UT and year columns to calculate increasing-case regions.")

        st.markdown("## Explore one region")
        chosen = st.selectbox(
            "Select state or union territory",
            options=state_df["state_ut"].astype(str).tolist(),
        )
        selected_row = state_df[
            state_df["state_ut"].astype(str) == chosen
        ].iloc[0]
        st.metric(f"Reported cases — {chosen}", fmt_num(selected_row["Total_Cases"]))

        st.markdown("## How to interpret this chart")
        st.write(
            "This ranking compares absolute reported case totals. It does not "
            "measure individual risk: population size, reporting completeness "
            "and the time period should be considered before comparing regions."
        )

        with st.expander("View complete regional table"):
            st.dataframe(state_df, use_container_width=True, hide_index=True)

        st.download_button(
            "⬇️ Download state summary",
            data=state_df.to_csv(index=False).encode("utf-8"),
            file_name="state_analysis.csv",
            mime="text/csv",
        )

# ============================================================
# PAGE 4: DISEASE ANALYSIS
# ============================================================

elif page == PAGES[3]:
    show_hero(
        "Disease-wise Analysis",
        "Compare disease categories using reported case totals and explore individual categories.",
        theme["accent"],
        "DISEASE COMPARISON",
    )

    if disease_df.empty:
        show_missing_file("disease_summary.csv")
    else:
        disease_names = disease_df["Disease"].astype(str)
        search = st.text_input(
            "Search for a disease",
            placeholder="Type a disease name...",
        )
        filtered = disease_df[
            disease_names.str.contains(search, case=False, na=False)
        ].copy()

        if filtered.empty:
            st.warning("No disease category matches that search.")
        else:
            disease_count = len(filtered)
            disease_options = sorted(
                {n for n in [5, 10, 15, 20] if n < disease_count} | {disease_count}
            )
            top_n = st.select_slider(
                "Number of disease categories shown",
                options=disease_options,
                value=disease_count,
                key="disease_top_n",
                help=f"Defaults to all {disease_count} matching disease categories. Search above to narrow the list, or choose a smaller number to focus on the highest totals.",
            )
            top = filtered.head(top_n).sort_values("Total_Cases", ascending=True)
            leader = filtered.iloc[0]

            a, b = st.columns(2)
            a.metric("Categories matching search", f"{len(filtered):,}")
            b.metric(
                "Highest case total in results",
                str(leader["Disease"]),
                f"{fmt_num(leader['Total_Cases'])} cases",
            )

            fig = px.bar(
                top,
                x="Total_Cases",
                y="Disease",
                orientation="h",
                title="Reported cases by disease category",
                labels={"Total_Cases": "Reported cases", "Disease": "Disease"},
                hover_data={"Total_Cases": ":,.0f"},
            )
            fig.update_traces(marker_color=theme["accent"])
            fig.update_layout(**chart_layout(theme["accent"], max(420, top_n * 38)))
            fig.update_xaxes(rangemode="tozero")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("## Explore a disease")
            selected_disease = st.selectbox(
                "Choose disease category",
                options=filtered["Disease"].astype(str).tolist(),
            )
            row = filtered[filtered["Disease"].astype(str) == selected_disease].iloc[0]
            st.metric("Reported cases", fmt_num(row["Total_Cases"]))

            st.markdown("## How to interpret this chart")
            st.write(
                "The chart ranks categories by recorded case totals. Differences "
                "may reflect disease frequency, surveillance coverage, data "
                "collection practices or category definitions. The ranking does "
                "not by itself establish severity or cause."
            )

            with st.expander("View disease summary table"):
                st.dataframe(filtered, use_container_width=True, hide_index=True)

            st.download_button(
                "⬇️ Download disease summary",
                data=filtered.to_csv(index=False).encode("utf-8"),
                file_name="disease_analysis.csv",
                mime="text/csv",
            )

# ============================================================
# PAGE 5: ANOMALY DETECTION
# ============================================================

elif page == PAGES[4]:
    show_hero(
        "Unusual Case Count Review",
        "Use a simple statistical threshold to flag annual totals that may deserve data-quality review.",
        theme["accent"],
        "STATISTICAL SCREENING",
    )

    if year_df.empty or len(year_df) < 4:
        st.warning(
            "At least four valid years are needed for this simple IQR-based "
            "screen. Check year_summary.csv."
        )
    else:
        q1 = year_df["Total_Cases"].quantile(0.25)
        q3 = year_df["Total_Cases"].quantile(0.75)
        iqr = q3 - q1
        threshold = q3 + 1.5 * iqr

        flagged = year_df[year_df["Total_Cases"] > threshold].copy()
        data = year_df.copy()
        data["Review status"] = data["Total_Cases"].apply(
            lambda v: "Review" if v > threshold else "Within threshold"
        )

        a, b, c = st.columns(3)
        a.metric("IQR review threshold", fmt_num(threshold))
        b.metric("Years flagged", f"{len(flagged)}")
        c.metric("Years assessed", f"{len(data)}")

        fig = px.scatter(
            data,
            x="year",
            y="Total_Cases",
            color="Review status",
            color_discrete_map={
                "Review": "#FBBF24",
                "Within threshold": "#38BDF8",
            },
            title="Annual totals and statistical review threshold",
            labels={"year": "Year", "Total_Cases": "Reported cases"},
            hover_data={"Total_Cases": ":,.0f"},
        )
        fig.add_hline(
            y=threshold,
            line_dash="dash",
            line_color="#FBBF24",
            annotation_text="IQR review threshold",
        )
        fig.update_layout(**chart_layout(theme["accent"], 430))
        fig.update_xaxes(type="linear", dtick=1, tickformat="d")
        fig.update_yaxes(rangemode="tozero")
        st.plotly_chart(fig, use_container_width=True)

        if flagged.empty:
            st.success("No annual total exceeded this IQR threshold.")
        else:
            st.warning(
                f"{len(flagged)} year(s) exceeded the threshold. "
                "Review these records for data quality and context."
            )
            st.dataframe(flagged, use_container_width=True, hide_index=True)

        st.markdown("## Alert explanation")
        if not flagged.empty:
            # Highlight the flagged year with the largest reported total.
            alert_row = flagged.sort_values("Total_Cases", ascending=False).iloc[0]
            alert_year = int(alert_row["year"]) if pd.notna(alert_row["year"]) else alert_row["year"]
            alert_cases = float(alert_row["Total_Cases"])
            above_threshold = alert_cases - float(threshold)

            st.warning(
                f"Unusual reported total detected for {alert_year}. "
                "This is a statistical flag for review, not confirmation of an outbreak."
            )
            alert_col1, alert_col2, alert_col3 = st.columns(3)
            alert_col1.metric("Year flagged", str(alert_year))
            alert_col2.metric("Reported cases", fmt_num(alert_cases))
            alert_col3.metric("Above review threshold", f"{above_threshold:,.1f}")

            st.markdown("**Why was this year flagged?**")
            st.write(
                f"The reported total ({fmt_num(alert_cases)}) is above the "
                f"statistical review threshold ({threshold:,.1f}). "
                "The dashboard marks this year so the data can be checked."
            )
        else:
            st.success(
                "No annual total exceeded the current IQR review threshold. "
                "This does not guarantee that the data contains no issues."
            )

        st.markdown("**Recommended next steps**")
        st.write(
            "1. Check the original source records for missing values, duplicates, or entry errors.\n"
            "2. Compare the flagged year with reliable official public-health reports.\n"
            "3. Consider whether reporting coverage or data collection changed between years."
        )
        st.caption(
            "Important: IQR is a basic statistical screening rule. An unusual value is not, "
            "by itself, evidence that a disease outbreak occurred."
        )

        with st.expander("View how the threshold is calculated"):
            st.write(f"Q1 (25th percentile): {q1:,.2f}")
            st.write(f"Q3 (75th percentile): {q3:,.2f}")
            st.write(f"IQR (Q3 − Q1): {iqr:,.2f}")
            st.write(f"Review threshold = Q3 + 1.5 × IQR: {threshold:,.2f}")

# ============================================================
# PAGE 6: PYSPARK PERFORMANCE
# ============================================================

elif page == PAGES[5]:
    show_hero(
        "PySpark Performance Lab",
        "Compare equivalent Pandas and local PySpark grouping workloads using measured timings and repeatable runs.",
        theme["accent"],
        "PARALLEL & DISTRIBUTED COMPUTING",
    )

    st.markdown("## What this experiment measures")
    st.write(
        "This experiment compares grouped record counts using Pandas and Apache Spark. "
        "It reports Pandas workload time, Spark workload time after the Spark DataFrame exists, "
        "and Spark end-to-end time including session startup and conversion. Spark may be slower "
        "for small datasets because startup and conversion overhead can dominate."
    )
    st.info(
        "This dashboard runs Spark in `local[*]` mode on one machine. It demonstrates Spark's "
        "local execution engine; it is not evidence of performance on a multi-node cluster."
    )

    if raw_df.empty:
        st.warning(
            "`cleaned_health_data.csv` is required for the benchmark. Place it beside "
            "`dashboard.py`, then reload the app."
        )
    else:
        data_path = BASE_DIR / "cleaned_health_data.csv"
        c1, c2, c3 = st.columns(3)
        c1.metric("Input rows", f"{len(raw_df):,}")
        c2.metric("Input columns", f"{len(raw_df.columns):,}")
        c3.metric("Input size", f"{data_path.stat().st_size / (1024 * 1024):.2f} MB")

        year_col = find_column(raw_df, ["year", "Year", "YEAR", "reporting_year"])
        state_col = find_column(raw_df, ["state_ut", "State/UT", "state", "State", "state_name", "State_Name"])
        disease_col = find_column(raw_df, ["Disease", "disease", "disease_name", "Disease_Name", "Disease Name"])
        dimensions = list(dict.fromkeys(c for c in [year_col, state_col, disease_col] if c))

        if not dimensions:
            st.error(
                "Could not detect a year, state/UT, or disease column in the cleaned dataset. "
                "Update the candidate column names in dashboard.py to match your CSV."
            )
        else:
            st.markdown("**Detected grouping columns**")
            st.write(", ".join(str(c) for c in dimensions))
            st.caption(
                "Each operation counts records by one detected dimension. It does not assume that "
                "each row represents a unique confirmed case."
            )
            sample_percent = st.select_slider(
                "Dataset size for this benchmark",
                options=[25, 50, 75, 100],
                value=100,
                format_func=lambda value: f"{value}% of rows",
                help="Use the same deterministic sample for Pandas and PySpark. Run multiple sizes to study scaling.",
            )
            repetitions = st.select_slider(
                "Timed repetitions (median is reported)",
                options=[3, 5, 7],
                value=3,
                help="More repetitions make timing less sensitive to one unusually fast or slow run.",
            )

            if st.button("▶ Run Pandas vs PySpark benchmark", type="primary", use_container_width=True):
                benchmark_df = raw_df if sample_percent == 100 else raw_df.sample(
                    frac=sample_percent / 100.0, random_state=42
                )
                pandas_times = []
                pandas_results = []
                pandas_error = None
                try:
                    # One untimed warm-up, followed by repeated measured workloads.
                    for col in dimensions:
                        benchmark_df.groupby(col, dropna=False).size()
                    for _ in range(repetitions):
                        start_time = time.perf_counter()
                        run_results = []
                        for col in dimensions:
                            grouped = benchmark_df.groupby(col, dropna=False).size()
                            run_results.append((str(col), int(len(grouped)), int(grouped.sum())))
                        pandas_times.append(time.perf_counter() - start_time)
                    pandas_results = run_results
                except Exception as exc:
                    pandas_error = str(exc)

                spark = None
                spark_error = None
                spark_startup_conversion_seconds = None
                spark_work_times = []
                spark_end_to_end_seconds = None
                spark_results = []
                equivalent_results = False
                spark_version = None
                java_path = shutil.which("java")
                java_home = os.environ.get("JAVA_HOME", "")

                if pandas_error is None:
                    try:
                        if not java_path:
                            raise RuntimeError(
                                "Java was not found on PATH. Install a Java version supported by your "
                                "PySpark version, set JAVA_HOME, add %JAVA_HOME%\\bin to PATH, and "
                                "restart VS Code before rerunning the benchmark."
                            )

                        from pyspark.sql import functions as F

                        # Reuse a single cached Spark session across Streamlit reruns.
                        spark_start = time.perf_counter()
                        spark = get_spark_session()
                        spark.sparkContext.setLogLevel("ERROR")
                        spark_df = spark.createDataFrame(benchmark_df)
                        spark_startup_conversion_seconds = time.perf_counter() - spark_start
                        spark_version = spark.version

                        def spark_grouping_workload():
                            output = []
                            for col in dimensions:
                                grouped_rows = (
                                    spark_df
                                    .groupBy(F.col(str(col)))
                                    .count()
                                    .collect()
                                )
                                distinct_groups = len(grouped_rows)
                                rows_counted = sum(int(row["count"]) for row in grouped_rows)
                                output.append((str(col), distinct_groups, rows_counted))
                            return output

                        # Warm-up run is deliberately excluded from measured timings.
                        spark_grouping_workload()

                        for _ in range(repetitions):
                            start_time = time.perf_counter()
                            run_results = spark_grouping_workload()
                            spark_work_times.append(time.perf_counter() - start_time)

                        spark_results = run_results
                        spark_work_median = float(pd.Series(spark_work_times).median())
                        spark_end_to_end_seconds = (
                            spark_startup_conversion_seconds + spark_work_median
                        )

                        pandas_lookup = {row[0]: (row[1], row[2]) for row in pandas_results}
                        spark_lookup = {row[0]: (row[1], row[2]) for row in spark_results}
                        equivalent_results = pandas_lookup == spark_lookup

                    except Exception as exc:
                        spark_error = f"{type(exc).__name__}: {exc}"

                pandas_median = float(pd.Series(pandas_times).median()) if pandas_times else None
                spark_work_median = float(pd.Series(spark_work_times).median()) if spark_work_times else None
                st.session_state["ph_benchmark"] = {
                    "pandas_seconds": pandas_median,
                    "pandas_times": pandas_times,
                    "spark_work_seconds": spark_work_median,
                    "spark_work_times": spark_work_times,
                    "spark_startup_conversion_seconds": spark_startup_conversion_seconds,
                    "spark_end_to_end_seconds": spark_end_to_end_seconds,
                    "pandas_results": pandas_results,
                    "spark_results": spark_results,
                    "spark_error": spark_error,
                    "pandas_error": pandas_error,
                    "equivalent_results": equivalent_results,
                    "dimensions": dimensions,
                    "rows": len(benchmark_df) if pandas_error is None else 0,
                    "source_rows": len(raw_df),
                    "sample_percent": sample_percent,
                    "repetitions": repetitions,
                    "spark_version": spark_version,
                    "java_path": java_path,
                    "java_home": java_home,
                }

            bench = st.session_state.get("ph_benchmark")
            if (bench and bench.get("source_rows") == len(raw_df)
                    and bench.get("sample_percent") == sample_percent
                    and bench.get("dimensions") == dimensions):
                st.markdown("## Measured results")
                if bench.get("pandas_error"):
                    st.error("Pandas benchmark failed.")
                    st.code(bench["pandas_error"])
                elif bench.get("spark_error"):
                    st.error("PySpark could not start. This is an environment setup issue, not a benchmark result.")
                    st.code(bench["spark_error"])
                    st.markdown("**Java environment check**")
                    st.write(f"Java executable found on PATH: {'Yes' if bench.get('java_path') else 'No'}")
                    st.write(f"JAVA_HOME configured: {'Yes' if bench.get('java_home') else 'No'}")
                    if "JAVA_GATEWAY_EXITED" in str(bench["spark_error"]) or "Java gateway" in str(bench["spark_error"]) or not bench.get("java_path"):
                        st.markdown("**Try these steps in the VS Code terminal:**")
                        st.code("java -version\nwhere java\necho %JAVA_HOME%\npython -m pip show pyspark", language="powershell")
                        st.write(
                            "Install a Java version supported by your installed PySpark/Spark version. "
                            "Set `JAVA_HOME` to the Java installation directory (not its `bin` folder), "
                            "add `%JAVA_HOME%\\bin` to `Path`, then fully close and reopen VS Code and restart Streamlit. "
                            "If Java is already installed, check that the Java version is supported by your PySpark version. "
                            "Do not report a speedup until the Spark run succeeds and output validation passes."
                        )
                    st.metric("Pandas median workload time", f"{bench['pandas_seconds']:.4f} s")
                    st.caption("This is a Pandas-only measurement. It is not a Pandas-versus-PySpark comparison because Spark did not run.")
                elif bench.get("spark_work_seconds") is not None:
                    a, b, c = st.columns(3)
                    a.metric("Pandas median workload", f"{bench['pandas_seconds']:.4f} s")
                    b.metric("PySpark median workload", f"{bench['spark_work_seconds']:.4f} s")
                    if bench["spark_work_seconds"] > 0:
                        speedup = bench["pandas_seconds"] / bench["spark_work_seconds"]
                        c.metric("Speedup (Pandas ÷ Spark)", f"{speedup:.2f}×")
                        if speedup > 1:
                            c.caption("PySpark workload was faster in these measured runs.")
                        elif speedup < 1:
                            c.caption("Pandas workload was faster in these measured runs.")
                        else:
                            c.caption("Workload times were approximately equal.")

                    st.metric("Spark startup + conversion", f"{bench['spark_startup_conversion_seconds']:.4f} s")
                    st.metric("Spark end-to-end estimate", f"{bench['spark_end_to_end_seconds']:.4f} s")
                    st.caption(
                        f"Measured {bench['rows']:,} rows ({bench['sample_percent']}% of the source dataset); "
                        f"median of {bench['repetitions']} timed repetitions after one warm-up. "
                        f"Spark version: {bench.get('spark_version') or 'unknown'}."
                    )
                    if bench.get("equivalent_results"):
                        st.success("Validation passed: Pandas and PySpark returned the same group counts and total rows.")
                    else:
                        st.error("Validation did not pass. Do not compare the timing results until the grouping outputs are reconciled.")

                    timing_df = pd.DataFrame([
                        {"Measurement": "Pandas median workload", "Seconds": bench["pandas_seconds"]},
                        {"Measurement": "PySpark median workload", "Seconds": bench["spark_work_seconds"]},
                        {"Measurement": "Spark startup + conversion", "Seconds": bench["spark_startup_conversion_seconds"]},
                        {"Measurement": "Spark end-to-end estimate", "Seconds": bench["spark_end_to_end_seconds"]},
                    ])
                    fig = px.bar(
                        timing_df, x="Measurement", y="Seconds", color="Measurement",
                        title="Measured execution times (lower is faster)", text="Seconds",
                    )
                    fig.update_traces(texttemplate="%{text:.4f} s", textposition="outside")
                    fig.update_layout(**chart_layout(theme["accent"], 400), showlegend=False)
                    fig.update_yaxes(rangemode="tozero")
                    st.plotly_chart(fig, use_container_width=True)

                    result_rows = []
                    for item in bench["pandas_results"]:
                        result_rows.append({
                            "Grouping column": item[0], "Engine": "Pandas",
                            "Distinct groups": item[1], "Rows counted": item[2],
                            "Median workload time (s)": round(bench["pandas_seconds"], 6),
                        })
                    for item in bench["spark_results"]:
                        result_rows.append({
                            "Grouping column": item[0], "Engine": "PySpark",
                            "Distinct groups": item[1], "Rows counted": item[2],
                            "Median workload time (s)": round(bench["spark_work_seconds"], 6),
                        })
                    results_df = pd.DataFrame(result_rows)
                    st.dataframe(results_df, use_container_width=True, hide_index=True)
                    st.download_button(
                        "⬇️ Download benchmark results",
                        data=results_df.to_csv(index=False).encode("utf-8"),
                        file_name="pyspark_benchmark_results.csv",
                        mime="text/csv",
                    )

                st.markdown("## How to explain the result in your viva")
                st.write(
                    "Pandas performs DataFrame operations on one machine. PySpark uses Apache Spark's execution engine; "
                    "`local[*]` lets Spark use available local worker threads but does not create a multi-node cluster. "
                    "A speedup above 1× means the measured PySpark workload took less time than Pandas; below 1× means "
                    "Pandas took less time. Report the dataset size, repetitions, machine configuration, Java/Spark versions, "
                    "and whether startup/conversion is included."
                )
            else:
                st.info("Click **Run Pandas vs PySpark benchmark** to measure real execution times. No timing values are pre-filled.")

    st.markdown("## Setup requirements")
    st.code("python -m pip install streamlit pandas plotly pyspark", language="bash")
    st.write(
        "PySpark requires a compatible Java runtime. If Spark reports a Java or gateway error, "
        "check your Java installation and JAVA_HOME, then restart the terminal and Streamlit."
    )

# ============================================================
# PAGE 7: DATA EXPLORER
# ============================================================

elif page == PAGES[6]:
    show_hero(
        "Data Explorer",
        "Search, filter, inspect and export the cleaned public-health records.",
        theme["accent"],
        "DATA ACCESS",
    )

    raw_status = next(
        (item for item in file_status if item["file"] == "cleaned_health_data.csv"),
        None,
    )

    if raw_df.empty:
        st.warning(
            "Detailed records are unavailable. Summary pages may still work if their summary CSV files loaded correctly."
        )
        if raw_status:
            st.caption(f"File status: {raw_status['status']} — {raw_status['details']}")
        st.write("Place a valid `cleaned_health_data.csv` beside `dashboard.py`.")
        if st.button("↻ Refresh data", key="refresh_missing_data"):
            st.cache_data.clear()
            st.rerun()
    else:
        # -------------------- Search and filters --------------------
        st.markdown("## Explore records")
        search = st.text_input(
            "Search all columns",
            placeholder="Try a state, district, disease or keyword...",
            help="Search is case-insensitive and treats the text as a literal phrase.",
        )
        filtered = raw_df.copy()

        if search.strip():
            search_mask = filtered.astype(str).apply(
                lambda col: col.str.contains(search.strip(), case=False, na=False, regex=False)
            ).any(axis=1)
            filtered = filtered.loc[search_mask]

        year_col = find_column(raw_df, ["year", "Year", "YEAR", "reporting_year"])
        if year_col:
            year_values = pd.to_numeric(raw_df[year_col], errors="coerce")
            year_options = sorted(year_values.dropna().astype(int).unique().tolist())
            if year_options:
                selected_years = st.multiselect(
                    "Filter by year (optional)",
                    options=year_options,
                    default=[],
                    help="Leave this empty to include every year.",
                    placeholder="All years (no filter)",
                )
                if selected_years:
                    filtered_years = pd.to_numeric(filtered[year_col], errors="coerce")
                    filtered = filtered.loc[filtered_years.isin(selected_years)]

        metric_cols = st.columns(3)
        metric_cols[0].metric("Matching records", f"{len(filtered):,}")
        metric_cols[1].metric("Total source records", f"{len(raw_df):,}")
        metric_cols[2].metric("Available columns", f"{len(raw_df.columns):,}")

        if filtered.empty:
            st.info("No records match these filters. Try a different search or remove the year filter.")
        else:
            st.dataframe(
                filtered,
                use_container_width=True,
                hide_index=True,
                height=420,
            )

        st.download_button(
            "⬇️ Download filtered records",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="filtered_health_records.csv",
            mime="text/csv",
            type="primary",
            disabled=filtered.empty,
        )

        # -------------------- File status --------------------
        with st.expander("Data file status", expanded=False):
            st.caption(
                "These checks confirm whether the expected CSV files could be read. A successful load does not prove that their contents are accurate."
            )
            status_df = pd.DataFrame(file_status)
            st.dataframe(status_df, use_container_width=True, hide_index=True)
            problem_files = [item for item in file_status if item["status"] != "Loaded"]
            if problem_files:
                st.warning("One or more expected CSV files could not be loaded. Check the status and details above.")
                with st.expander("How to fix file problems"):
                    st.markdown(
                        "1. Confirm each filename matches exactly.\n"
                        "2. Keep the files beside `dashboard.py`.\n"
                        "3. Check that each CSV is non-empty and valid.\n"
                        "4. Use the displayed error details to troubleshoot read errors."
                    )
            else:
                st.success("All four expected CSV files loaded successfully.")
            if st.button("↻ Refresh data", key="refresh_data_explorer"):
                st.cache_data.clear()
                st.rerun()

        # -------------------- Data quality --------------------
        st.markdown("## Data quality report")
        total_cells = int(raw_df.shape[0] * raw_df.shape[1])
        missing_cells = int(raw_df.isna().sum().sum())
        rows_with_missing = int(raw_df.isna().any(axis=1).sum())
        duplicate_rows = int(raw_df.duplicated().sum())
        completeness = (100 * (total_cells - missing_cells) / total_cells) if total_cells else 0.0
        duplicate_pct = (100 * duplicate_rows / len(raw_df)) if len(raw_df) else 0.0

        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Source records", f"{len(raw_df):,}")
        q2.metric("Missing cells", f"{missing_cells:,}")
        q3.metric("Rows with missing data", f"{rows_with_missing:,}")
        q4.metric("Exact duplicate rows", f"{duplicate_rows:,}")
        st.progress(min(max(completeness / 100, 0.0), 1.0))
        st.caption(
            f"Cell completeness: {completeness:.1f}% · Duplicate rows: {duplicate_pct:.2f}% of all records. "
            "Missing cells are counted individually; a row can contain more than one missing cell."
        )

        quality = pd.DataFrame({
            "Column": [str(c) for c in raw_df.columns],
            "Missing values": [int(raw_df[c].isna().sum()) for c in raw_df.columns],
            "Missing (%)": [round(100 * raw_df[c].isna().mean(), 2) for c in raw_df.columns],
            "Distinct values": [int(raw_df[c].nunique(dropna=True)) for c in raw_df.columns],
            "Data type": [str(raw_df[c].dtype) for c in raw_df.columns],
        }).sort_values(["Missing values", "Column"], ascending=[False, True])
        st.markdown("### Missing values by column")
        st.dataframe(quality, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Download column quality report",
            data=quality.to_csv(index=False).encode("utf-8"),
            file_name="health_data_quality_report.csv",
            mime="text/csv",
        )

        # -------------------- Conditional validation checks --------------------
        raw_year_col = find_column(raw_df, ["year", "Year", "YEAR", "reporting_year"])
        raw_count_col = find_column(
            raw_df,
            ["Total_Cases", "total_cases", "Cases", "case_count", "confirmed_cases", "Count"],
        )
        validation_issues = []
        if raw_year_col:
            numeric_years = pd.to_numeric(raw_df[raw_year_col], errors="coerce")
            max_year = pd.Timestamp.now().year + 1
            suspicious_years = int(
                (numeric_years.notna() & ((numeric_years < 1900) | (numeric_years > max_year))).sum()
            )
            if suspicious_years:
                validation_issues.append(
                    f"{suspicious_years:,} rows have years outside the broad 1900–{max_year} review range."
                )
        if raw_count_col:
            numeric_counts = pd.to_numeric(raw_df[raw_count_col], errors="coerce")
            negative_counts = int((numeric_counts < 0).sum())
            if negative_counts:
                validation_issues.append(
                    f"{negative_counts:,} rows have negative values in `{raw_count_col}`."
                )

        if validation_issues:
            st.warning("Potential validation issues to review:\n\n" + "\n".join(f"- {issue}" for issue in validation_issues))
        elif raw_year_col or raw_count_col:
            st.success(
                "The available basic checks found no negative case counts or clearly out-of-range years. "
                "This is a screening check, not proof that all data is correct."
            )
        else:
            st.info("Year and case-count checks were skipped because matching columns were not found.")

        if missing_cells or duplicate_rows:
            st.warning(
                "Review missing values and exact duplicate rows before interpreting results. "
                "This report does not automatically delete or correct records."
            )

        # -------------------- Source and interpretation notes --------------------
        with st.expander("About this dataset and its limitations", expanded=False):
            st.write(
                "This table contains the cleaned records used by the dashboard. Data-quality checks flag items for review; "
                "they do not modify the source records. Reporting coverage, case definitions and geographic labels can affect comparisons."
            )
            if DATASET_SOURCE_NAME:
                st.write(f"**Dataset source:** {DATASET_SOURCE_NAME}")
            else:
                st.info("Dataset source details have not been configured yet. Add the verified source name before presenting or publishing the dashboard.")
            if DATASET_SOURCE_URL:
                st.markdown(f"**Original source:** [{DATASET_SOURCE_URL}]({DATASET_SOURCE_URL})")
            if DATASET_REPORTING_PERIOD:
                st.write(f"**Reporting period:** {DATASET_REPORTING_PERIOD}")
            if DATASET_LAST_UPDATED:
                st.write(f"**Source last updated:** {DATASET_LAST_UPDATED}")
            if DATASET_CASE_DEFINITION:
                st.write(f"**Case/count definition:** {DATASET_CASE_DEFINITION}")
            else:
                st.caption("Add a verified case/count definition so viewers understand what each reported number represents.")
            if DATASET_GEOGRAPHIC_COVERAGE:
                st.write(f"**Geographic coverage:** {DATASET_GEOGRAPHIC_COVERAGE}")
            if DATASET_LICENSE:
                st.write(f"**Data license / reuse terms:** {DATASET_LICENSE}")
            st.caption(
                "An unusual value is not, by itself, evidence that an outbreak occurred. These results are for educational analysis, not medical advice."
            )

# --------------------------- PAGE TRANSITIONS ---------------------------

st.divider()
previous_col, status_col, next_col = st.columns([1.4, 2, 1.4])

with previous_col:
    if page_index > 0:
        st.button(
            "← Previous page",
            key=f"previous_page_{page_index}",
            use_container_width=True,
            on_click=set_page,
            args=(PAGES[page_index - 1],),
        )

with status_col:
    st.markdown(
        f"<div class='page-counter'>PAGE {page_index + 1} OF {len(PAGES)}</div>",
        unsafe_allow_html=True,
    )
    st.progress((page_index + 1) / len(PAGES))

with next_col:
    if page_index < len(PAGES) - 1:
        st.button(
            "Next page →",
            key=f"next_page_{page_index}",
            type="primary",
            use_container_width=True,
            on_click=set_page,
            args=(PAGES[page_index + 1],),
        )

st.caption("Public Health Intelligence Dashboard · Educational Use Only")
