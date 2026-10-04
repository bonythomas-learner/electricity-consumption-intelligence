import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


def find_project_root():
    """Find the repository even when Streamlit starts outside its root."""
    candidates = []
    if "__file__" in globals():
        candidates.append(Path(__file__).resolve().parent)
    candidates.extend([Path.cwd(), *Path.cwd().parents])
    for candidate in candidates:
        if (candidate / "src" / "data_utils.py").is_file():
            return candidate
    raise FileNotFoundError(
        "Could not find the project root. Run Streamlit from the repository "
        "or use the full app.py path."
    )


PROJECT_ROOT = find_project_root()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_utils import build_features, load_data, make_target
from solar import solar_scenario


st.set_page_config(
    page_title="Electricity Consumption Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


REQUIRED_COLUMNS = {
    "Consumer Name",
    "AREA",
    "FEEDER_NAME",
    "VILLAGE_NAME",
    "TARRIF",
    "CONTRACT_LOAD",
    "SOLAR_CONSUMER",
}


def format_number(value, decimals=1):
    if pd.isna(value):
        return "-"
    return f"{value:,.{decimals}f}"


def unique_values(frame, column):
    if column not in frame.columns:
        return []
    return sorted(frame[column].dropna().astype(str).unique().tolist())


def filter_options(frame, column, label):
    options = unique_values(frame, column)
    return st.sidebar.multiselect(
        label,
        options,
        default=options,
        key=f"filter_{column}",
    )


def build_anomaly_table(frame, month, hours_per_day, days_in_month, low_limit, high_limit):
    benchmark = frame["CONTRACT_LOAD"] * hours_per_day * days_in_month * 0.8
    ratio = frame[month] / benchmark.replace(0, np.nan)
    result = pd.DataFrame(
        {
            "Consumer": frame["Consumer Name"].astype(str),
            "Tariff": frame["TARRIF"].astype(str),
            "Area": frame["AREA"].astype(str),
            "Village": frame["VILLAGE_NAME"].astype(str),
            "Contract load (kW)": frame["CONTRACT_LOAD"],
            f"{month} consumption (kWh)": frame[month],
            "Benchmark (kWh)": benchmark,
            "Consumption ratio": ratio,
        }
    )
    result["Review flag"] = np.select(
        [ratio < low_limit, ratio > high_limit],
        ["Low-consumption review", "High-consumption review"],
        default="Normal / insufficient evidence",
    )
    return result.sort_values("Consumption ratio", ascending=False, na_position="last")


def monthly_summary(frame, months):
    summary = frame[months].mean(axis=0).rename("Average consumption (kWh)").to_frame()
    summary["Median consumption (kWh)"] = frame[months].median(axis=0)
    summary["Consumers with readings"] = frame[months].notna().sum(axis=0)
    summary.index.name = "Month"
    return summary.reset_index()


def load_result_csv(filename):
    path = PROJECT_ROOT / "results" / filename
    if not path.exists():
        return None
    return pd.read_csv(path)


st.title("Electricity Consumption Intelligence")
st.caption(
    "Hansot subdivision dashboard for consumption trends, consumer profiling, "
    "review queues, model evaluation, and preliminary solar scenarios."
)

default_data_path = PROJECT_ROOT / "data" / "Hansot_adjusted_consumption.xlsx"

with st.sidebar:
    st.header("Data and filters")
    uploaded = st.file_uploader(
        "Upload an authorized workbook",
        type=["xlsx", "xls"],
        help="Use the permitted, anonymized electricity-consumption workbook.",
    )
    st.divider()

data_source = uploaded if uploaded is not None else default_data_path if default_data_path.exists() else None

if data_source is None:
    st.info("Upload the authorized workbook in the sidebar to open the dashboard.")
    st.subheader("Expected workbook structure")
    st.dataframe(
        pd.DataFrame(
            {
                "Field group": [
                    "Consumer identity",
                    "Location",
                    "Tariff and load",
                    "Solar status",
                    "Monthly readings",
                ],
                "Expected fields": [
                    "Consumer Name",
                    "AREA, FEEDER_NAME, VILLAGE_NAME",
                    "TARRIF, CONTRACT_LOAD",
                    "SOLAR_CONSUMER",
                    "Ten numeric month columns",
                ],
            }
        ),
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        "The private source workbook is intentionally excluded from GitHub. "
        "See docs/DATA_DICTIONARY.md for the full schema and limitations."
    )
    st.stop()

try:
    raw_df, months = load_data(data_source)
except Exception as exc:
    st.error(f"The workbook could not be read: {exc}")
    st.stop()

missing = sorted(REQUIRED_COLUMNS - set(raw_df.columns))
if len(months) < 2 or missing:
    if missing:
        st.error("Missing required columns: " + ", ".join(missing))
    if len(months) < 2:
        st.error("At least two numeric monthly columns are required.")
    st.stop()

df, first7, last3 = build_features(raw_df, months)
df, target_threshold = make_target(df)

with st.sidebar:
    st.caption(f"Loaded {len(df):,} records and {len(months)} monthly columns.")
    tariff_filter = filter_options(df, "TARRIF", "Tariff")
    area_filter = filter_options(df, "AREA", "Area")
    feeder_filter = filter_options(df, "FEEDER_NAME", "Feeder")
    village_filter = filter_options(df, "VILLAGE_NAME", "Village")
    solar_filter = st.multiselect(
        "Solar status",
        unique_values(df, "SOLAR_CONSUMER"),
        default=unique_values(df, "SOLAR_CONSUMER"),
    )

filtered = df[
    df["TARRIF"].astype(str).isin(tariff_filter)
    & df["AREA"].astype(str).isin(area_filter)
    & df["FEEDER_NAME"].astype(str).isin(feeder_filter)
    & df["VILLAGE_NAME"].astype(str).isin(village_filter)
    & df["SOLAR_CONSUMER"].astype(str).isin(solar_filter)
].copy()

if filtered.empty:
    st.warning("No consumers match the selected filters. Widen the sidebar filters to continue.")
    st.stop()

observed_months = len(months)
average_monthly = filtered[months].mean(axis=1).mean()
solar_count = (filtered["SOLAR_CONSUMER"].astype(str).str.upper() == "Y").sum()

tabs = st.tabs(
    [
        "Overview",
        "Consumption",
        "Load anomalies",
        "Model results",
        "PCA and clusters",
        "Solar scenario",
    ]
)

with tabs[0]:
    kpi = st.columns(5)
    kpi[0].metric("Consumers", f"{len(filtered):,}")
    kpi[1].metric("Average monthly kWh", format_number(average_monthly))
    kpi[2].metric("Solar consumers", f"{solar_count:,}")
    kpi[3].metric("Feeders", f"{filtered['FEEDER_NAME'].nunique():,}")
    kpi[4].metric("Missing readings", f"{int(filtered[months].isna().sum().sum()):,}")

    st.subheader("Consumption trend")
    trend = monthly_summary(filtered, months)
    st.line_chart(
        trend.set_index("Month")[["Average consumption (kWh)", "Median consumption (kWh)"]],
        height=330,
    )
    st.dataframe(trend.round(1), hide_index=True, use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Highest-consumption areas")
        area_summary = (
            filtered.groupby("AREA", dropna=False)["train_avg"]
            .agg(Consumers="size", Average="mean", Median="median")
            .sort_values("Average", ascending=False)
            .head(10)
            .round(1)
        )
        st.bar_chart(area_summary["Average"], height=280)
        st.dataframe(area_summary, use_container_width=True)
    with right:
        st.subheader("Tariff mix")
        tariff_summary = (
            filtered.groupby("TARRIF", dropna=False)
            .agg(Consumers=("TARRIF", "size"), Average_kWh=("train_avg", "mean"))
            .sort_values("Consumers", ascending=False)
            .round(1)
        )
        st.bar_chart(tariff_summary["Consumers"], height=280)
        st.dataframe(tariff_summary, use_container_width=True)

with tabs[1]:
    st.subheader("Consumption by tariff")
    tariff_table = filtered.groupby("TARRIF", dropna=False)[months].mean().T.round(1)
    st.line_chart(tariff_table, height=340)
    st.dataframe(tariff_table, use_container_width=True)

    st.subheader("Consumer review table")
    sort_choice = st.selectbox(
        "Sort consumers by",
        ["Average consumption", "Consumption variability", "Consumption per kW"],
    )
    sort_column = {
        "Average consumption": "train_avg",
        "Consumption variability": "cv",
        "Consumption per kW": "consumption_per_kw",
    }[sort_choice]
    review_columns = [
        "Consumer Name",
        "TARRIF",
        "AREA",
        "FEEDER_NAME",
        "VILLAGE_NAME",
        "CONTRACT_LOAD",
        "train_avg",
        "train_std",
        "cv",
        "consumption_per_kw",
    ]
    review = filtered[review_columns].sort_values(sort_column, ascending=False).head(250).copy()
    st.dataframe(review.round(2), hide_index=True, use_container_width=True)
    st.download_button(
        "Download filtered consumer table",
        data=review.to_csv(index=False).encode("utf-8"),
        file_name="filtered_consumer_review.csv",
        mime="text/csv",
    )

with tabs[2]:
    st.subheader("Assumption-based load-ratio review")
    st.caption(
        "This screen is a review aid, not proof of theft, meter failure, or unauthorized load. "
        "The benchmark assumes a percentage of contracted load is used during the selected hours."
    )
    control_a, control_b, control_c, control_d = st.columns(4)
    with control_a:
        selected_month = st.selectbox("Month to review", months, index=len(months) - 1)
    with control_b:
        hours_per_day = st.number_input("Assumed hours/day", 1.0, 24.0, 8.0, 1.0)
    with control_c:
        low_limit = st.number_input("Low ratio below", 0.0, 1.0, 0.25, 0.05)
    with control_d:
        high_limit = st.number_input("High ratio above", 1.0, 5.0, 1.5, 0.1)

    anomaly_table = build_anomaly_table(
        filtered,
        selected_month,
        hours_per_day,
        31,
        low_limit,
        high_limit,
    )
    counts = anomaly_table["Review flag"].value_counts()
    flag_columns = st.columns(3)
    flag_columns[0].metric("Low-consumption reviews", f"{int(counts.get('Low-consumption review', 0)):,}")
    flag_columns[1].metric("High-consumption reviews", f"{int(counts.get('High-consumption review', 0)):,}")
    flag_columns[2].metric("Normal or insufficient evidence", f"{int(counts.get('Normal / insufficient evidence', 0)):,}")
    st.dataframe(anomaly_table.head(250).round(2), hide_index=True, use_container_width=True)
    st.download_button(
        "Download anomaly review queue",
        data=anomaly_table.to_csv(index=False).encode("utf-8"),
        file_name="load_anomaly_review_queue.csv",
        mime="text/csv",
    )

with tabs[3]:
    st.subheader("Saved classification experiments")
    st.caption(
        "The target is a high-consumption proxy based on a future-period threshold. "
        f"The current first-period threshold is {target_threshold:.2f} kWh."
    )
    baseline = load_result_csv("model_metrics.csv")
    tuned = load_result_csv("model_metrics_tuned.csv")
    if baseline is None and tuned is None:
        st.info("Run src/train_models.py to create model result tables.")
    else:
        if baseline is not None:
            st.markdown("**Baseline metrics**")
            st.dataframe(baseline.round(3), hide_index=True, use_container_width=True)
        if tuned is not None:
            st.markdown("**Tuned metrics**")
            st.dataframe(tuned.round(3), hide_index=True, use_container_width=True)
            chart_columns = [column for column in ["f1", "roc_auc", "accuracy"] if column in tuned.columns]
            if chart_columns and "model" in tuned.columns:
                st.bar_chart(tuned.set_index("model")[chart_columns], height=300)

        image_columns = st.columns(2)
        for column, filename, label in [
            (image_columns[0], "knn_confusion_matrix.png", "KNN confusion matrix"),
            (image_columns[1], "decision_tree_confusion_matrix.png", "Decision Tree confusion matrix"),
        ]:
            image_path = PROJECT_ROOT / "results" / "figures" / filename
            with column:
                st.markdown(f"**{label}**")
                if image_path.exists():
                    st.image(str(image_path), use_container_width=True)

with tabs[4]:
    st.subheader("PCA projection and consumer clusters")
    pca_file = PROJECT_ROOT / "results" / "figures" / "pca_clusters.png"
    if pca_file.exists():
        st.image(str(pca_file), use_container_width=True)
    else:
        st.info("Run src/train_models.py to generate the PCA figure.")
    cluster_metrics = load_result_csv("clustering_metrics.csv")
    if cluster_metrics is not None:
        st.dataframe(cluster_metrics.round(4), hide_index=True, use_container_width=True)
    pca_variance = load_result_csv("pca_explained_variance.csv")
    if pca_variance is not None:
        st.markdown("**Explained variance by principal component**")
        st.dataframe(pca_variance.round(4), hide_index=True, use_container_width=True)

with tabs[5]:
    st.subheader("Preliminary rooftop solar scenario")
    st.caption(
        "The scenario uses the selected consumers' observed-period consumption scaled to 12 months. "
        "It is not a technical feasibility or investment recommendation."
    )
    observed_annual = filtered[months].sum(axis=1) * (12 / observed_months)
    solar_control_a, solar_control_b, solar_control_c = st.columns(3)
    with solar_control_a:
        annual = st.number_input(
            "Annual consumption (kWh)",
            min_value=0.0,
            value=float(observed_annual.median()),
            step=100.0,
        )
        capacity = st.slider("Solar capacity (kW)", 3.0, 20.0, 3.0, 0.5)
    with solar_control_b:
        tariff = st.number_input("Tariff assumption (INR/kWh)", 0.0, 100.0, 7.0, 0.5)
        yield_value = st.number_input("Yield (kWh per kW/year)", 500.0, 2500.0, 1400.0, 50.0)
    with solar_control_c:
        daytime_share = st.slider("Daytime share of demand", 0.0, 1.0, 0.45, 0.05)
        pricing_scenario = st.selectbox(
            "Pricing scenario",
            ["incremental_above_3kw", "all_capacity_above_3kw_rate"],
            format_func=lambda value: {
                "incremental_above_3kw": "3 kW base + incremental rate above 3 kW",
                "all_capacity_above_3kw_rate": "Entire system at above-3-kW rate",
            }[value],
        )

    scenario = solar_scenario(
        annual,
        capacity,
        yield_kwh_per_kw=yield_value,
        tariff=tariff,
        daytime_consumption_share=daytime_share,
        pricing_scenario=pricing_scenario,
    )
    solar_kpi = st.columns(5)
    solar_kpi[0].metric("Generation", f"{scenario['generation']:,.0f} kWh")
    solar_kpi[1].metric("On-site use", f"{scenario['on_site_consumption']:,.0f} kWh")
    solar_kpi[2].metric("Export", f"{scenario['export']:,.0f} kWh")
    solar_kpi[3].metric("Net cost", f"INR {scenario['net_cost']:,.0f}")
    solar_kpi[4].metric(
        "Payback",
        "N/A" if scenario["payback"] is None else f"{scenario['payback']:.1f} years",
    )
    st.dataframe(
        pd.DataFrame(
            {
                "Scenario output": [
                    "Annual net savings",
                    "Gross system cost",
                    "Subsidy",
                    "Maintenance",
                    "Annual bill savings",
                ],
                "Value": [
                    f"INR {scenario['annual_net_savings']:,.0f}",
                    f"INR {scenario['gross_cost']:,.0f}",
                    f"INR {scenario['subsidy']:,.0f}",
                    f"INR {scenario['maintenance']:,.0f}",
                    f"INR {scenario['gross_savings']:,.0f}",
                ],
            }
        ),
        hide_index=True,
        use_container_width=True,
    )
    st.warning(
        "Technical feasibility requires rooftop area, shading, daytime demand, tariff rules, "
        "export compensation, and site-specific yield."
    )
