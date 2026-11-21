from datetime import date
import os
import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Hospital Resource & Budget Forecaster",
    layout="wide",
)

DEPARTMENTS = ["Emergency", "ICU", "General Ward", "Pediatrics"]


@st.cache_resource
def load_model():
    if not os.path.exists("model.pkl"):
        return None
    return joblib.load("model.pkl")


def format_inr(amount: float) -> str:
    if amount >= 1e7:
        return f"₹{amount / 1e7:.2f} Cr"
    if amount >= 1e5:
        return f"₹{amount / 1e5:.2f} Lakhs"
    return f"₹{amount:,.0f}"


def generate_forecast(model, start_date, days: int, selected_dept: str, surge_pct: int) -> pd.DataFrame:
    dates = pd.date_range(start=start_date, periods=days, freq="D")
    depts = DEPARTMENTS if selected_dept == "All Departments" else [selected_dept]

    rows = []
    for d in dates:
        for dept in depts:
            rows.append({
                "date": d,
                "department": dept,
                "day_of_year": d.dayofyear,
                "month": d.month,
                "day_of_week": d.dayofweek,
                "is_weekend": 1 if d.dayofweek >= 5 else 0,
            })

    input_df = pd.DataFrame(rows)
    feature_cols = ["department", "day_of_year", "month", "day_of_week", "is_weekend"]
    preds = model.predict(input_df[feature_cols])

    multiplier = 1 + (surge_pct / 100.0)

    return pd.DataFrame({
        "Date": input_df["date"].dt.strftime("%Y-%m-%d"),
        "Department": input_df["department"],
        "Predicted Admissions": np.round(preds[:, 0] * multiplier).astype(int),
        "Nurses Needed": np.ceil(preds[:, 1] * multiplier).astype(int),
        "Doctors Needed": np.ceil(preds[:, 2] * multiplier).astype(int),
        "Estimated Budget (INR)": np.round(preds[:, 3] * multiplier).astype(int),
    })


# --- Main UI ---
st.title("Healthcare Resource Allocation & Budget Forecasting")
st.caption("AI-driven patient volume, staffing, and operational budget planner (SDG 3)")

model = load_model()
if model is None:
    st.error("`model.pkl` not found in this folder. Run `train_model.py` first.")
    st.stop()

st.sidebar.header("Forecast Controls")
selected_dept = st.sidebar.selectbox("Department", ["All Departments"] + DEPARTMENTS)
start_date = st.sidebar.date_input("Start Date", value=date.today())
forecast_days = st.sidebar.slider("Forecast Horizon (Days)", min_value=1, max_value=90, value=30)
surge_pct = st.sidebar.slider(
    "Surge / Contingency Buffer (%)",
    min_value=0,
    max_value=30,
    value=0,
    help="Simulate seasonal outbreaks or supply cost inflation.",
)

forecast_df = generate_forecast(model, start_date, forecast_days, selected_dept, surge_pct)

daily_totals = (
    forecast_df.groupby("Date", as_index=False)[
        ["Predicted Admissions", "Nurses Needed", "Doctors Needed", "Estimated Budget (INR)"]
    ]
    .sum()
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Forecasted Admissions", f"{int(daily_totals['Predicted Admissions'].sum()):,}")
col2.metric("Peak Daily Nurses Needed", f"{int(daily_totals['Nurses Needed'].max())}")
col3.metric("Peak Daily Doctors Needed", f"{int(daily_totals['Doctors Needed'].max())}")
col4.metric("Total Estimated Budget", format_inr(daily_totals["Estimated Budget (INR)"].sum()))

st.divider()

# Properly scaled Altair charts (no clipped tops or accidental scroll zoom)
adm_max = int(daily_totals["Predicted Admissions"].max() * 1.15) + 2
budget_max = int(daily_totals["Estimated Budget (INR)"].max() * 1.15)

chart_col1, chart_col2 = st.columns(2)
with chart_col1:
    st.subheader("Daily Predicted Admissions")
    adm_chart = (
        alt.Chart(daily_totals)
        .mark_line(point=True, color="#1f77b4")
        .encode(
            x=alt.X("Date:T", title="Date"),
            y=alt.Y("Predicted Admissions:Q", scale=alt.Scale(domain=[0, adm_max]), title="Admissions"),
            tooltip=["Date:T", "Predicted Admissions:Q"],
        )
        .properties(height=320)
    )
    st.altair_chart(adm_chart, use_container_width=True)

with chart_col2:
    st.subheader("Daily Estimated Budget (INR)")
    budget_chart = (
        alt.Chart(daily_totals)
        .mark_line(point=True, color="#008080")
        .encode(
            x=alt.X("Date:T", title="Date"),
            y=alt.Y("Estimated Budget (INR):Q", scale=alt.Scale(domain=[0, budget_max]), title="Budget (INR)"),
            tooltip=["Date:T", "Estimated Budget (INR):Q"],
        )
        .properties(height=320)
    )
    st.altair_chart(budget_chart, use_container_width=True)

if selected_dept == "All Departments":
    st.subheader("Department Budget & Resource Breakdown")
    dept_summary = (
        forecast_df.groupby("Department", as_index=False)
        .agg(
            Total_Admissions=("Predicted Admissions", "sum"),
            Peak_Daily_Nurses=("Nurses Needed", "max"),
            Peak_Daily_Doctors=("Doctors Needed", "max"),
            Total_Budget_INR=("Estimated Budget (INR)", "sum"),
        )
    )
    b_col1, b_col2 = st.columns([1.2, 1])
    with b_col1:
        st.bar_chart(dept_summary.set_index("Department")["Total_Budget_INR"])
    with b_col2:
        st.dataframe(dept_summary, hide_index=True)

st.subheader("Detailed Daily Forecast")
st.dataframe(forecast_df, hide_index=True)

csv_data = forecast_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download Forecast as CSV",
    data=csv_data,
    file_name=f"hospital_forecast_{start_date}_{forecast_days}d.csv",
    mime="text/csv",
)
