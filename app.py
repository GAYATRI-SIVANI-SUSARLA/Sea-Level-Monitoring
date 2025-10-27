
# app.py
# pip install streamlit
import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
from io import StringIO
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.graph_objects as go

# -----------------------------------
# 1. App configuration
# -----------------------------------
st.set_page_config(page_title="Sea Level Monitoring App", layout="wide")

# Base URL for RLR monthly data from PSMSL
PSMSL_RLR_MONTHLY_URL = "https://psmsl.org/data/obtaining/rlr.monthly.data/{sid}.rlrdata"

# -----------------------------------
# 2. Function: Fetch data from PSMSL
# -----------------------------------
@st.cache_data(ttl=3600)
def fetch_rlr_monthly(station_id: str):
    url = PSMSL_RLR_MONTHLY_URL.format(sid=station_id)
    response = requests.get(url, timeout=20)

    if response.status_code != 200:
        raise ValueError("Failed to fetch data. Check station ID or network connection.")

    text = response.text.strip()
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.startswith('#')]
    rows = []

    for line in lines:
        parts = re.split(r"[;,\s]+", line.strip())
        numeric = [p for p in parts if re.match(r"^-?\d+(\.\d+)?$", p)]
        if len(numeric) >= 2:
            date_dec = float(numeric[0])
            value = float(numeric[1])
            if value == -99999:
                value = np.nan
            rows.append((date_dec, value))

    if not rows:
        raise ValueError("No data parsed. The station might not exist or is missing monthly data.")

    df = pd.DataFrame(rows, columns=["decimal_year", "msl_mm"])

    # Convert decimal year to datetime
    def decimal_to_datetime(dec):
        year = int(np.floor(dec))
        month = int(round((dec - year) * 12 + 0.5))
        month = max(1, min(12, month))
        return pd.Timestamp(year=year, month=month, day=15)

    df["date"] = df["decimal_year"].apply(decimal_to_datetime)
    df.set_index("date", inplace=True)
    return df[["msl_mm"]]

# -----------------------------------
# 3. Function: Compute linear trend
# -----------------------------------
def compute_linear_trend(df):
    df_valid = df.dropna()
    x = df_valid.index.year + (df_valid.index.month - 0.5) / 12.0
    y = df_valid["msl_mm"].values
    slope, intercept = np.polyfit(x, y, 1)  # slope in mm/year

    # Generate trend line for plotting
    x_all = df.index.year + (df.index.month - 0.5) / 12.0
    trend = intercept + slope * x_all
    return slope, intercept, pd.Series(trend, index=df.index)

# -----------------------------------
# 4. Function: Plot interactive graph
# -----------------------------------
def plot_sea_level(df, trend_series, station_name):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index, y=df["msl_mm"],
        mode='lines', name='Monthly Mean Sea Level (mm)',
        line=dict(color='royalblue')
    ))

    fig.add_trace(go.Scatter(
        x=trend_series.index, y=trend_series,
        mode='lines', name='Linear Trend',
        line=dict(color='red')
    ))

    rolling = df["msl_mm"].rolling(12, center=True).mean()
    fig.add_trace(go.Scatter(
        x=rolling.index, y=rolling,
        mode='lines', name='12-Month Rolling Mean',
        line=dict(dash='dash', color='green')
    ))

    fig.update_layout(
        title=f"Sea Level Trend - {station_name}",
        xaxis_title="Year",
        yaxis_title="Mean Sea Level (mm, RLR datum)",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# -----------------------------------
# 5. Streamlit UI
# -----------------------------------
st.title("🌊 Regional Sea Level Monitoring App (PSMSL)")
st.markdown("""
This app visualizes monthly sea level changes for **one tide gauge location** using data from the 
[Permanent Service for Mean Sea Level (PSMSL)](https://psmsl.org/data/obtaining/).

**Instructions:**
1. Enter a valid PSMSL Station ID (e.g., `202` for Newlyn, UK).
2. Click **Fetch Data** to download and visualize the tide gauge data.
3. Explore sea-level trends, rolling averages, and seasonal components.
""")

# User inputs
station_id = st.text_input("Enter PSMSL Station ID (e.g., 202):", "202")
station_name = st.text_input("Enter Station Name (optional):", "Newlyn, UK")

if st.button("🔎 Fetch Data"):
    try:
        with st.spinner("Fetching data from PSMSL..."):
            df = fetch_rlr_monthly(station_id.strip())

        st.success(f"Data fetched successfully! {len(df)} monthly records available.")
        st.write("**Preview of the dataset:**")
        st.dataframe(df.head(10))

        # Trend calculation
        slope, intercept, trend_series = compute_linear_trend(df)
        st.metric("📈 Linear Trend", f"{slope:.3f} mm/year")

        # Plot data
        fig = plot_sea_level(df, trend_series, station_name)
        st.plotly_chart(fig, use_container_width=True)

        # Seasonal Decomposition
        st.subheader("🔎 Seasonal Decomposition (Trend + Seasonal + Residual)")
        df_interp = df.interpolate()
        result = seasonal_decompose(df_interp["msl_mm"], period=12, model="additive", extrapolate_trend='freq')

        st.line_chart(pd.DataFrame({
            "Observed": result.observed,
            "Trend": result.trend,
            "Seasonal": result.seasonal,
            "Residual": result.resid
        }))

        # CSV download option
        csv = df.to_csv()
        st.download_button(
            label="📥 Download CSV Data",
            data=csv,
            file_name=f"psmsl_station_{station_id}.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"❌ Error: {e}")
