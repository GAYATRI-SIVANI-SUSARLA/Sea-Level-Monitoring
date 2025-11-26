# app.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.graph_objects as go
from io import StringIO

st.set_page_config(page_title="Sea Level Monitoring App (PSMSL)", layout="wide")

# ----------------------------------------------------
# URLs + Headers
# ----------------------------------------------------
PSMSL_STATION_LIST_URL = "https://psmsl.org/data/obtaining/stations.lst"
PSMSL_RLR_MONTHLY_URL = "https://psmsl.org/data/obtaining/rlr.monthly.data/{sid}.rlrdata"

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
}

# ----------------------------------------------------
# Fetch Station Metadata
# ----------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_station_metadata():
    resp = requests.get(PSMSL_STATION_LIST_URL, headers=REQUEST_HEADERS, timeout=20)
    if resp.status_code != 200:
        raise ValueError(f"Failed to fetch station list. Status: {resp.status_code}")

    station_map = {}

    for line in resp.text.splitlines():
        if not line.strip():
            continue
        if not line[0].isdigit():
            continue

        parts = line.split()
        if len(parts) < 5:
            continue

        sid = parts[0]
        name = " ".join(parts[4:])
        name = re.sub(r'\s+', ' ', name)
        station_map[sid] = name

    if not station_map:
        raise ValueError("Station list parsed but no stations found.")

    return station_map

# ----------------------------------------------------
# Fetch Individual Station Data
# ----------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_rlr_monthly(station_id: str):
    url = PSMSL_RLR_MONTHLY_URL.format(sid=station_id)
    resp = requests.get(url, headers=REQUEST_HEADERS, timeout=20)

    if resp.status_code != 200:
        raise ValueError(f"Failed to fetch data for station {station_id}")

    text = resp.text
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]
    rows = []

    for line in lines:
        parts = re.split(r"[;,\s]+", line.strip())
        nums = [p for p in parts if re.match(r"^-?\d+(\.\d+)?$", p)]

        if len(nums) >= 2:
            dec = float(nums[0])
            val = float(nums[1])
            if val == -99999:
                val = np.nan
            rows.append((dec, val))

    if not rows:
        raise ValueError(f"No valid monthly data for station {station_id}")

    df = pd.DataFrame(rows, columns=["dec", "msl_mm"])

    def convert(dec):
        yr = int(np.floor(dec))
        mo = int(round((dec - yr) * 12 + 0.5))
        mo = max(1, min(mo, 12))
        return pd.Timestamp(year=yr, month=mo, day=15)

    df["date"] = df["dec"].apply(convert)
    df = df.set_index("date").sort_index()

    return df[["msl_mm"]]

# ----------------------------------------------------
# Compute Trend
# ----------------------------------------------------
def compute_linear_trend(df):
    df_valid = df.dropna()
    if df_valid.empty:
        raise ValueError("Not enough data for trend.")

    x = df_valid.index.year + (df_valid.index.month - 0.5) / 12
    y = df_valid["msl_mm"].values

    slope, intercept = np.polyfit(x, y, 1)
    x_all = df.index.year + (df.index.month - 0.5) / 12
    trend_series = intercept + slope * x_all

    return slope, pd.Series(trend_series, index=df.index)

# ----------------------------------------------------
# Multi-station Plot
# ----------------------------------------------------
def plot_multi_station(data_dict):
    fig = go.Figure()

    for sid, info in data_dict.items():
        df = info["df"]
        trend = info["trend"]
        name = info["name"]

        fig.add_trace(go.Scatter(
            x=df.index,
            y=df["msl_mm"],
            mode="lines",
            name=f"{name} ({sid})"
        ))

        fig.add_trace(go.Scatter(
            x=trend.index,
            y=trend,
            mode="lines",
            line=dict(dash="dash"),
            name=f"Trend — {name} ({sid})"
        ))

    fig.update_layout(
        title="Sea Level Comparison Across Tide Gauges",
        xaxis_title="Year",
        yaxis_title="Sea Level (mm)",
        template="plotly_white"
    )

    return fig

# ----------------------------------------------------
# Initialize station_map safely
# ----------------------------------------------------
station_map = {}
all_ids = []

try:
    station_map = fetch_station_metadata()
    all_ids = sorted(station_map.keys(), key=lambda s: int(s))
except Exception as e:
    st.error(f"❌ Unable to fetch PSMSL station list: {e}")
    st.stop()

# ----------------------------------------------------
# Streamlit UI
# ----------------------------------------------------
st.title("🌊 PSMSL Sea Level Monitoring (Live Data)")

st.subheader("Select one or more stations")

selected_ids = st.multiselect(
    "Choose station IDs:",
    options=all_ids,
    format_func=lambda sid: f"{sid} — {station_map[sid]}"
)

if not selected_ids:
    st.info("Select at least one station to continue.")
    st.stop()

# ----------------------------------------------------
# Fetch & Plot Data
# ----------------------------------------------------
if st.button("Fetch and Plot"):
    data_dict = {}
    errors = []

    for sid in selected_ids:
        try:
            df = fetch_rlr_monthly(sid)
            slope, trend = compute_linear_trend(df)
            data_dict[sid] = {
                "df": df,
                "trend": trend,
                "slope": slope,
                "name": station_map[sid]
            }
        except Exception as e:
            errors.append((sid, str(e)))

    for sid, msg in errors:
        st.error(f"{sid}: {msg}")

    if not data_dict:
        st.error("No valid data to plot.")
        st.stop()

    fig = plot_multi_station(data_dict)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Linear Trends (mm/year)")
    cols = st.columns(3)
    i = 0
    for sid, info in data_dict.items():
        cols[i % 3].metric(
            f"{info['name']} ({sid})",
            f"{info['slope']:.3f}"
        )
        i += 1

    if len(data_dict) == 1:
        sid = list(data_dict.keys())[0]
        st.subheader(f"Seasonal Decomposition — {data_dict[sid]['name']} ({sid})")

        df_interp = data_dict[sid]["df"].interpolate()

        try:
            result = seasonal_decompose(df_interp["msl_mm"], period=12, model="additive")
            st.line_chart(pd.DataFrame({
                "Observed": result.observed,
                "Trend": result.trend,
                "Seasonal": result.seasonal,
                "Residual": result.resid
            }))
        except:
            st.warning("Not enough data for seasonal decomposition.")

    st.success("Done!")
