# app.py
# pip install streamlit pandas numpy requests statsmodels plotly

import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
import time
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.graph_objects as go

# ----------------------------------- 
# 1. App configuration
# ----------------------------------- 
st.set_page_config(page_title="Sea Level Monitoring App", layout="wide")

# Base URL for PSMSL data
PSMSL_RLR_MONTHLY_URL = "https://psmsl.org/data/obtaining/rlr.monthly.data/{sid}.rlrdata"

# Expanded station database
STATION_DATABASE = {
    # Europe
    "1": "BREST", "2": "MARSEILLE", "183": "LIVERPOOL", "202": "NEWLYN",
    "229": "HOEK VAN HOLLAND", "235": "DELFZIJL", "308": "CASCAIS",
    "1551": "OSLO", "1560": "STOCKHOLM", "1571": "HELSINKI",
    "1619": "COPENHAGEN", "1639": "GDANSK", "1657": "ST. PETERSBURG",
    "1196": "REYKJAVIK", "1800": "GENOA", "24": "ABERDEEN", "25": "NORTH SHIELDS",
    # North America - East Coast
    "12": "SAN FRANCISCO", "13": "BALTIMORE", "22": "KEY WEST",
    "50": "SEATTLE", "135": "PORTLAND (MAINE)", "366": "GALVESTON",
    "1072": "SITKA", "1703": "BOSTON", "1820": "NEW YORK (THE BATTERY)",
    "2039": "ATLANTIC CITY", "2080": "CHARLESTON", "2311": "GALVESTON II",
    "2440": "SAN DIEGO", "2595": "LOS ANGELES", "2619": "HONOLULU",
    "519": "MONTAUK", "20": "EASTPORT", "368": "SANDY HOOK", "531": "PROVIDENCE",
    # South America
    "1214": "BUENOS AIRES", "1305": "RIO DE JANEIRO",
    # Africa
    "10": "LAGOS", "825": "ADEN", "1339": "CAPE TOWN",
    # Asia
    "363": "KARACHI", "367": "MUMBAI (BOMBAY)", "500": "COLOMBO",
    "523": "MANILA", "850": "SINGAPORE", "1402": "TOKYO",
    "1415": "HONG KONG", "1484": "SHANGHAI", "1517": "TIANJIN",
    # Oceania
    "656": "WELLINGTON", "679": "SYDNEY (FORT DENISON)", "680": "FREMANTLE",
}

# ----------------------------------- 
# 2. Function: Get station name
# ----------------------------------- 
def get_station_name(station_id, data_text=None):
    """Get station name from database or data file."""
    # Try hardcoded database first
    if station_id in STATION_DATABASE:
        return STATION_DATABASE[station_id]
    
    # Extract from data file header - improved extraction
    if data_text:
        lines = data_text.splitlines()
        for line in lines[:20]:  # Check first 20 lines for header
            if line.strip().startswith('#'):
                # Remove # and get content
                content = line.replace('#', '').strip()
                
                # Look for patterns like "STATION_NAME; coordinates; other_info"
                if ';' in content:
                    parts = content.split(';')
                    if len(parts) >= 1:
                        potential_name = parts[0].strip()
                        
                        # Clean up the name
                        potential_name = potential_name.split('(')[0].strip()  # Remove parentheses content
                        potential_name = potential_name.split('[')[0].strip()  # Remove bracket content
                        
                        # Validate it's actually a name (not coordinates, numbers, or metadata)
                        if potential_name and len(potential_name) >= 3:
                            # Check it's not just numbers or coordinates
                            if not potential_name.replace(' ', '').replace('.', '').replace('-', '').replace(',', '').isdigit():
                                # Check it doesn't look like coordinates (e.g., "50 06N")
                                if not any(coord in potential_name.upper() for coord in ['N ', 'S ', 'E ', 'W ', ' N', ' S', ' E', ' W']):
                                    return potential_name.upper()
    
    return f"STATION {station_id}"

# ----------------------------------- 
# 3. Function: Fetch data from PSMSL with retry logic
# ----------------------------------- 
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_rlr_monthly(station_id: str, max_retries=3):
    """Fetch monthly RLR data with retry logic."""
    last_error = None
    
    for attempt in range(max_retries):
        try:
            url = PSMSL_RLR_MONTHLY_URL.format(sid=station_id)
            response = requests.get(url, timeout=30)
            
            if response.status_code != 200:
                raise ValueError(f"HTTP {response.status_code}")
            
            text = response.text.strip()
            
            if not text or len(text) < 50:
                raise ValueError("Empty or invalid response")
            
            # Get station name
            station_name = get_station_name(station_id, text)
            
            # Parse data
            lines = [ln for ln in text.splitlines() if ln.strip() and not ln.startswith('#')]
            
            if not lines:
                raise ValueError("No data lines found")
            
            rows = []
            for line in lines:
                parts = re.split(r"[;,\s]+", line.strip())
                numeric = [p for p in parts if re.match(r"^-?\d+(\.\d+)?$", p)]
                
                if len(numeric) >= 2:
                    date_dec = float(numeric[0])
                    value = float(numeric[1])
                    if value != -99999:
                        rows.append((date_dec, value))
            
            if not rows:
                raise ValueError("No valid data points")
            
            df = pd.DataFrame(rows, columns=["decimal_year", "msl_mm"])
            
            # Convert to datetime
            def decimal_to_datetime(dec):
                year = int(np.floor(dec))
                month = int(round((dec - year) * 12 + 0.5))
                month = max(1, min(12, month))
                return pd.Timestamp(year=year, month=month, day=15)
            
            df["date"] = df["decimal_year"].apply(decimal_to_datetime)
            df.set_index("date", inplace=True)
            
            return df[["msl_mm"]], station_name
            
        except Exception as e:
            last_error = str(e)
            if attempt < max_retries - 1:
                time.sleep(1 * (attempt + 1))  # Exponential backoff
            continue
    
    raise ValueError(f"Failed after {max_retries} attempts: {last_error}")

# ----------------------------------- 
# 4. Function: Compute linear trend
# ----------------------------------- 
def compute_linear_trend(df):
    """Calculate linear trend for sea level data."""
    df_valid = df.dropna()
    
    if len(df_valid) < 2:
        return 0, 0, pd.Series(index=df.index, data=np.nan)
    
    x = df_valid.index.year + (df_valid.index.month - 0.5) / 12.0
    y = df_valid["msl_mm"].values
    
    slope, intercept = np.polyfit(x, y, 1)
    
    x_all = df.index.year + (df.index.month - 0.5) / 12.0
    trend = intercept + slope * x_all
    
    return slope, intercept, pd.Series(trend, index=df.index)

# ----------------------------------- 
# 5. Function: Plot multiple stations
# ----------------------------------- 
def plot_sea_level_multi(data_dict, show_trends=True, show_rolling=True):
    """Plot sea level data for multiple stations."""
    fig = go.Figure()
    
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    for idx, (station_id, info) in enumerate(data_dict.items()):
        color = colors[idx % len(colors)]
        df = info['df']
        station_name = info['name']
        
        # Main data
        fig.add_trace(go.Scatter(
            x=df.index, y=df["msl_mm"], mode='lines',
            name=f'{station_name} (ID: {station_id})',
            line=dict(color=color, width=2), legendgroup=station_id
        ))
        
        # Trend
        if show_trends and 'trend' in info:
            fig.add_trace(go.Scatter(
                x=info['trend'].index, y=info['trend'], mode='lines',
                name=f'Trend: {info["slope"]:.2f} mm/yr',
                line=dict(color=color, dash='dot', width=2.5), legendgroup=station_id
            ))
        
        # Rolling mean
        if show_rolling:
            rolling = df["msl_mm"].rolling(12, center=True).mean()
            fig.add_trace(go.Scatter(
                x=rolling.index, y=rolling, mode='lines',
                name=f'12-Month Avg', line=dict(color=color, dash='dash', width=1.5),
                legendgroup=station_id, visible='legendonly'
            ))
    
    fig.update_layout(
        title="Sea Level Trends - Multi-Station Comparison",
        xaxis_title="Year", yaxis_title="Mean Sea Level (mm, RLR datum)",
        template="plotly_white", hovermode='x unified',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.01),
        height=600
    )
    
    return fig

# ----------------------------------- 
# 6. Streamlit UI
# ----------------------------------- 

st.title("🌊 Sea Level Monitoring App (PSMSL)")

st.markdown("""
Compare sea level trends from **ANY** tide gauge station worldwide using data from the 
[Permanent Service for Mean Sea Level (PSMSL)](https://psmsl.org/).

**How to use:**
- Enter ANY PSMSL station ID (1 to 2000+) - station names are automatically fetched!
- Or select from popular stations dropdown
- Compare 2-6 stations for optimal visualization
- The app works with all 1000+ PSMSL tide gauge stations globally
""")

# Station Selection
st.subheader("📍 Select Tide Gauge Stations")

st.info("""
💡 **This app works with ALL 1000+ PSMSL stations!** 
- Popular stations are pre-loaded in the dropdown below
- For ANY other station: Just enter the Station ID number directly
- Station names are **automatically fetched** from PSMSL data files
- Don't know the station ID? Check the [PSMSL Station Map](https://psmsl.org/data/obtaining/map.html)
""")

col1, col2 = st.columns([3, 1])

with col1:
    popular_options = {f"{name} (ID: {sid})": sid 
                      for sid, name in STATION_DATABASE.items()}
    
    selected_names = st.multiselect(
        "Search and select stations:",
        options=sorted(popular_options.keys()),
        default=["BREST (ID: 1)", "SAN FRANCISCO (ID: 12)", "NEWLYN (ID: 202)", "MONTAUK (ID: 519)", "BOSTON (ID: 1703)", "SYDNEY (FORT DENISON) (ID: 679)"],
        help="Select multiple stations to compare"
    )
    
    station_ids_to_fetch = [popular_options[name] for name in selected_names]
    
    # Advanced: custom IDs
    with st.expander("🔧 Advanced: Enter Custom Station IDs"):
        custom_input = st.text_input(
            "Additional Station IDs (comma-separated):",
            placeholder="e.g., 500, 1234, 1800",
            help="Add any PSMSL station ID"
        )
        
        if custom_input.strip():
            custom_ids = [sid.strip() for sid in custom_input.split(",") if sid.strip()]
            station_ids_to_fetch = list(dict.fromkeys(station_ids_to_fetch + custom_ids))

with col2:
    st.markdown("**Popular Regions:**")
    st.caption("Europe: 1, 202, 229")
    st.caption("Americas: 12, 22, 1214")
    st.caption("Asia: 367, 850, 1402")
    st.caption("Oceania: 656, 679, 680")
    st.info("💡 **Tip:** Select 2-6 stations for best comparison results, and also try adding Station ID's to see the trends at different locations")

# Options
col_a, col_b = st.columns(2)
with col_a:
    show_trends = st.checkbox("Show linear trends", value=True)
with col_b:
    show_rolling = st.checkbox("Show 12-month rolling averages", value=False)

# Fetch Data
if st.button("🔎 Fetch & Compare Data", type="primary", use_container_width=True):
    if not station_ids_to_fetch:
        st.error("❌ Please select at least one station.")
    else:
        data_dict = {}
        errors = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, station_id in enumerate(station_ids_to_fetch):
            try:
                status_text.text(f"Fetching {station_id}... ({idx+1}/{len(station_ids_to_fetch)})")
                
                df, station_name = fetch_rlr_monthly(station_id)
                
                if df is None or len(df) == 0:
                    raise ValueError("No data")
                
                slope, intercept, trend_series = compute_linear_trend(df)
                
                data_dict[station_id] = {
                    'df': df, 'name': station_name,
                    'trend': trend_series, 'slope': slope
                }
                
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                errors.append(f"Station {station_id}: {str(e)}")
            
            progress_bar.progress((idx + 1) / len(station_ids_to_fetch))
        
        status_text.empty()
        progress_bar.empty()
        
        # Show errors
        if errors:
            with st.expander(f"⚠️ {len(errors)} station(s) could not be fetched"):
                for error in errors:
                    st.warning(error)
        
        # Display results
        if data_dict:
            st.success(f"✅ Successfully loaded {len(data_dict)} station(s)!")
            
            # Summary cards
            st.subheader("📊 Sea Level Trend Summary")
            cols = st.columns(min(len(data_dict), 4))
            
            for idx, (station_id, info) in enumerate(data_dict.items()):
                with cols[idx % 4]:
                    trend_dir = "📈" if info['slope'] > 0 else "📉"
                    label = info['name'][:22] + "..." if len(info['name']) > 22 else info['name']
                    
                    st.metric(
                        label=f"{trend_dir} {label}",
                        value=f"{info['slope']:.2f} mm/yr",
                        delta=f"ID: {station_id}",
                        help=f"{info['name']} (ID: {station_id})"
                    )
            
            # Multi-station plot
            st.subheader("📈 Multi-Station Comparison")
            fig = plot_sea_level_multi(data_dict, show_trends, show_rolling)
            st.plotly_chart(fig, use_container_width=True)
            
            # Insights
            if len(data_dict) > 1:
                st.subheader("🔍 Key Insights")
                trends = [(info['name'], info['slope']) for info in data_dict.values()]
                trends_sorted = sorted(trends, key=lambda x: x[1], reverse=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Fastest Rising:**")
                    st.write(f"📈 {trends_sorted[0][0]}: **{trends_sorted[0][1]:.2f} mm/yr**")
                with col2:
                    st.markdown("**Slowest Rising:**")
                    st.write(f"📉 {trends_sorted[-1][0]}: **{trends_sorted[-1][1]:.2f} mm/yr**")
            
            # Individual analysis
            st.markdown("---")
            st.subheader("🔬 Detailed Station Analysis")
            
            selected = st.selectbox(
                "Select station:",
                options=list(data_dict.keys()),
                format_func=lambda x: f"{data_dict[x]['name']} (ID: {x})"
            )
            
            if selected:
                info = data_dict[selected]
                df_sel = info['df']
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Records", len(df_sel))
                with col2:
                    st.metric("Date Range", f"{df_sel.index.min().year}-{df_sel.index.max().year}")
                with col3:
                    valid = df_sel['msl_mm'].notna().sum()
                    st.metric("Valid Records", f"{valid} ({valid/len(df_sel)*100:.1f}%)")
                
                # Seasonal decomposition
                st.markdown("#### 🔎 Seasonal Decomposition")
                try:
                    df_interp = df_sel.interpolate()
                    if len(df_interp.dropna()) >= 24:
                        result = seasonal_decompose(df_interp["msl_mm"].dropna(), period=12, model="additive")
                        
                        st.line_chart(pd.DataFrame({
                            "Observed": result.observed, "Trend": result.trend,
                            "Seasonal": result.seasonal, "Residual": result.resid
                        }), height=400)
                    else:
                        st.info("⚠️ Need 24+ months for decomposition")
                except Exception as e:
                    st.error(f"Decomposition failed: {e}")
                
                # Download
                with st.expander("📋 View & Download Data"):
                    st.dataframe(df_sel.head(50), use_container_width=True)
                    
                    csv = df_sel.to_csv()
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv,
                        file_name=f"sea_level_{selected}_{info['name'].replace(' ', '_')}.csv",
                        mime="text/csv"
                    )
        else:
            st.error("❌ No data fetched. Check station IDs.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>Data:</strong> <a href='https://psmsl.org/' target='_blank'>PSMSL</a> | 
    RLR Monthly Mean Sea Level Data</p>
</div>
""", unsafe_allow_html=True)

