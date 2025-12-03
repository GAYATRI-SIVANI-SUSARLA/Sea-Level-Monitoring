# Project Overview
This web application provides an interactive platform for visualizing and comparing sea level trends from multiple tide gauge stations worldwide. The application fetches real-time data from the Permanent Service for Mean Sea Level (PSMSL) and automatically calculates trends, enabling users to analyze climate change impacts on coastal regions.
Live Application: https://sea-level-monitor.streamlit.app/


## Key Functions

### 1. Multi-Station Comparison
- **Compare up to 6 stations simultaneously** on a single interactive chart
- **Default stations pre-loaded:** Brest (France), San Francisco (USA), Newlyn (UK), Montauk (USA), Boston (USA), and Sydney (Australia)
- View sea level trends from different geographical locations side-by-side
- Each station displays with unique color coding for easy identification

### 2. Automatic Data Fetching
- **Real-time data retrieval** from PSMSL's global tide gauge network
- **Automatic station name detection** - enter any Station ID (1-2000+) and the app automatically fetches the station name from PSMSL data
- **Retry logic:** Each station gets 3 fetch attempts to ensure reliability
- Works with **all 1000+ PSMSL tide gauge stations** worldwide

### 3. Interactive Visualizations
- **Zoom, pan, and hover** over the chart to explore data details
- **Linear trend lines** showing calculated rate of sea level rise (mm/year)
- **12-month rolling averages** (optional) to smooth seasonal variations
- **Toggle controls** to show/hide trends and rolling averages as needed

### 4. Automated Trend Analysis
- **Automatic linear regression** calculates the rate of sea level change for each station
- Displays trend rate in millimeters per year (mm/yr)
- **Key insights section** identifies the fastest and slowest rising sea levels among selected stations
- Compare regional differences in sea level rise rates

### 5. Seasonal Decomposition Analysis
- Select any station for a detailed breakdown
- **Separates data into three components:**
  - **Trend:** Long-term directional change
  - **Seasonal:** Annual cyclical patterns (thermal expansion, ocean currents)
  - **Residual:** Random variations and noise
- Helps understand what drives sea level changes at each location

### 6. Data Quality Metrics
- View the total number of records for each station
- See date range coverage (some stations have data since 1807!)
- Check data completeness percentage
- Understand the reliability of trend calculations

### 7. CSV Data Export
- Download raw monthly data for any station
- Files automatically named with Station ID and name
- Use exported data for further analysis in Excel, Python, R, or other tools

---

## How to Use

### Quick Start (Using Default Stations)

1. Visit: **https://sea-level-monitor.streamlit.app/**
2. The app loads with **6 default stations** already selected:
   - Brest, France (ID: 1)
   - San Francisco, USA (ID: 12)
   - Newlyn, UK (ID: 202)
   - Montauk, USA (ID: 519)
   - Boston, USA (ID: 1703)
   - Sydney, Australia (ID: 679)
3. Click **"Fetch & Compare Data"** button
4. View the results:
   - **Summary cards** showing trend rate for each station
   - **Interactive comparison chart** with all 6 stations
   - **Key insights** comparing fastest vs. slowest rising

### Selecting Different Stations

**Method 1 - Using Dropdown (Easy):**
1. Click on the station selection dropdown
2. Remove default stations or keep some and add more
3. Type to search (e.g., "Mumbai", "San Francisco", "Boston")
4. Select 2-6 stations for optimal comparison
5. Click "Fetch & Compare Data."

**Method 2 - Enter Station IDs (Advanced):**
1. Click **"Advanced: Enter Custom Station IDs"** expander
2. Enter station IDs separated by commas (e.g., `500, 1234, 1800`)
3. Station names will be fetched automatically
4. Click "Fetch & Compare Data."

### Popular Station IDs to Try
- **367** - Mumbai, India
- **1402** - Tokyo, Japan
- **1214** - Buenos Aires, Argentina
- **500** - Colombo, Sri Lanka
- **680** - Fremantle, Australia
- **22** - Key West, Florida (USA)

### Understanding the Results

**Trend Rate (mm/year):**
- Positive value = Sea level is rising
- Example: **+3.63 mm/yr** means sea level rises 3.63 millimeters per year
- Typical range: 1.0 to 4.0 mm/yr for most stations
- Global average: ~3.4 mm/yr

**Chart Elements:**
- **Solid lines** = Actual monthly measurements
- **Dotted lines** = Calculated linear trends
- **Dashed lines** = 12-month rolling averages (if enabled)

---

## Key Features Explained


### Robust Data Fetching
The app uses retry logic to ensure reliability:
- **3 attempts** per station if initial fetch fails
- **Exponential backoff** (1s, 2s, 3s delays between retries)
- **30-second timeout** to handle slow connections
- **0.5-second delay** between stations to prevent server overload

### Multi-Station Visualization
- Compare 2-10+ stations on one chart
- Each station gets a unique color
- Legend allows showing/hiding individual stations
- Hover shows exact values for all stations at any time point

---

## Technical Information

### Data Source
- **Provider:** Permanent Service for Mean Sea Level (PSMSL)
- **Dataset:** Revised Local Reference (RLR) monthly mean sea level
- **Coverage:** 1000+ tide gauge stations globally
- **Quality:** Research-grade data used in climate studies

### Statistical Methods
- **Linear Trend:** Least-squares regression
- **Rolling Average:** 12-month centered moving average
- **Seasonal Decomposition:** Additive STL decomposition

### Technology Stack
- **Streamlit** - Web application framework
- **Pandas & NumPy** - Data processing
- **Plotly** - Interactive visualizations
- **Statsmodels** - Statistical analysis
- **Python Requests** - Data fetching with retry logic

---

## Installation for Local Use

### Requirements
- Python 3.8 or higher
- Internet connection

### Setup
```bash
# Clone repository
git clone https://github.com/GAYATRI-SIVANI-SUSARLA/Sea-Level-Monitoring.git
cd Sea-Level-Monitoring

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app.py
```

### Dependencies (`requirements.txt`)
```
streamlit
pandas
numpy
requests
statsmodels
plotly
```

---

## Project Structure
```
Sea-Level-Monitoring/
├── app.py              # Main application (final version)
├── requirements.txt    # Python dependencies
└── README.md          # This documentation
```

---

## Limitations

- **Data availability:** Not all station IDs have monthly data available
- **Missing data:** Some stations have gaps shown as breaks in lines
- **Minimum data requirement:** Seasonal decomposition needs 24+ months of continuous data
- **Network dependency:** Requires an internet connection to fetch PSMSL data

---

## Troubleshooting

### Issue: Station name shows as "STATION [ID]."
**Solution:** The station data file may not have a clear name in the header. The app still works and shows data correctly.

### Issue: Some stations fail to load
**Solution:** Check the error message in the expandable warning section. The station may not exist or have no RLR monthly data.

### Issue: Loading takes time
**Solution:** The app fetches data from PSMSL servers in real-time. With 6 default stations, expect 10-15 seconds. The progress bar shows the status.

---


**PSMSL Website:** https://psmsl.org/


## Contact

**Developer:** Gayatri Sivani Susarla  
**University:** Stony Brook University
**GitHub:** https://github.com/GAYATRI-SIVANI-SUSARLA  
**Repository:** https://github.com/GAYATRI-SIVANI-SUSARLA/Sea-Level-Monitoring
**LinkedIn:** https://www.linkedin.com/in/gayatri-sivani-susarla-975856263/


## Acknowledgments

- **PSMSL** for providing free access to global tide gauge data
- **Streamlit** for the excellent web framework

---

