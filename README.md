# Project Overview
This web application provides an interactive platform for visualizing and comparing sea level trends from multiple tide gauge stations worldwide. The application fetches real-time data from the Permanent Service for Mean Sea Level (PSMSL) and automatically calculates trends, enabling users to analyze climate change impacts on coastal regions.
Live Application: https://sea-level-monitor.streamlit.app/


## Key Features

### 1. Multi-Station Comparison
- Compare **2-10+ tide gauge stations** on a single interactive plot
- Automatic trend calculation showing rate of sea level rise (mm/year)
- Select from 50+ popular stations or enter any PSMSL station ID (1-2000+)

### 2. Automatic Data Fetching
- Real-time data from PSMSL's 1000+ global tide gauge network
- Station names fetched automatically - just enter the ID
- Robust fetching with retry logic and error handling

### 3. Interactive Visualizations
- **Zoom, pan, and hover** over charts for detailed information
- **Linear trend lines** showing calculated rate of change
- **12-month rolling averages** to smooth seasonal fluctuations
- **Seasonal decomposition** breaking data into trend, seasonal, and residual components

### 4. Data Analysis Tools
- View data quality metrics (records, date range, completeness)
- Compare fastest vs. slowest rising sea levels
- Download raw data as CSV for external analysis

---

## How to Use

### Quick Start
1. Visit: https://sea-level-monitor.streamlit.app/
2. Select stations from the dropdown (or enter Station IDs manually)
3. Click **"Fetch & Compare Data"**
4. Explore interactive charts and trends

### Popular Station IDs
- **1** - Brest, France
- **12** - San Francisco, USA
- **202** - Newlyn, UK
- **367** - Mumbai, India
- **679** - Sydney, Australia
- **850** - Singapore
- **1402** - Tokyo, Japan

### Understanding Results
- **Positive mm/yr** = Sea level rising
- **Trend line** = Long-term rate of change
- **Seasonal pattern** = Annual variations (thermal expansion, currents)

---

## Technical Details

**Data Source:** Permanent Service for Mean Sea Level (PSMSL)  
**Dataset:** RLR monthly mean sea level data  
**Coverage:** 1000+ stations globally, some dating back to 1807

**Technology Stack:**
- Streamlit (web framework)
- Pandas & NumPy (data processing)
- Plotly (interactive charts)
- Statsmodels (seasonal decomposition)

**Reliability Features:**
- 3 retry attempts per station with exponential backoff
- 30-second timeout and rate limiting
- Validates data before processing

---

## Local Installation

```bash
# Clone repository
git clone (https://github.com/GAYATRI-SIVANI-SUSARLA/Sea-Level-Monitoring).git
cd sea-level-monitoring

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app.py
```

**Requirements:**
```
streamlit
pandas
numpy
requests
statsmodels
plotly
```

---

## Example Use Cases

**Regional Comparison:** Select European stations (Brest, Newlyn, Amsterdam) to compare regional sea level trends

**Global Analysis:** Compare stations across continents to identify areas with the fastest sea level rise

**Long-term Trends:** Analyze century-long records to observe acceleration in recent decades

---

## Project Structure

```
sea-level-monitoring/
├── app.py              # Main application
├── requirements.txt    # Dependencies
└── README.md          # This file
```

---

**PSMSL Website:** https://psmsl.org/

---

## Contact

**Developer:** Gayatri Sivani Susarla  
**Institution:** Stony Brook University  

---
 
