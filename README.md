# SkyGuard AI — SIH 2026 PS 26073

AI/ML anomaly detection for Indian Automatic Weather Stations.

**Live at:** [Streamlit Community Cloud](https://skyguard.streamlit.app)

## Features
- Real-time Open-Meteo weather data for 154 WMO stations across India
- Frozen Isolation Forest anomaly detector (contamination=0.01)
- Spatial + temporal feature engineering
- SHAP-based explainability
- Sensor health scoring
- Interactive Folium map with India boundary

## Local Setup
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Architecture
| File | Role |
|---|---|
| `app.py` | Streamlit UI |
| `anomaly_engine.py` | Isolation Forest + root-cause + SHAP |
| `data_engine.py` | Station registry + feature engineering |
| `live_aws_source.py` | Open-Meteo live data fetcher |
| `data_source.py` | Simulated data generator |
| `imd_wis2_source.py` | IMD WIS2 direct feed (fallback) |
| `boundary/` | India state GeoJSON |
