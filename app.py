"""
app.py — SkyGuard AI Phase 3 Tactical Command Center
======================================================
Streamlit dashboard. All ML intelligence lives in data_engine.py and anomaly_engine.py.
This file is UI only.

Data source modes:
  1. Simulated Data       — existing Phase 2 simulator, full ground-truth evaluation
  2. Live IMD/AWS         — Open-Meteo real observations (72 h history, refreshable)
  3. Historical CSV       — user-uploaded CSV, same pipeline

Map:
  - OSM as visual basemap
  - Official India boundary from boundary/india_states.geojson (udit-001/india-maps-data)
    district-level GeoJSON, 760 features, includes Ladakh as separate UT (st_code=38)
    and J&K (st_code=01) per the 2019 Reorganisation Act.
  - If boundary file is missing, Bhuvan WMS is attempted as fallback.

Tabs:
  1. Live Map
  2. Time Series & Anomalies
  3. Explainability (SHAP)
  4. Sensor Health & Reliability
  5. Alert Log
  6. Model Performance / Operational Metrics
"""

import os
import time
import json
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import plotly.graph_objects as go

from data_engine import STATIONS, neighbor_map, build_neighbor_map_from_df
from data_source import SimulatedDataSource, ActualAWSDataSource, validate_csv_bytes
from anomaly_engine import (
    run_pipeline, explain_with_shap, sensor_health,
    evaluate_pipeline, FEATURE_COLS, CONTAMINATION, N_ESTIMATORS,
    ROLLING_WINDOW, SPIKE_DIFF_THRESH, SPATIAL_DEV_THRESH,
    SPATIAL_AGREE_THRESH, MULTIVAR_MAG_THRESH, DRIFT_SLOPE_THRESH,
)

warnings.filterwarnings("ignore")

st.set_page_config(page_title="SkyGuard AI - Tactical Command Center", layout="wide")

st.markdown("""
    <style>
    .stAppDeployButton {display: none;}
    .badge-sim    { display:inline-block; padding:3px 10px; border-radius:20px;
                    background:#1a3a1a; color:#2ecc71; font-weight:600;
                    font-size:12px; border:1.5px solid #2ecc71; margin-left:8px; }
    .badge-live   { display:inline-block; padding:3px 10px; border-radius:20px;
                    background:#1a1a3a; color:#9b59b6; font-weight:600;
                    font-size:12px; border:1.5px solid #9b59b6; margin-left:8px; }
    .badge-csv    { display:inline-block; padding:3px 10px; border-radius:20px;
                    background:#1a2a3a; color:#3498db; font-weight:600;
                    font-size:12px; border:1.5px solid #3498db; margin-left:8px; }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
ROOT_CAUSE_COLOR = {
    "Normal":                              "#2ecc71",
    "Sensor Spike / Fault":               "#e74c3c",
    "Frozen / Stuck Sensor":              "#9b59b6",
    "Calibration Drift":                  "#f39c12",
    "Communication Failure":              "#7f8c8d",
    "Multivariate Inconsistency":         "#c0392b",
    "Genuine Weather Event (not a fault)":"#3498db",
    "Unclassified Anomaly":               "#e67e22",
    "Provider Data Gap (Open-Meteo)": "#7f8c8d",
}

# ---------------------------------------------------------------------------
# India boundary GeoJSON — loaded once, cached
# ---------------------------------------------------------------------------
BOUNDARY_FILE = os.path.join(os.path.dirname(__file__), "boundary", "india_states.geojson")
BOUNDARY_FALLBACK_WMS = "https://bhuvan-vec2.nrsc.gov.in/bhuvan/wms"


@st.cache_data(show_spinner=False)
def _load_boundary_geojson():
    """Load the district-level India GeoJSON (760 features, Ladakh separate UT)."""
    if os.path.exists(BOUNDARY_FILE):
        with open(BOUNDARY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


boundary_geojson = _load_boundary_geojson()

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "data_mode" not in st.session_state:
    st.session_state.data_mode = "simulated"
if "sim_pos" not in st.session_state:
    st.session_state.sim_pos = 71
if "live_df" not in st.session_state:
    st.session_state.live_df = None
if "live_nbrs" not in st.session_state:
    st.session_state.live_nbrs = None
if "live_error" not in st.session_state:
    st.session_state.live_error = None
if "csv_cache" not in st.session_state:
    st.session_state.csv_cache = None

# ---------------------------------------------------------------------------
# Cached simulated pipeline
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _load_simulated_pipeline():
    src = SimulatedDataSource()
    sensor_df = src.get_sensor_df()
    nbrs = src.get_neighbor_map()
    processed, _ = run_pipeline(sensor_df, nbrs)
    return processed, nbrs, src.get_ground_truth()


def _make_cache_key(mode: str, df) -> str:
    """Lightweight hashable key that encodes data identity for cache discrimination.

    The underscore prefix on _df parameters tells Streamlit NOT to hash the
    DataFrame itself (too slow / unhashable for large frames). We pass this
    string key alongside _df so Streamlit uses it as the cache discriminator.
    The key changes whenever the data source or its content changes.
    """
    try:
        ts_max = str(df["timestamp"].max())[:16]   # e.g. "2026-09-06 17:30"
    except Exception:
        ts_max = "unknown"
    return f"{mode}|rows={len(df)}|ts={ts_max}"


@st.cache_data(show_spinner=False)
def _load_shap_cached(_df, source_key: str):
    """Compute SHAP explanations keyed by source_key, not _df itself."""
    if _df is None or _df.empty or "anomaly_score_pct" not in _df.columns:
        return {}
    return explain_with_shap(_df)


@st.cache_data(show_spinner=False)
def _load_eval_cached(_df, _gt, source_key: str):
    """Compute pipeline evaluation keyed by source_key."""
    return evaluate_pipeline(_df, _gt)


# ===========================================================================
# SIDEBAR — Mode selector
# ===========================================================================
st.sidebar.markdown("## 📡 Data Source")
mode_choice = st.sidebar.radio(
    "Select source",
    ["🔬 Simulated Data", "🌐 Live Meteorological Data", "📂 Historical CSV"],
    key="mode_radio",
    index=0,
)
new_mode = {"🔬 Simulated Data": "simulated",
            "🌐 Live Meteorological Data":   "live",
            "📂 Historical CSV": "csv"}[mode_choice]

if new_mode != st.session_state.data_mode:
    st.session_state.data_mode = new_mode
    st.rerun()

MODE = st.session_state.data_mode
st.sidebar.markdown("---")

# ===========================================================================
# Load data based on mode
# ===========================================================================

HAS_GT = False

if MODE == "simulated":
    with st.spinner("Running SkyGuard AI pipeline…"):
        full_df, NEIGHBORS, gt_df = _load_simulated_pipeline()
    HAS_GT = True
    all_timestamps = sorted(full_df["timestamp"].unique())

    st.sidebar.header("Simulation Playback")
    auto_play = st.sidebar.checkbox("Auto-play live stream", value=False)
    sim_pos = st.sidebar.slider(
        "Simulated network time",
        min_value=24, max_value=len(all_timestamps) - 1,
        value=st.session_state.sim_pos, format="hour %d",
    )
    st.session_state.sim_pos = sim_pos
    current_time = all_timestamps[sim_pos]
    st.sidebar.markdown(f"**Current simulated time:** `{pd.Timestamp(current_time)}`")

elif MODE == "live":
    # -----------------------------------------------------------------------
    # LIVE IMD/AWS — Open-Meteo real observations
    # -----------------------------------------------------------------------


    st.sidebar.header("Live Observation Settings")
    provider = st.sidebar.selectbox(
        "Data Provider", 
        ["IMD WIS2", "Live provider fallback", "IMD AWS API (Direct)"]
    )
    
    if provider == "IMD AWS API (Direct)":
        st.sidebar.error("IMD AWS API requires IP whitelisting and credentials. Currently unauthorized.")
        st.error("Cannot connect to direct IMD AWS API. Please select IMD WIS2 or Live provider fallback.")
        st.stop()
        
    hours_back = st.sidebar.slider("Hours of history", 6, 72, 24, step=6)
    refresh_sec = st.sidebar.selectbox("Live refresh interval", [5, 10, 30, 60], index=1, format_func=lambda x: f"{x} seconds")
    
    st.sidebar.markdown("---")
    if provider == "IMD WIS2":
        from imd_wis2_source import fetch_historical_observations, fetch_live_observations, validate_live_data, IMD_WIS2_AVAILABLE
        st.sidebar.caption(
            "**SOURCE:** India Meteorological Department — WIS2\n\n"
            "**DATA TYPE:** WMO/SYNOP surface observations\n\n"
            "**PIPELINE:** Same SkyGuard AI anomaly-detection pipeline"
        )
        if not IMD_WIS2_AVAILABLE:
            st.error("IMD WIS2 API is unreachable. Please check internet connection.")
            st.stop()
    else:
        from live_aws_source import fetch_historical_observations, fetch_live_observations, validate_live_data, OPEN_METEO_AVAILABLE
        st.sidebar.caption(
            "**SOURCE:** Live provider fallback\n\n"
            "**DATA TYPE:** Real-time Meteorological Data\n\n"
            "*(Proxy for WMO/SYNOP)*"
        )
        if not OPEN_METEO_AVAILABLE:
            st.error("Fallback API is unreachable. Please check internet connection.")
            st.stop()

    # ------------------------------------------------------------------
    # Live refresh architecture — @st.fragment prevents full-page reruns
    # on every auto-refresh cycle. The fragment auto-runs every 90 seconds;
    # it calls st.rerun() (full page rerun) ONLY when new data arrives.
    # Manual buttons inside the fragment work identically to before.
    # ------------------------------------------------------------------

    import datetime as dt
    @st.fragment(run_every=f"{refresh_sec}s")
    def _live_data_refresher():
        """Isolated fragment: fetch live data and update session state."""
        col_ref, col_snap = st.columns(2)
        do_refresh_frag  = col_ref.button("🔄 Refresh Now", key="btn_refresh")
        do_snapshot_frag = col_snap.button("📸 Snapshot Only", key="btn_snapshot")
        
        now = dt.datetime.now()
        last_check = st.session_state.get("last_live_check_time")
        
        last_refresh = now.strftime("%Y-%m-%d %H:%M:%S")
        if st.session_state.live_df is not None and not st.session_state.live_df.empty:
            last_source = st.session_state.live_df['timestamp'].max().strftime("%Y-%m-%d %H:%M:%S")
        else:
            last_source = "Never"
            
        st.caption(f"**Last source update:** {last_source} UTC &nbsp;|&nbsp; **Last app refresh:** {last_refresh} Local")

        need_fetch_frag = (
            st.session_state.live_df is None
            or do_refresh_frag
        )

        empty_df = pd.DataFrame(columns=[
            "timestamp", "station_id", "name", "lat", "lon", 
            "temp", "humidity", "pressure", "root_cause", "is_anomaly", "severity"
        ])

        if need_fetch_frag or do_snapshot_frag:
            st.session_state["last_live_check_time"] = now # PREVENT DOUBLE FETCH
            with st.spinner("Connecting to live observations (this may take a few seconds)..."):
                import time
                max_retries = 3
                retry_delay = 2
                
                for attempt in range(max_retries):
                    if do_snapshot_frag:
                        live_obs = fetch_live_observations(STATIONS)
                        if live_obs is not None and not live_obs.empty:
                            frames = [live_obs.copy() for _ in range(10)]
                            for idx_f, frame in enumerate(frames):
                                frame["timestamp"] = (
                                    live_obs["timestamp"] - pd.Timedelta(hours=(9 - idx_f))
                                )
                            live_full = (
                                pd.concat(frames)
                                .sort_values(["station_id", "timestamp"])
                                .reset_index(drop=True)
                            )
                        else:
                            live_full = None
                    else:
                        live_full = fetch_historical_observations(STATIONS, hours=hours_back)
                    
                    if live_full is not None and not live_full.empty:
                        break  # success!
                    elif attempt < max_retries - 1:
                        time.sleep(retry_delay)

            if live_full is None or live_full.empty:
                st.session_state.live_error = "SOURCE UNAVAILABLE: The provider returned no data for the requested window."
                st.session_state.live_df = empty_df
                st.session_state.live_nbrs = {}
                st.rerun()
                return

            # Phase 7: Lightweight validation
            is_valid, val_msg, live_full = validate_live_data(live_full, min_hours=6)
            if not is_valid:
                st.session_state.live_error = val_msg
                st.session_state.live_df = empty_df
                st.session_state.live_nbrs = {}
                st.rerun()
                return

            nbrs = build_neighbor_map_from_df(live_full, top_k=4)
            with st.spinner("Running anomaly detection pipeline..."):
                processed, _ = run_pipeline(live_full, nbrs)

            prev_ts = (
                st.session_state.live_df["timestamp"].max()
                if st.session_state.live_df is not None and not st.session_state.live_df.empty
                else None
            )
            new_ts = processed["timestamp"].max()

            st.session_state.live_df   = processed
            st.session_state.live_nbrs = nbrs
            st.session_state.live_error = None

            if prev_ts != new_ts or need_fetch_frag:
                st.rerun()

        else:
            # Auto-refresh cycle: check if upstream data changed
            if last_check is None or (now - last_check).total_seconds() >= refresh_sec:
                st.session_state["last_live_check_time"] = now
                live_full = fetch_historical_observations(STATIONS, hours=hours_back)
                
                if live_full is None or live_full.empty:
                    # Do not overwrite live_error silently here to avoid auto-crashing the UI,
                    # just keep existing data if a transient failure occurs.
                    return
                
                prev_ts = (
                    st.session_state.live_df["timestamp"].max()
                    if st.session_state.live_df is not None and not st.session_state.live_df.empty
                    else None
                )
                new_ts = live_full["timestamp"].max()
                if prev_ts != new_ts:
                    is_valid, val_msg, live_full = validate_live_data(live_full, min_hours=6)
                    if not is_valid:
                        return # silently ignore transient failure in background
                        
                    nbrs = build_neighbor_map_from_df(live_full, top_k=4)
                    processed, _ = run_pipeline(live_full, nbrs)
                    st.session_state.live_df   = processed
                    st.session_state.live_nbrs = nbrs
                    st.session_state.live_error = None
                    st.rerun()   # new data arrived — one full-page rerun

    _live_data_refresher()

    if st.session_state.live_error:
        st.error(f"⚠️ **Live Data Temporarily Unavailable:** {st.session_state.live_error}")
        # Do NOT st.stop() - allow the rest of the application to remain alive

    if st.session_state.live_df is None or st.session_state.live_df.empty:
        st.warning("⏳ **Connecting to live observations...** The system is retrying automatically in the background. Please wait.")
        full_df = pd.DataFrame(columns=[
            "timestamp", "station_id", "name", "lat", "lon", 
            "temp", "humidity", "pressure", "root_cause", "is_anomaly", "severity"
        ])
        NEIGHBORS = {}
        current_time = pd.Timestamp.now()
        all_timestamps = []
    else:
        full_df   = st.session_state.live_df.copy()
        if MODE == "live" and provider == "Open-Meteo (Fallback)":
            full_df.loc[full_df["root_cause"] == "Communication Failure", "root_cause"] = "Provider Data Gap (Open-Meteo)"
        NEIGHBORS = st.session_state.live_nbrs
        current_time   = full_df["timestamp"].max()
        all_timestamps = sorted(full_df["timestamp"].unique())
        
    gt_df = None


else:
    # -----------------------------------------------------------------------
    # HISTORICAL CSV
    # -----------------------------------------------------------------------
    st.sidebar.header("Upload Historical CSV")
    uploaded = st.sidebar.file_uploader(
        "Upload CSV file",
        type=["csv"],
        help="Required: timestamp, station_id, temperature, pressure, humidity\nOptional: lat, lon, name",
    )

    if uploaded is None:
        st.title("SkyGuard AI: Tactical Command Center")
        st.markdown(
            '<span class="badge-csv">● HISTORICAL CSV</span>',
            unsafe_allow_html=True,
        )
        st.info(
            "### Upload a CSV file to analyse historical AWS data\n\n"
            "**Required columns:** `timestamp`, `station_id`, `temperature`, `pressure`, `humidity`\n\n"
            "**Optional columns:** `lat`, `lon`, `name`\n\n"
            "The uploaded data is passed through the **same Isolation Forest pipeline** "
            "as simulated data. No separate model is created.\n\n"
            "```\n"
            "timestamp,station_id,temperature,pressure,humidity,lat,lon\n"
            "2026-09-01 00:00:00,AWS-001,32.5,1005.2,68.0,28.61,77.21\n"
            "```"
        )
        st.stop()

    raw_bytes = uploaded.read()
    cache_key = f"{uploaded.name}_{len(raw_bytes)}"

    if st.session_state.csv_cache is None or st.session_state.csv_cache.get("key") != cache_key:
        actual_src, _ = validate_csv_bytes(raw_bytes)

        if not actual_src.is_valid:
            st.sidebar.error("Validation failed:")
            for e in actual_src.validation_errors:
                st.sidebar.error(f"• {e}")
            st.stop()

        for w in actual_src.validation_warnings:
            st.sidebar.warning(f"⚠ {w}")

        with st.spinner(f"Running pipeline on {uploaded.name}…"):
            sensor_df = actual_src.get_sensor_df()
            nbrs = actual_src.get_neighbor_map()
            processed, _ = run_pipeline(sensor_df, nbrs)

        st.session_state.csv_cache = {
            "key": cache_key, "processed": processed, "nbrs": nbrs,
        }
        st.sidebar.success(
            f"✅ {len(sensor_df)} rows, "
            f"{sensor_df['station_id'].nunique()} station(s)"
        )

    cache = st.session_state.csv_cache
    full_df   = cache["processed"]
    NEIGHBORS = cache["nbrs"]
    gt_df     = None
    current_time   = full_df["timestamp"].max()
    all_timestamps = sorted(full_df["timestamp"].unique())

# ---------------------------------------------------------------------------
# Sidebar — common filters
# ---------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.header("Network Filters")
view_mode = st.sidebar.radio(
    "Map Display Mode:",
    ["Global Network", "Anomalies Only", "Faults Only (exclude genuine events)"],
)
st.sidebar.markdown("---")
st.sidebar.subheader("Detection Settings")
cont_str = f"{CONTAMINATION:.1%}" if isinstance(CONTAMINATION, float) else str(CONTAMINATION)
st.sidebar.text(
    f"Model: Isolation Forest\n"
    f"Features: {len(FEATURE_COLS)} engineered\n"
    f"Contamination: {cont_str}\n"
    f"Trees: {N_ESTIMATORS}\n"
    f"+ SHAP (surrogate RF)\n"
    f"+ Evidence-based root cause\n"
    f"+ Separate severity scoring"
)

# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------
window_df = full_df[full_df["timestamp"] <= current_time]
snapshot = (
    window_df.sort_values("timestamp")
    .groupby("station_id").tail(1)
    .reset_index(drop=True)
)
def is_valid_coordinate(lat, lon):
    import pandas as pd, numpy as np
    try:
        lat = float(lat)
        lon = float(lon)
        return (not pd.isna(lat) and not pd.isna(lon) and 
                np.isfinite(lat) and np.isfinite(lon) and 
                -90 <= lat <= 90 and -180 <= lon <= 180)
    except (ValueError, TypeError):
        return False

# Filter snapshot to only include valid coordinates for authoritative count & map
valid_coords = snapshot.apply(lambda r: is_valid_coordinate(r["lat"], r["lon"]) if "lat" in r and "lon" in r else False, axis=1)
snapshot_valid = snapshot.loc[valid_coords]
invalid_count = len(snapshot) - len(snapshot_valid)
if invalid_count > 0:
    st.warning(f"⚠ {invalid_count} stations have missing/invalid coordinates and are excluded from the active nodes count and map.")

if "root_cause" in snapshot_valid.columns and not snapshot_valid.empty:
    if view_mode == "Anomalies Only":
        snapshot_display = snapshot_valid[snapshot_valid["root_cause"] != "Normal"]
    elif view_mode == "Faults Only (exclude genuine events)":
        snapshot_display = snapshot_valid[
            (snapshot_valid["root_cause"] != "Normal") &
            (snapshot_valid["root_cause"] != "Genuine Weather Event (not a fault)")
        ]
    else:
        snapshot_display = snapshot_valid
else:
    snapshot_display = snapshot_valid

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
if MODE == "live":
    if st.session_state.live_df is None or st.session_state.live_df.empty:
        badge_html = '<span class="badge-live" style="background:#3a1a1a;color:#e74c3c;border-color:#e74c3c;">● LIVE UNAVAILABLE</span>'
    elif provider == "Live provider fallback":
        badge_html = '<span class="badge-live" style="background:#1a2a3a;color:#f39c12;border-color:#f39c12;">● FALLBACK ACTIVE</span>'
    else:
        badge_html = '<span class="badge-live">● IMD/WIS2</span>'
elif MODE == "csv":
    badge_html = '<span class="badge-csv">● HISTORICAL CSV</span>'
else:
    badge_html = '<span class="badge-sim">● SIMULATED</span>' 

st.title("SkyGuard AI: Tactical Command Center")
st.markdown(
    f"AI/ML anomaly detection for Automatic Weather Stations — SIH 2026 #26073 &nbsp;"
    f"**DATA SOURCE** {badge_html}",
    unsafe_allow_html=True,
)

if MODE == "live":
    _n_live_st  = full_df['station_id'].nunique() if not full_df.empty and 'station_id' in full_df.columns else 0
    _n_live_ts  = len(all_timestamps)
    st.info(
        f"🌐 **Live observations via {provider}** — "
        f"fetched at `{current_time}` UTC | "
        f"{_n_live_st} stations | "
        f"{_n_live_ts} hourly snapshots | "
        f"Fields: temperature_2m, relative_humidity_2m, surface_pressure"
    )



# ---------------------------------------------------------------------------
# Top Metrics
# ---------------------------------------------------------------------------
_has_root_cause = "root_cause" in window_df.columns and not window_df.empty

total_nodes   = len(snapshot_valid)  # Active nodes still based on current map snapshot
flagged_total = int((window_df["root_cause"] != "Normal").sum()) if _has_root_cause else 0
genuine_total = int((window_df["root_cause"] == "Genuine Weather Event (not a fault)").sum()) if _has_root_cause else 0
faults_total  = int(window_df["is_anomaly"].sum()) if _has_root_cause and "is_anomaly" in window_df.columns else 0
critical_total  = int((window_df.get("severity", pd.Series(dtype=str)) == "CRITICAL").sum()) if _has_root_cause else 0

# For live mode: show total anomalies across ALL history (not just latest snapshot)
if MODE == "live" and "is_anomaly" in full_df.columns and not full_df.empty:
    total_live_faults = int(full_df["is_anomaly"].sum())
    total_live_genuine = int((full_df["root_cause"] == "Genuine Weather Event (not a fault)").sum()) if "root_cause" in full_df.columns else 0
    total_live_flagged = total_live_faults + total_live_genuine
    if total_live_flagged > 0:
        st.warning(
            f"🚨 **{total_live_flagged} events flagged** across the last {hours_back}h of live data "
            f"({total_live_faults} sensor faults, {total_live_genuine} genuine weather events). "
            f"Select an anomalous station in the **Time Series & Anomalies** tab — stations with anomalies are listed first and marked with ⚠."
        )
    else:
        st.success("✅ No anomalies detected in the current live data window.")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Active Nodes", total_nodes)
m2.metric("Total Flagged", flagged_total)
m3.metric("Sensor Faults", faults_total,
          delta=f"{faults_total} need attention" if faults_total else "all clear",
          delta_color="inverse" if faults_total else "off")
m4.metric("Genuine Weather", genuine_total)
m5.metric("CRITICAL Severity", critical_total,
          delta_color="inverse" if critical_total > 0 else "off")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
perf_label = "Model Performance" if HAS_GT else "Operational Metrics"
tabs = st.tabs([
    "Live Map", "Time Series & Anomalies", "Explainability",
    "Sensor Health & Reliability", "Alert Log", perf_label,
])

# ================================================================ TAB 1: MAP
with tabs[0]:
    st.subheader("Geospatial Telemetry")
    map_col, log_col = st.columns([2, 1])

    with map_col:
        # ---- Base map (OSM, visual basemap only) ----
        m = folium.Map(
            location=[22.6, 80.0], zoom_start=5,
            tiles="OpenStreetMap",
            attr="Basemap: © OpenStreetMap contributors",
        )

        # ---- Official India Boundary Overlay ----
        # Source: udit-001/india-maps-data (github.com/udit-001/india-maps-data)
        # District-level GeoJSON, 760 features.
        # J&K = st_code "01" (23 district features)
        # Ladakh = st_code "38" (3 district features: Leh, Kargil, Zanskar)
        # Reflects the J&K Reorganisation Act, 2019 (two separate Union Territories).
        # This geometry is NOT the OSM basemap political lines.
        # It is rendered as a separate overlay on top of OSM.

        if boundary_geojson:
            # Define a colour per UT/state for the boundary fill
            # J&K and Ladakh get highlighted so the boundary is clearly visible
            UT_HIGHLIGHT = {
                "Jammu and Kashmir": "#2980b9",
                "Ladakh":            "#8e44ad",
            }

            def _state_style(feature):
                st_nm = feature["properties"].get("st_nm", "")
                if st_nm in UT_HIGHLIGHT:
                    return {
                        "fillColor": UT_HIGHLIGHT[st_nm],
                        "color": "#ffffff",
                        "weight": 1.0,
                        "fillOpacity": 0.18,
                    }
                return {
                    "fillColor": "#27ae60",
                    "color": "#ffffff",
                    "weight": 0.5,
                    "fillOpacity": 0.04,
                }

            def _state_highlight(feature):
                return {"weight": 2.0, "color": "#f1c40f", "fillOpacity": 0.25}

            folium.GeoJson(
                boundary_geojson,
                name="India Official Boundary (district-level)",
                style_function=_state_style,
                highlight_function=_state_highlight,
                tooltip=folium.GeoJsonTooltip(
                    fields=["st_nm", "district"],
                    aliases=["State/UT:", "District:"],
                    localize=True,
                ),
            ).add_to(m)

            boundary_source_note = (
                "**India boundary:** District-level GeoJSON — "
                "[udit-001/india-maps-data](https://github.com/udit-001/india-maps-data) | "
                "J&K and Ladakh shown as separate Union Territories per the J&K Reorganisation Act 2019 | "
                "**Basemap:** © OpenStreetMap contributors"
            )
        else:
            # Fallback: Bhuvan WMS
            try:
                folium.WmsTileLayer(
                    url=BOUNDARY_FALLBACK_WMS,
                    name="India Boundary (Bhuvan/NRSC fallback)",
                    layers="india_img:India_Boundary",
                    fmt="image/png",
                    transparent=True,
                    version="1.1.1",
                    attr="India boundary: Bhuvan/NRSC-ISRO | Basemap: © OpenStreetMap contributors",
                    overlay=True,
                    control=True,
                    show=True,
                    opacity=0.85,
                ).add_to(m)
            except Exception:
                pass

            boundary_source_note = (
                "⚠ **boundary/india_states.geojson not found** — "
                "Bhuvan WMS fallback attempted. "
                "Run `python probe_boundary.py` to download the official boundary file. | "
                "**Basemap:** © OpenStreetMap contributors"
            )

        # ---- Station markers ----
        map_df = snapshot_display
        
        # Map center relies ONLY on valid coordinates
        if not map_df.empty:
            center_lat, center_lon = map_df["lat"].mean(), map_df["lon"].mean()
            m.location = [center_lat, center_lon]

        from folium.plugins import MarkerCluster
        marker_cluster = MarkerCluster(name="Weather Stations").add_to(m)

        for _, row in map_df.iterrows():
            color = ROOT_CAUSE_COLOR.get(row["root_cause"], "#2ecc71")
            sev   = row.get("severity", "LOW")
            conf  = row.get("anomaly_score_pct", 0)
            cc    = row.get("classification_confidence", 0)
            st_type = row.get("station_type", "SYNOP")
            provider = row.get("data_provider", "WMO/WIS2")
            
            popup = (
                f"<div style='min-width: 240px; font-family:sans-serif; font-size:12px'>"
                f"<b>{row['name']}</b><br>"
                f"<b>ID:</b> {row['station_id']}<br>"
                f"<b>Type:</b> {st_type} | <b>Provider:</b> {provider}<br>"
                f"<b>Live Feed:</b> {'Yes' if row.get('live_available', True) else 'No (Historical Only)'}<br>"
                f"<hr style='margin: 8px 0;'>"
                f"Root cause: <b style='color:{color}'>{row['root_cause']}</b><br>"
                f"Severity: <b>{sev}</b><br>"
                f"Temp: {row['temp']:.1f} °C (est: {row['temp_corrected']:.1f})<br>"
                f"Humidity: {row['humidity']:.0f}%"
                f"</div>"
            )
            radius = 10 if not row["is_anomaly"] else (14 if sev in ["CRITICAL", "HIGH"] else 11)
            
            folium.CircleMarker(
                location=[float(row["lat"]), float(row["lon"])],
                radius=radius,
                color=color, fill=True, fill_color=color, fill_opacity=0.85,
                popup=folium.Popup(popup, max_width=300),
                tooltip=f"{row['name']} ({st_type}) | {row['root_cause']} | Sev: {sev}",
            ).add_to(marker_cluster)

        folium.LayerControl(collapsed=False).add_to(m)
        st_folium(m, width="100%", height=540, key="map", returned_objects=[])
        st.caption(boundary_source_note)

    with log_col:
        st.markdown("**Legend**")
        for k, v in ROOT_CAUSE_COLOR.items():
            st.markdown(
                f"<span style='color:{v}'>&#9679;</span> {k}",
                unsafe_allow_html=True,
            )
        st.markdown("---")
        st.markdown("**Snapshot — current time**")
        disp = ["name", "root_cause", "severity", "anomaly_score_pct"]
        _snap_cols = [c for c in disp if c in snapshot_display.columns]
        _snap_disp = snapshot_display[_snap_cols] if _snap_cols else snapshot_display
        if "anomaly_score_pct" in _snap_disp.columns and not _snap_disp.empty:
            _snap_disp = _snap_disp.sort_values("anomaly_score_pct", ascending=False)
        st.dataframe(_snap_disp, use_container_width=True, hide_index=True)

# ============================================================= TAB 2: TIME SERIES
with tabs[1]:
    st.subheader("Per-Station Time Series with Anomaly Overlay")
    
    # Use station_id natively to avoid name collisions
    station_options = full_df[["station_id", "name"]].drop_duplicates().sort_values("name")

    # Count anomalies per station so user can find flagged stations easily
    if "is_anomaly" in full_df.columns:
        anom_counts = full_df[full_df["is_anomaly"]].groupby("station_id").size().reset_index(name="n_anom")
        station_options = station_options.merge(anom_counts, on="station_id", how="left")
        station_options["n_anom"] = station_options["n_anom"].fillna(0).astype(int)
        # Put stations with anomalies first
        station_options = station_options.sort_values(["n_anom", "name"], ascending=[False, True])
        n_flagged_stations = (station_options["n_anom"] > 0).sum()
        if n_flagged_stations > 0:
            st.info("Stations that have detected anomalies are listed first in the dropdown below.")
    else:
        station_options["n_anom"] = 0
        n_flagged_stations = 0

    station_dict = dict(zip(station_options["station_id"], station_options["name"]))
    anom_dict    = dict(zip(station_options["station_id"], station_options["n_anom"])) if "n_anom" in station_options.columns else {}

    def format_station(sid):
        name  = station_dict.get(sid, "Unknown")
        n_anom = anom_dict.get(sid, 0)
        flag  = f" ⚠ {n_anom} anomal{'ies' if n_anom != 1 else 'y'}" if n_anom > 0 else ""
        return f"{name} ({sid}){flag}"

    sel_sid = st.selectbox("Select station", station_options["station_id"].tolist(), format_func=format_station)

    sub = full_df[
        (full_df["station_id"] == sel_sid) & (full_df["timestamp"] <= current_time)
    ]
    param = st.radio("Parameter", ["temp", "humidity", "pressure"], horizontal=True)
    label = {"temp": "Temperature (°C)", "humidity": "RH (%)", "pressure": "Pressure (hPa)"}[param]

    fig = go.Figure()
    if sub.empty:
        st.warning("No observations available for selected period.")
    else:
        fault_count = ((sub["is_anomaly"] == True) & (sub["root_cause"] != "Genuine Weather Event (not a fault)")).sum()
        genuine_count = (sub["root_cause"] == "Genuine Weather Event (not a fault)").sum()
        if fault_count == 0 and genuine_count == 0:
            st.info("No detected anomalies in selected period.")

    # 1. Plot raw
    fig.add_trace(go.Scatter(
        x=sub["timestamp"], y=sub[param],
        mode="lines", name="Raw Reading", line=dict(color="#3498db", width=2),
        hovertemplate="%{x}<br>Raw: %{y}<extra></extra>"
    ))
    
    # 2. Plot corrected
    if f"{param}_corrected" in sub.columns:
        fig.add_trace(go.Scatter(
            x=sub["timestamp"], y=sub[f"{param}_corrected"],
            mode="lines", name="AI Corrected Reading",
            line=dict(color="#2ecc71", dash="dash", width=2),
            hovertemplate="%{x}<br>Corrected: %{y}<extra></extra>"
        ))
        
    # 3. Plot Genuine Weather Events
    genuine_sub = sub[sub["root_cause"] == "Genuine Weather Event (not a fault)"]
    if not genuine_sub.empty:
        fig.add_trace(go.Scatter(
            x=genuine_sub["timestamp"], y=genuine_sub[param],
            mode="markers", name="Genuine Weather Event",
            marker=dict(color="#f1c40f", size=14, symbol="star", line=dict(width=1, color="black")),
            text=[
                f"Corrected: {corr:.2f}<br>Genuine Weather Event"
                for corr in genuine_sub.get(f"{param}_corrected", genuine_sub[param])
            ],
            hovertemplate="%{x}<br>Raw: %{y}<br>%{text}<extra></extra>"
        ))

    # 4. Plot Flagged Faults
    anom_sub = sub[(sub["is_anomaly"] == True) & (sub["root_cause"] != "Genuine Weather Event (not a fault)")]
    if not anom_sub.empty:
        sev_col = anom_sub.get("severity", pd.Series(["MEDIUM"] * len(anom_sub)))
        fig.add_trace(go.Scatter(
            x=anom_sub["timestamp"], y=anom_sub[param],
            mode="markers", name="Flagged Fault",
            marker=dict(
                color=[ROOT_CAUSE_COLOR.get(rc, "#e74c3c") for rc in anom_sub["root_cause"]],
                size=[14 if s in ("CRITICAL", "HIGH") else 12 for s in sev_col],
                symbol="x",
                line=dict(width=2, color="black")
            ),
            text=[
                f"Corrected: {corr:.2f}<br>{rc} | Sev: {sv}<br>Score: {sc:.0f}%"
                for corr, rc, sv, sc in zip(
                    anom_sub.get(f"{param}_corrected", anom_sub[param]),
                    anom_sub["root_cause"],
                    anom_sub.get("severity", [""] * len(anom_sub)),
                    anom_sub.get("anomaly_score_pct", [0.0] * len(anom_sub))
                )
            ],
            hovertemplate="%{x}<br>Raw: %{y}<br>%{text}<extra></extra>",
        ))

    fig.update_layout(
        height=450, margin=dict(l=0, r=0, t=50, b=0),
        xaxis_title="Time (UTC)", yaxis_title=label,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    st.plotly_chart(fig, use_container_width=True)

with tabs[2]:
    st.subheader("Explainable AI — Why was this flagged?")
    st.info(
        "**Method:** A surrogate Random Forest is trained to reproduce the Isolation "
        "Forest's anomaly labels. SHAP TreeExplainer is applied to the surrogate RF. "
        "The explanation answers: *Which features most strongly pushed this observation "
        "toward being classified as anomalous?*"
    )
    shap_expl = _load_shap_cached(full_df, _make_cache_key(MODE, full_df))
    flagged_idx = list(shap_expl.keys())

    if not flagged_idx:
        st.info("No anomalies to explain in the current dataset.")
    else:
        flagged = full_df.loc[flagged_idx]
        if "anomaly_score_pct" in flagged.columns and not flagged.empty:
            flagged = flagged.sort_values("anomaly_score_pct", ascending=False)
        choice = st.selectbox(
            "Select event",
            flagged.index,
            format_func=lambda i: (
                f"{full_df.loc[i,'name']} @ {full_df.loc[i,'timestamp']} | "
                f"{full_df.loc[i,'root_cause']} | "
                f"Sev: {full_df.loc[i,'severity']} | "
                f"Score: {full_df.loc[i,'anomaly_score_pct']:.0f}%"
            ),
        )
        row   = full_df.loc[choice]
        pairs = shap_expl[choice]
        ev    = row.get("evidence", {})

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Station:** {row['name']}")
            st.markdown(f"**Timestamp:** {row['timestamp']}")
            st.markdown(f"**Root cause:** {row['root_cause']}")
            st.markdown(
                f"**Anomaly score:** {row['anomaly_score_pct']:.1f}%  \n"
                f"*(normalised IF decision function)*"
            )
            st.markdown(
                f"**Classification confidence:** {row['classification_confidence']:.1f}%  \n"
                f"*(evidence-strength score — not a probability)*"
            )
            st.markdown(f"**Severity:** {row['severity']}")
            st.markdown(
                f"**Raw temp:** {row['temp']:.1f}°C → "
                f"**Corrected:** {row['temp_corrected']:.1f}°C"
            )
            if ev:
                st.markdown("**Evidence signals:**")
                for k, v in ev.items():
                    st.markdown(f"- `{k}`: `{round(v,3) if isinstance(v,float) else v}`")
        with c2:
            st.markdown("**SHAP contributions (surrogate RF):**")
            st.caption("Red = pushed toward anomaly. Green = pushed toward normal.")
            bar = go.Figure(go.Bar(
                x=[v for _, v in pairs], y=[f for f, _ in pairs],
                orientation="h",
                marker_color=["#e74c3c" if v > 0 else "#2ecc71" for _, v in pairs],
            ))
            bar.update_layout(
                height=300, margin=dict(l=10, r=10, t=20, b=10),
                xaxis_title="SHAP contribution",
                plot_bgcolor="#0f1117", paper_bgcolor="#0f1117",
                font=dict(color="#ffffff"),
            )
            st.plotly_chart(bar, use_container_width=True)

# ====================================================== TAB 4: SENSOR HEALTH
with tabs[3]:
    st.subheader("Sensor Health & Predictive Maintenance")
    st.caption(
        "Health score explicitly ignores Genuine Weather Events. "
        "Trends compare current 48h window vs previous 48h. "
        "Predictive Maintenance Risk combines trend, faults, freezes, drift, and missing data."
    )
    health = sensor_health(full_df[full_df["timestamp"] <= current_time])

    # Health metrics are now correctly computed directly in anomaly_engine.py
    icons  = {"Healthy": "🟢", "Degraded": "🟡", "Warning": "🟠", "Critical": "🔴"}
    
    # Display cards for each station
    cols = st.columns(4)
    for i, (_, r) in enumerate(health.iterrows()):
        with cols[i % 4]:
            # Metric delta shows trend
            delta = None
            if r.get('trend') == 'DETERIORATING': delta = f"{r.get('health_score') - r.get('previous_health'):.1f} (Deteriorating)"
            elif r.get('trend') == 'IMPROVING': delta = f"{r.get('health_score') - r.get('previous_health'):.1f} (Improving)"
            else: delta = "Unchanged"
            
            st.metric(
                f"{icons.get(r.get('status', 'Healthy'), '⚪')} {r['name']}",
                f"{r.get('health_score', 100)}%",
                delta=delta,
                delta_color="inverse" if r.get('trend') == 'DETERIORATING' else "normal",
                help=(f"Genuine Events: {r.get('genuine_events', 0)} | " f"Fault Anomalies: {r.get('faults', 0)} | " f"Missing: {r.get('missing_readings', 0)} | " f"Dominant fault: {r.get('dominant_fault', '—')}"),
            )
            
            risk = r.get('maintenance_risk', 'LOW')
            r_color = {"CRITICAL": "red", "HIGH": "orange", "MEDIUM": "yellow", "LOW": "green"}[risk]
            st.markdown(f"**Risk:** :{r_color}[{risk}]")
            st.caption(f"_{r.get('recommendation', 'No action')}_")
            st.markdown("---")

    avail = [c for c in ["name", "readings", "faults", "genuine_events", "anomaly_rate",
                         "missing_readings", "health_score", "trend", "status", "dominant_fault", "maintenance_risk"]
             if c in health.columns]
    
    st.markdown("### Network Health View")
    st.dataframe(health[avail], use_container_width=True, hide_index=True)

# ======================================================== TAB 5: ALERT LOG
with tabs[4]:
    st.subheader("Live Event Log")
    log = window_df[window_df["root_cause"] != "Normal"].sort_values("timestamp", ascending=False).copy()
    action_map = {
        "Sensor Spike / Fault":                "Dispatch field technician; exclude readings from archive.",
        "Frozen / Stuck Sensor":               "Restart data logger firmware / replace sensor module.",
        "Calibration Drift":                   "Schedule recalibration against WMO reference instrument.",
        "Communication Failure":               "Check network link and power supply to station.",
        "Multivariate Inconsistency":          "Inspect all three sensor modules; cross-check with neighbours.",
        "Genuine Weather Event (not a fault)": "No action — issue advisory to disaster management cell.",
        "Unclassified Anomaly":                "Manual review required; flag for expert inspection.",
    }
    log["suggested_action"] = log["root_cause"].map(action_map).fillna("Manual review recommended.")
    log_cols = ["timestamp","name","root_cause","severity",
                "anomaly_score_pct","classification_confidence",
                "temp","temp_corrected","suggested_action"]
    disp_log = log[[c for c in log_cols if c in log.columns]].rename(columns={
        "name": "station", "temp": "raw_temp", "temp_corrected": "corrected_temp",
        "anomaly_score_pct": "anomaly_score_%",
        "classification_confidence": "class_confidence_%",
    })
    st.dataframe(disp_log, use_container_width=True, hide_index=True, height=480)
    st.download_button(
        "Download alert log (CSV)", log.to_csv(index=False).encode(),
        "skyguard_alert_log.csv", "text/csv",
    )

# ================================================== TAB 6: PERFORMANCE / OPERATIONAL
with tabs[5]:
    if not HAS_GT:
        st.info("📊 **Operational Metrics**\n\n"
                "Evaluation metrics (Precision, Recall, F1, FPR, etc.) require **Ground Truth** labels, which are not available in live unlabelled data streams.\n\n"
                "To view our academic benchmark metrics, switch the Data Source to **Simulated Data**.")
    
    if HAS_GT:
        st.subheader("Model Performance & Evaluation")
        st.caption(
            "Metrics vs simulator's hidden ground truth — "
            "ground truth was NEVER passed to the ML model."
        )
        eval_results = _load_eval_cached(full_df, gt_df, _make_cache_key(MODE, full_df))
        if "error" in eval_results:
            st.error(eval_results["error"])
        else:
            det = eval_results["anomaly_detection"]
            summ = eval_results["summary"]
            st.markdown("### Binary Anomaly Detection")
            d1,d2,d3,d4,d5,d6 = st.columns(6)
            d1.metric("Total Obs.", summ["total_observations"])
            d2.metric("GT Anomalies", summ["n_anomalies_gt"])
            d3.metric("Detected", summ["n_detected"])
            d4.metric("True Positives", det["TP"])
            d5.metric("False Alarms", det["FP"])
            d6.metric("Missed", det["FN"])
            p1,p2,p3,p4,p5 = st.columns(5)
            p1.metric("Precision", f"{det['precision']:.3f}")
            p2.metric("Recall",    f"{det['recall']:.3f}")
            p3.metric("F1 Score",  f"{det['F1']:.3f}")
            p4.metric("FPR",       f"{det['FPR']:.3f}")
            p5.metric("FNR",       f"{det['FNR']:.3f}")
            st.markdown("---")
            st.markdown("### Per Root-Cause Performance")
            rc_data = eval_results.get("root_cause", {})
            if rc_data:
                rc_rows = [
                    {"Anomaly Type": cat, "GT Count": m["support"],
                     "TP": m["TP"], "FP": m["FP"], "FN": m["FN"],
                     "Precision": m["precision"], "Recall": m["recall"], "F1": m["F1"],
                     "FPR": m.get("FPR", 0.0), "FNR": m.get("FNR", 0.0)}
                    for cat, m in rc_data.items()
                ]
                st.dataframe(pd.DataFrame(rc_rows), use_container_width=True, hide_index=True)
            cm = eval_results.get("confusion_matrix")
            if cm is not None and not cm.empty:
                st.markdown("---")
                st.markdown("### Confusion Matrix")
                st.caption("Rows = Predicted, Columns = True label")
                fig_cm = go.Figure(go.Heatmap(
                    z=cm.values.tolist(), x=list(cm.columns), y=list(cm.index),
                    colorscale="Blues", text=cm.values.tolist(),
                    texttemplate="%{text}", showscale=True,
                ))
                fig_cm.update_layout(
                    height=400, margin=dict(l=10,r=10,t=30,b=10),
                    xaxis_title="True Root Cause", yaxis_title="Predicted Root Cause",
                    plot_bgcolor="#0f1117", paper_bgcolor="#0f1117",
                    font=dict(color="#ffffff"),
                )
                st.plotly_chart(fig_cm, use_container_width=True)
            st.markdown("---")
            st.markdown("### Model Configuration")
            cfg = {
                "Model": "Isolation Forest (unsupervised)", "Trees": N_ESTIMATORS,
                "Contamination": f"{CONTAMINATION:.1%}" if isinstance(CONTAMINATION, float) else str(CONTAMINATION), "Features": len(FEATURE_COLS),
                "Root-Cause Logic": "Evidence rule engine (7 rules)",
                "Spike Diff Threshold": f"{SPIKE_DIFF_THRESH} °C/h",
                "Spatial Dev Threshold": f"{SPATIAL_DEV_THRESH} °C",
                "Spatial Agree Threshold": f"{SPATIAL_AGREE_THRESH:.0%}",
                "Multivar Mag Threshold": MULTIVAR_MAG_THRESH,
                "Drift Slope Threshold": DRIFT_SLOPE_THRESH,
            }
            st.dataframe(
                pd.DataFrame([(k, str(v)) for k, v in cfg.items()], columns=["Parameter", "Value"]),
                use_container_width=True, hide_index=True,
            )
            st.markdown("### Feature List")
            st.dataframe(
                pd.DataFrame(enumerate(FEATURE_COLS, 1), columns=["#", "Feature"]),
                use_container_width=True, hide_index=True,
            )
    else:
        # ---- LIVE / CSV mode: Operational Metrics only ----
        st.subheader("Operational Metrics")
        if MODE == "live":
            st.info(
                "**No ground-truth labels are available for live observations.** "
                "Precision, Recall, and F1 are not shown — they require known labels. "
                "The metrics below reflect what the pipeline observed and detected "
                "in the live Open-Meteo data."
            )
        else:
            st.info(
                "**No ground-truth labels in the uploaded CSV.** "
                "Showing operational metrics only."
            )

        # Only render metrics if the ML pipeline actually ran on this data
        has_ml_data = (
            not full_df.empty
            and "is_anomaly" in full_df.columns
            and "root_cause" in full_df.columns
            and "anomaly_score_pct" in full_df.columns
        )

        if not has_ml_data:
            st.warning(
                "No live observations available — operational metrics cannot be computed. "
                "Please wait for the provider to return data, or press **Refresh Now**."
            )
        else:
            total_obs   = len(full_df)
            n_anom      = int(full_df["is_anomaly"].sum())
            anom_rate   = round(n_anom / total_obs * 100, 2) if total_obs else 0
            n_stations  = full_df["station_id"].nunique()
            n_genuine   = int((full_df["root_cause"] == "Genuine Weather Event (not a fault)").sum())
            n_comm_fail = int((full_df["root_cause"] == "Communication Failure").sum())
            latest_det  = full_df[full_df["is_anomaly"]]["timestamp"].max() if n_anom else None

            o1,o2,o3,o4 = st.columns(4)
            o1.metric("Total Observations", total_obs)
            o2.metric("Anomalies Detected", n_anom)
            o3.metric("Anomaly Rate", f"{anom_rate}%")
            o4.metric("Stations Monitored", n_stations)
            o5,o6,o7,o8 = st.columns(4)
            o5.metric("Genuine Weather Events", n_genuine)
            o6.metric("Communication Failures", n_comm_fail)
            o7.metric("CRITICAL Detections", int((full_df.get("severity", pd.Series(dtype=str)) == "CRITICAL").sum()))
            o8.metric("Latest Anomaly Time", str(pd.Timestamp(latest_det)) if latest_det and pd.notna(latest_det) else "—")

            st.markdown("---")
            st.markdown("### Breakdown by Root Cause")
            rc_counts = full_df[full_df["is_anomaly"]]["root_cause"].value_counts().reset_index()
            rc_counts.columns = ["Root Cause", "Count"]
            st.dataframe(rc_counts, use_container_width=True, hide_index=True)

            st.markdown("### Breakdown by Severity")
            sev_counts = full_df[full_df["is_anomaly"]]["severity"].value_counts().reset_index()
            sev_counts.columns = ["Severity", "Count"]
            st.dataframe(sev_counts, use_container_width=True, hide_index=True)

            ts_min, ts_max = full_df["timestamp"].min(), full_df["timestamp"].max()
            hrs = (ts_max - ts_min).total_seconds() / 3600
            st.markdown("---")
            st.markdown(f"**Time range:** `{ts_min}` to `{ts_max}` ({hrs:.1f} h)")

        st.markdown("---")
        st.markdown("### Data Provenance & Transparency")
        
        if MODE == "simulated":
            st.markdown("**SCENARIO:** Controlled labelled simulation/evaluation")
            st.markdown("**SOURCE:** Real WMO/WIS2 station registry mapped with **synthetically injected faults**.")
            st.markdown("**GROUND TRUTH:** Fully available for ML evaluation (Precision, Recall, F1).")
        elif MODE == "live":
            st.markdown("**SCENARIO:** Live operational monitoring")
            if "provider" in locals() and provider == "IMD WIS2":
                st.markdown("**SOURCE:** Live observation (IMD WIS2 surface observations).")
            else:
                st.markdown("**SOURCE:** Live observation (Fallback provider active).")
            st.markdown("**GROUND TRUTH:** Ground truth unavailable — operational monitoring mode.")
        else:
            st.markdown("**SCENARIO:** Historical CSV playback")
            st.markdown("**SOURCE:** User-uploaded file.")
            st.markdown("**GROUND TRUTH:** Ground truth unavailable.")

        _ts_max_str    = str(full_df["timestamp"].max()) if not full_df.empty and "timestamp" in full_df.columns else "N/A"
        _n_stations    = full_df["station_id"].nunique() if not full_df.empty and "station_id" in full_df.columns else 0
        _total_obs     = len(full_df)
        st.markdown(f"**LAST TIMESTAMP:** {_ts_max_str} UTC")
        st.markdown(f"**STATIONS:** {_n_stations} | **OBSERVATIONS:** {_total_obs}")
        st.markdown("**PIPELINE:** Frozen Causal + Isolation Forest Architecture")
        st.caption("Note: Stations are verified WMO/WIS2 nodes (SYNOP), but may not be strictly classified as automated hardware (AWS) without secondary verification.")

# ---- Auto-play (simulated only) ----
if MODE == "simulated" and locals().get("auto_play", False) and locals().get("sim_pos", 0) < len(all_timestamps) - 1:
    time.sleep(1.2)
    st.session_state.sim_pos = sim_pos + 1
    st.rerun()
