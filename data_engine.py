"""
data_engine.py — SkyGuard AI Phase 2
======================================
Generates a realistic multi-station Indian AWS network dataset.

Key improvements over Phase 1:
  - Station individuality: each station has its own amplitude, phase, noise level
  - Coastal/arid/inland climate characteristics
  - Realistic multi-variable correlations
  - 10 injected fault scenarios covering all required anomaly types
  - Spatially coherent genuine weather event (staggered onset, different magnitudes)
  - Explicit ground-truth dataframe returned alongside sensor data
  - Ground truth is NEVER passed to the ML pipeline — used only for evaluation

Injected anomalies (ground truth):
  AWS-001 Delhi       h=14        Sensor Spike / Fault         (+15.5°C, neighbours normal)
  AWS-002 Mumbai      h=36-42     Genuine Weather Event        (monsoon surge, spatial coherent)
  AWS-003 Chennai     h=28-34     Frozen / Stuck Sensor        (all 3 vars stuck for 7h)
  AWS-004 Kolkata     h=48-60     Calibration Drift            (temp drifts +0.4°C/h)
  AWS-006 Jaipur      h=20-22     Communication Failure        (NaN readings)
  AWS-007 Hyderabad   h=55        Multivariate Inconsistency   (T+H+P physically inconsistent)
  AWS-010 Ahmedabad   h=30        Sensor Spike / Fault         (temperature drop -12°C)
  AWS-009 Patna       h=45        Sensor Spike / Fault         (humidity spike +35%)
  AWS-011 Lucknow     h=58-65     Calibration Drift            (pressure drifts +0.6 hPa/h)
  AWS-008 Pune        h=36-42     Genuine Weather Event        (same monsoon surge, weaker)
  AWS-005 Bhopal      —           Normal reference             (no faults injected)
"""

import json
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Station registry — each station has climate-specific parameters
#
# amplitude_temp:  ± degrees C in daily cycle (larger for arid/continental)
# phase_offset:    hours after 06:00 when temperature peaks (12–15 h typical)
# noise_temp:      σ of Gaussian sensor noise (°C)
# amplitude_hum:   ± % humidity swing in daily cycle
# climate:         used for narrative context
# ---------------------------------------------------------------------------
json_path = os.path.join(os.path.dirname(__file__), 'stations_india.json')
with open(json_path, 'r', encoding='utf-8') as f:
    STATIONS = json.load(f)


def generate_dataset(hours: int = 72, seed: int = 42):
    """
    Generate a synthetic multi-station weather dataset with ground truth.

    Parameters
    ----------
    hours : int — number of simulated hours (rows per station)
    seed  : int — NumPy random seed for reproducibility

    Returns
    -------
    sensor_df    : pd.DataFrame
        Raw sensor observations as an AWS would report them.
        Columns: timestamp, station_id, name, lat, lon, temp, humidity, pressure
        NaN where communication failure was injected.
        Anomalous values where faults were injected.
        NO ground-truth columns — the ML pipeline must detect anomalies.

    ground_truth_df : pd.DataFrame
        Hidden labels for evaluation ONLY. Never pass to the ML pipeline.
        Columns: timestamp, station_id, is_anomaly_gt, true_root_cause
    """
    rng = np.random.default_rng(seed)
    start = datetime(2026, 9, 1, 0, 0, 0)
    timestamps = [start + timedelta(hours=h) for h in range(hours)]

    records = []

    for stn_i, stn in enumerate(STATIONS):
        sid          = stn["station_id"]
        base_temp    = stn["base_temp"]
        base_hum     = stn["base_hum"]
        base_pres    = stn["base_pres"]
        amp_t        = stn["amplitude_temp"]
        phase        = stn["phase_offset"]   # hours after 06:00 when peak occurs
        noise_t      = stn["noise_temp"]
        amp_h        = stn["amplitude_hum"]
        noise_h      = stn["noise_hum"]
        noise_p      = stn["noise_pres"]

        # Per-station random stream (derived from seed + station index)
        st_rng = np.random.default_rng(seed + stn_i * 1000)

        # Slow multi-day pressure trend (random ±1 hPa drift over 72 h)
        pressure_trend_slope = st_rng.uniform(-0.015, 0.015)  # hPa/h

        for h, ts in enumerate(timestamps):
            hour_of_day = h % 24

            # ----------------------------------------------------------------
            # Temperature: daily sine cycle with station-specific amplitude and phase
            # Peak occurs at (6 + phase + 8) = 14-17:00 depending on station
            # ----------------------------------------------------------------
            temp = (base_temp
                    + amp_t * np.sin((hour_of_day - 6 - phase) * np.pi / 12)
                    + st_rng.normal(0, noise_t))

            # ----------------------------------------------------------------
            # Humidity: negatively correlated with temp cycle, station-specific amplitude
            # Also includes a random slow drift component
            # ----------------------------------------------------------------
            hum = (base_hum
                   - amp_h * np.sin((hour_of_day - 6 - phase) * np.pi / 12)
                   + st_rng.normal(0, noise_h))
            hum = float(np.clip(hum, 5, 100))

            # ----------------------------------------------------------------
            # Pressure: semi-diurnal variation + slow trend + noise
            # ----------------------------------------------------------------
            pres = (base_pres
                    + 1.2 * np.sin(hour_of_day * np.pi / 12)
                    + 0.4 * np.sin(hour_of_day * np.pi / 6)
                    + pressure_trend_slope * h
                    + st_rng.normal(0, noise_p))

            records.append({
                "timestamp":  ts,
                "station_id": sid,
                "name":       stn["name"],
                "lat":        stn["lat"],
                "lon":        stn["lon"],
                "temp":       round(float(temp), 2),
                "humidity":   round(float(hum),  2),
                "pressure":   round(float(pres), 2),
            })

    df = pd.DataFrame(records)

    # -----------------------------------------------------------------------
    # Ground truth: initially all Normal
    # -----------------------------------------------------------------------
    gt = df[["timestamp", "station_id"]].copy()
    gt["is_anomaly_gt"]  = False
    gt["true_root_cause"] = "Normal"

    # -----------------------------------------------------------------------
    # INJECT ANOMALIES
    # Sensor observations are modified; ground truth is updated in parallel.
    # -----------------------------------------------------------------------

    # ------------------------------------------------------------------
    # 1. 0-20000-0-42176 Delhi - Sensor Spike at hour 14
    # ------------------------------------------------------------------
    _inject(df, gt, "0-20000-0-42176", [timestamps[14]],
            temp_delta=+15.5,
            true_root_cause="Sensor Spike / Fault")

    # ------------------------------------------------------------------
    # 2. 0-20000-0-43278 to 0-20000-0-43303 Chennai - Genuine Monsoon Surge h=36-42
    #    Spatially coherent event across all Chennai stations.
    # ------------------------------------------------------------------
    for h in range(36, 43):
        for sid in ["0-20000-0-43278", "0-20000-0-43279", "0-20000-0-43277", "0-20000-0-43275", "0-20000-0-43303"]:
            _inject(df, gt, sid, [timestamps[h]],
                    temp_delta=-4.5, hum_delta=+18.0, pres_delta=-7.0,
                    true_root_cause="Genuine Weather Event (not a fault)")

    # Partial/weaker signal at 0-20000-0-43295 (Bengaluru)
    for h in range(37, 43):
        _inject(df, gt, "0-20000-0-43295", [timestamps[h]],
                hum_delta=+6.0, pres_delta=-2.0,
                true_root_cause=None)

    # ------------------------------------------------------------------
    # 3. 0-20000-0-43296 Bengaluru - Frozen Sensor h=28-34 (7 consecutive hours)
    # ------------------------------------------------------------------
    frozen_row = df[(df["station_id"] == "0-20000-0-43296") & (df["timestamp"] == timestamps[28])]
    if not frozen_row.empty:
        ft = float(frozen_row["temp"].iloc[0])
        fh = float(frozen_row["humidity"].iloc[0])
        fp = float(frozen_row["pressure"].iloc[0])
        for h in range(28, 35):
            mask = (df["station_id"] == "0-20000-0-43296") & (df["timestamp"] == timestamps[h])
            df.loc[mask, ["temp", "humidity", "pressure"]] = ft, fh, fp
            gt.loc[(gt["station_id"] == "0-20000-0-43296") & (gt["timestamp"] == timestamps[h]),
                   ["is_anomaly_gt", "true_root_cause"]] = True, "Frozen / Stuck Sensor"

    # ------------------------------------------------------------------
    # 4. 0-20000-0-42807 Kolkata - Temperature Calibration Drift h=48-60
    # ------------------------------------------------------------------
    for i, h in enumerate(range(48, 61)):
        _inject(df, gt, "0-20000-0-42807", [timestamps[h]],
                temp_delta=(i + 1) * 0.4,
                true_root_cause="Calibration Drift")

    # ------------------------------------------------------------------
    # 5. 0-20000-0-42348 Jaipur - Communication Failure h=20-22 (NaN)
    # ------------------------------------------------------------------
    for h in range(20, 23):
        mask = (df["station_id"] == "0-20000-0-42348") & (df["timestamp"] == timestamps[h])
        df.loc[mask, ["temp", "humidity", "pressure"]] = np.nan
        gt.loc[(gt["station_id"] == "0-20000-0-42348") & (gt["timestamp"] == timestamps[h]),
               ["is_anomaly_gt", "true_root_cause"]] = True, "Communication Failure"

    # ------------------------------------------------------------------
    # 6. 0-20000-0-43128 Hyderabad - Multivariate Inconsistency h=55
    # ------------------------------------------------------------------
    _inject(df, gt, "0-20000-0-43128", [timestamps[55]],
            temp_delta=+9.0, hum_delta=-32.0, pres_delta=-10.0,
            true_root_cause="Multivariate Inconsistency")

    # ------------------------------------------------------------------
    # 7. 0-20000-0-42647 Ahmedabad - Temperature Drop h=30
    # ------------------------------------------------------------------
    _inject(df, gt, "0-20000-0-42647", [timestamps[30]],
            temp_delta=-12.0,
            true_root_cause="Sensor Spike / Fault")

    # ------------------------------------------------------------------
    # 8. 0-20000-0-42971 Bhubaneswar - Humidity Spike h=45
    # ------------------------------------------------------------------
    _inject(df, gt, "0-20000-0-42971", [timestamps[45]],
            hum_delta=+35.0,
            true_root_cause="Sensor Spike / Fault")

    # ------------------------------------------------------------------
    # 9. 0-20000-0-43003 Pune - Pressure Calibration Drift h=58-65
    # ------------------------------------------------------------------
    for i, h in enumerate(range(58, 66)):
        _inject(df, gt, "0-20000-0-43003", [timestamps[h]],
                pres_delta=(i + 1) * 0.6,
                true_root_cause="Calibration Drift")

    # ------------------------------------------------------------------
    # 10. 0-20000-0-43057 Mumbai - Normal reference station (no faults)
    # ------------------------------------------------------------------

    # Sort both dataframes consistently
    df = df.sort_values(["timestamp", "station_id"]).reset_index(drop=True)
    gt = gt.sort_values(["timestamp", "station_id"]).reset_index(drop=True)

    return df, gt


def _inject(df: pd.DataFrame, gt: pd.DataFrame, sid: str, ts_list: list, *,
            temp_delta: float = 0.0, hum_delta: float = 0.0, pres_delta: float = 0.0,
            true_root_cause: str | None = None) -> None:
    """
    In-place helper: apply sensor-reading deltas and update ground truth.

    true_root_cause=None means the event is a partial/boundary signal
    and should not be included in ground-truth evaluation labels.
    """
    for ts in ts_list:
        mask = (df["station_id"] == sid) & (df["timestamp"] == ts)
        if temp_delta != 0.0:
            df.loc[mask, "temp"]     = df.loc[mask, "temp"]     + temp_delta
        if hum_delta != 0.0:
            df.loc[mask, "humidity"] = df.loc[mask, "humidity"] + hum_delta
            df.loc[mask, "humidity"] = df.loc[mask, "humidity"].clip(0, 100)
        if pres_delta != 0.0:
            df.loc[mask, "pressure"] = df.loc[mask, "pressure"] + pres_delta

        if true_root_cause is not None:
            gt_mask = (gt["station_id"] == sid) & (gt["timestamp"] == ts)
            gt.loc[gt_mask, "is_anomaly_gt"]  = True
            gt.loc[gt_mask, "true_root_cause"] = true_root_cause


def validate_registry(stations: list[dict]) -> None:
    """Validate that the station registry does not contain catastrophic errors."""
    invalid = []
    for st in stations:
        lat, lon = st.get("lat"), st.get("lon")
        if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
            invalid.append(st)
    if invalid:
        print(f"[WARNING] Registry contains {len(invalid)} stations with missing/invalid coordinates.")
        # DO NOT modify or fabricate coordinates. We simply report them and allow downstream functions
        # to handle or skip them explicitly.

validate_registry(STATIONS)

def neighbor_map(top_k: int = 4) -> dict:
    """
    Spatial proximity graph: each station -> list of nearest station IDs.

    Uses Euclidean distance on (lat, lon) degrees.
    Stations with missing coordinates are assigned an empty neighbor list.
    """
    nbrs = {}
    for stn in STATIONS:
        # If this station lacks coordinates, it gets no neighbors.
        if not isinstance(stn.get("lat"), (int, float)) or not isinstance(stn.get("lon"), (int, float)):
            nbrs[stn["station_id"]] = []
            continue
            
        distances = []
        for other in STATIONS:
            if other["station_id"] == stn["station_id"]:
                continue
            # If the OTHER station lacks coordinates, skip it as a potential neighbor.
            if not isinstance(other.get("lat"), (int, float)) or not isinstance(other.get("lon"), (int, float)):
                continue
                
            d = ((stn["lat"] - other["lat"]) ** 2 + (stn["lon"] - other["lon"]) ** 2) ** 0.5
            distances.append((other["station_id"], d))
            
        distances.sort(key=lambda x: x[1])
        nbrs[stn["station_id"]] = [s for s, _ in distances[:top_k]]
    return nbrs


def get_station_info() -> pd.DataFrame:
    """Return station metadata as a DataFrame (useful for UI)."""
    return pd.DataFrame([
        {k: v for k, v in stn.items()
         if k in ("station_id", "name", "lat", "lon", "climate", "base_temp", "base_hum", "base_pres")}
        for stn in STATIONS
    ])


def build_neighbor_map_from_df(sensor_df: pd.DataFrame, top_k: int = 4, max_radius: float = 0.5) -> dict:
    stations = sensor_df.groupby("station_id")[["lat", "lon"]].first().dropna().reset_index()
    nbrs: dict = {sid: [] for sid in sensor_df["station_id"].unique()}
    for _, row in stations.iterrows():
        sid = row["station_id"]
        dists = []
        for _, other in stations.iterrows():
            osid = other["station_id"]
            if osid == sid: continue
            d = ((row["lat"] - other["lat"])**2 + (row["lon"] - other["lon"])**2)**0.5
            if d <= max_radius:
                dists.append((osid, d))
        dists.sort(key=lambda x: x[1])
        nbrs[sid] = [x[0] for x in dists[:top_k]]
    return nbrs

