"""
imd_wis2_source.py
===================
Adapter for official IMD WIS2 surface observations (SYNOP).
"""

import urllib.request
import urllib.error
import ssl
import json
import warnings
from datetime import datetime, timezone, timedelta
import math
from typing import Optional

import pandas as pd

_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE

WIS2_BASE_URL = "https://wis2box.imd.gov.in/oapi/collections/urn:wmo:md:in-imd:surface-based-observations.synop/items"
FETCH_TIMEOUT_S = 10

def _calculate_rh(temp_c, dewpoint_c):
    if pd.isna(temp_c) or pd.isna(dewpoint_c):
        return None
    try:
        # August-Roche-Magnus approximation
        es = 6.112 * math.exp((17.67 * temp_c) / (temp_c + 243.5))
        e = 6.112 * math.exp((17.67 * dewpoint_c) / (dewpoint_c + 243.5))
        rh = (e / es) * 100
        return min(max(rh, 0.0), 100.0)
    except Exception:
        return None

def _check_wis2_availability() -> bool:
    try:
        url = f"{WIS2_BASE_URL}?f=json&limit=1"
        req = urllib.request.Request(url, headers={"User-Agent": "SkyGuardAI/3.0"})
        with urllib.request.urlopen(req, timeout=6, context=_SSL_CTX) as r:
            return r.status == 200
    except Exception:
        return False

IMD_WIS2_AVAILABLE: bool = _check_wis2_availability()

def _fetch_station_data(station: dict, hours: int) -> list:
    """Fetch recent data for a single station"""
    wmo_id = station.get("wmo_id")
    if not wmo_id:
        return []
        
    # WIS2 uses OGC API Features datetime filtering
    # Example: 2026-05-30T00:00:00Z/..
    start_time = datetime.utcnow() - timedelta(hours=hours)
    dt_str = f"{start_time.strftime('%Y-%m-%dT%H:%M:%SZ')}/.."
    
    url = f"{WIS2_BASE_URL}?wigos_station_identifier=0-20000-0-{wmo_id}&datetime={dt_str}&f=json&limit=10000"
    
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "SkyGuardAI/3.0"})
        with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT_S, context=_SSL_CTX) as r:
            raw = r.read()
        data = json.loads(raw)
        return data.get("features", [])
    except Exception as e:
        # warnings.warn(f"Failed to fetch WIS2 for {station['name']}: {e}")
        return []

def _parse_features(features: list, station: dict) -> list:
    # Group by reportTime
    grouped = {}
    for f in features:
        props = f.get("properties", {})
        report_time = props.get("reportTime")
        if not report_time: continue
        
        try:
            ts = pd.Timestamp(report_time).tz_convert(None) # Naive UTC
        except Exception:
            continue
            
        # Ignore future timestamps (forecast safety)
        if ts > pd.Timestamp.utcnow().tz_localize(None) + pd.Timedelta(hours=1):
            continue
            
        if ts not in grouped:
            grouped[ts] = {}
            
        name = props.get("name")
        val = props.get("value")
        if name and val is not None:
            grouped[ts][name] = float(val)

    # Normalize to SkyGuard Schema
    records = []
    for ts, vars_dict in grouped.items():
        temp = vars_dict.get("air_temperature")
        pressure = vars_dict.get("non_coordinate_pressure")
        if pressure is None:
            pressure = vars_dict.get("air_pressure_at_sea_level")
            
        dewpoint = vars_dict.get("dewpoint_temperature")
        humidity = vars_dict.get("relative_humidity")
        if humidity is None and temp is not None and dewpoint is not None:
            humidity = _calculate_rh(temp, dewpoint)
            
        if temp is None and pressure is None and humidity is None:
            continue
            
        records.append({
            "timestamp": ts,
            "station_id": station["station_id"],
            "name": station["name"],
            "lat": station["lat"],
            "lon": station["lon"],
            "temp": temp if temp is not None else float("nan"),
            "pressure": pressure if pressure is not None else float("nan"),
            "humidity": humidity if humidity is not None else float("nan"),
        })
    return records

def fetch_historical_observations(stations: list[dict], hours: int = 72) -> Optional[pd.DataFrame]:
    """Fetch historical WIS2 observations for the requested stations."""
    import concurrent.futures
    all_records = []
    
    def _fetch_and_parse(st):
        features = _fetch_station_data(st, hours=hours)
        return _parse_features(features, st)

    # Use ThreadPoolExecutor to prevent hanging (max 10 seconds total wait time)
    with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
        future_to_station = {executor.submit(_fetch_and_parse, st): st for st in stations if st.get("live_available", False)}
        
        # Only wait for a maximum of 15 seconds for the ENTIRE batch
        done, not_done = concurrent.futures.wait(
            future_to_station.keys(), 
            timeout=15.0, 
            return_when=concurrent.futures.ALL_COMPLETED
        )
        
        for future in done:
            try:
                records = future.result()
                if records:
                    all_records.extend(records)
            except Exception:
                pass
                
    if not all_records:
        return None
        
    df = pd.DataFrame(all_records)
    cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(hours=hours)
    df = df[df["timestamp"] >= cutoff].copy()
    
    if df.empty:
        return None
        
    return df.sort_values(["station_id", "timestamp"]).reset_index(drop=True)

def fetch_live_observations(stations: list[dict]) -> Optional[pd.DataFrame]:
    """Fetch a single latest snapshot of WIS2 observations."""
    all_records = []
    for st in stations:
        # 12 hours gives a safe buffer to find the latest synop report
        features = _fetch_station_data(st, hours=12)
        records = _parse_features(features, st)
        if records:
            # sort by timestamp and keep latest
            records.sort(key=lambda x: x["timestamp"])
            all_records.append(records[-1])
            
    if not all_records:
        return None
        
    df = pd.DataFrame(all_records)
    return df.sort_values("station_id").reset_index(drop=True)

def validate_live_data(df: pd.DataFrame, min_hours: int = 6, min_stations: int = 3) -> tuple[bool, str, pd.DataFrame]:
    if df is None or df.empty:
        return False, "Data fetch returned empty results.", df
        
    df_clean = df.copy()
    
    # 1. Drop duplicates (station_id + timestamp)
    df_clean = df_clean.drop_duplicates(subset=["station_id", "timestamp"], keep="last")
    
    # 2. Check required columns
    required_cols = ["timestamp", "station_id", "temp", "pressure", "humidity", "lat", "lon"]
    missing = [c for c in required_cols if c not in df_clean.columns]
    if missing:
        return False, f"Malformed records: missing columns {missing}", df_clean
        
    for col in ["temp", "pressure", "humidity", "lat", "lon"]:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
        
    # 3. Check spatial coverage
    n_stations = df_clean["station_id"].nunique()
    if n_stations < min_stations:
        return False, f"Insufficient spatial coverage: only {n_stations} stations reported.", df_clean
        
    # 4. Check temporal history
    ts_min = df_clean["timestamp"].min()
    ts_max = df_clean["timestamp"].max()
    history_hours = (ts_max - ts_min).total_seconds() / 3600.0
    
    # Snapshot mode handles this by checking if the UI padded the DataFrame
    # If the df spans at least min_hours, we pass.
    if history_hours < min_hours:
        return False, f"Insufficient History: got {history_hours:.1f}h of data, need {min_hours}h for rolling features.", df_clean
        
    # 5. Check staleness - max timestamp older than 24 hours
    now = pd.Timestamp.utcnow().tz_localize(None)
    stale_hours = (now - ts_max).total_seconds() / 3600.0
    if stale_hours > 24:
        return False, f"Stale observations: latest data is {stale_hours:.1f} hours old.", df_clean

    return True, "Data validated successfully.", df_clean
