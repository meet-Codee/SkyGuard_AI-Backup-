"""
live_aws_source.py — SkyGuard AI Phase 3B
==========================================
Live weather observation fetcher using Open-Meteo API.

WHY Open-Meteo, not IMD direct API:
  - Official IMD API (api.imd.gov.in) requires registration + IP whitelisting;
    the /observation endpoint returns HTTP 401 without credentials.
  - Bhuvan/NRSC WMS timed out from this environment.
  - Open-Meteo (open-meteo.com) is a free, open-source, no-key-required
    meteorological API that sources data from NWP models (ERA5, GFS, ECMWF)
    which ingest WMO/SYNOP surface observations globally including Indian AWS.
  - Batch endpoint supports up to 50 lat/lon pairs in a single request.
  - Fields verified: temperature_2m (°C), relative_humidity_2m (%),
    surface_pressure (hPa), time (ISO-8601 local).

BUG FIXES (2026-09-22):
  - CRITICAL: Stations with lat=None/lon=None were included in URL strings,
    producing literal "None" in the query which caused HTTP 400 Bad Request.
    Fix: filter to only valid numeric coordinates before any URL construction.
  - CRITICAL: fetch_historical_observations() requested hourly data but the
    parser searched for resp["current"] which doesn't exist in hourly responses.
    The parser now correctly reads resp["hourly"]["time"] / ["temperature_2m"].
  - CRITICAL: If len(responses) != len(stations) the function returned None.
    This is fragile for batch requests. Fix: zip by index with fallback.
  - ADDED: safe batching in chunks of <=50 stations (Open-Meteo documented limit).
  - ADDED: explicit coordinate validation before request construction.

This module is the ONLY place that touches the network for live data.
All downstream code (feature engineering, Isolation Forest, root-cause
classification, SHAP, sensor health) is data-source-agnostic.

Public API:
    fetch_live_observations(stations)             -> pd.DataFrame | None
    fetch_historical_observations(stations, hours) -> pd.DataFrame | None
    OPEN_METEO_AVAILABLE                          -> bool (checked at import time)
"""

from __future__ import annotations

import math
import urllib.request
import urllib.error
import ssl
import json
import warnings
from datetime import datetime, timezone, timedelta
from typing import Optional

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# SSL context (ignore cert errors for some government proxy environments)
# ---------------------------------------------------------------------------
_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE

OPEN_METEO_BASE = "https://api.open-meteo.com/v1/forecast"

# Open-Meteo documented batch limit per single request
_BATCH_SIZE = 50

# ---------------------------------------------------------------------------
# Verified field mapping
# ---------------------------------------------------------------------------
# Open-Meteo field        → SkyGuard internal column
OM_FIELD_MAP = {
    "temperature_2m":       "temp",       # °C, 2 m above ground
    "relative_humidity_2m": "humidity",   # %, 2 m above ground
    "surface_pressure":     "pressure",   # hPa, surface level
}
OM_CURRENT_FIELDS = ",".join(OM_FIELD_MAP.keys())
OM_HOURLY_FIELDS  = ",".join(OM_FIELD_MAP.keys())

FETCH_TIMEOUT_S = 20   # seconds per individual batch request


# ---------------------------------------------------------------------------
# Coordinate validation helper
# ---------------------------------------------------------------------------
def _is_valid_coord(lat, lon) -> bool:
    """Return True iff lat and lon are finite numbers in valid geographic range."""
    try:
        lat_f = float(lat)
        lon_f = float(lon)
        return (
            math.isfinite(lat_f) and math.isfinite(lon_f)
            and -90.0 <= lat_f <= 90.0
            and -180.0 <= lon_f <= 180.0
        )
    except (TypeError, ValueError):
        return False


def _filter_valid_stations(stations: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Split station list into (valid, invalid) based on coordinate validity.
    Logs a warning for each excluded station.
    """
    valid, invalid = [], []
    for s in stations:
        if _is_valid_coord(s.get("lat"), s.get("lon")):
            valid.append(s)
        else:
            invalid.append(s)
            warnings.warn(
                f"Excluding station {s.get('station_id')} ({s.get('name')}) "
                f"— invalid coordinates lat={s.get('lat')} lon={s.get('lon')}"
            )
    return valid, invalid


# ---------------------------------------------------------------------------
# Check availability once at import time (fast HEAD-like check)
# ---------------------------------------------------------------------------
def _check_open_meteo() -> bool:
    try:
        url = (
            f"{OPEN_METEO_BASE}?latitude=28.6&longitude=77.2"
            f"&current=temperature_2m&forecast_days=1"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "SkyGuardAI/3.0"})
        with urllib.request.urlopen(req, timeout=6, context=_SSL_CTX) as r:
            return r.status == 200
    except Exception:
        return False


OPEN_METEO_AVAILABLE: bool = _check_open_meteo()


# ---------------------------------------------------------------------------
# Internal: fetch one batch of ≤ _BATCH_SIZE stations (current-weather)
# ---------------------------------------------------------------------------
def _fetch_current_batch(batch: list[dict]) -> list[dict]:
    """
    Fetch current-weather for one batch of ≤ _BATCH_SIZE valid stations.
    Returns a list of raw dicts (one per station); empty list on failure.
    """
    if not batch:
        return []

    lats = ",".join(str(s["lat"]) for s in batch)
    lons = ",".join(str(s["lon"]) for s in batch)
    url = (
        f"{OPEN_METEO_BASE}?"
        f"latitude={lats}&longitude={lons}"
        f"&current={OM_CURRENT_FIELDS}"
        f"&forecast_days=1&timezone=auto"
    )

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "SkyGuardAI/3.0"})
        with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT_S, context=_SSL_CTX) as r:
            raw = r.read()
        responses = json.loads(raw)
    except Exception as e:
        warnings.warn(f"Open-Meteo current-weather batch failed: {e}")
        return []

    # Open-Meteo: single station → dict, multiple stations → list
    if isinstance(responses, dict):
        responses = [responses]

    return responses if isinstance(responses, list) else []


# ---------------------------------------------------------------------------
# Internal: fetch one batch of ≤ _BATCH_SIZE stations (hourly history)
# ---------------------------------------------------------------------------
def _fetch_hourly_batch(batch: list[dict], start_str: str, end_str: str) -> list[dict]:
    """
    Fetch hourly data for one batch of ≤ _BATCH_SIZE valid stations.
    Returns a list of raw dicts (one per station); empty list on failure.
    """
    if not batch:
        return []

    lats = ",".join(str(s["lat"]) for s in batch)
    lons = ",".join(str(s["lon"]) for s in batch)
    url = (
        f"{OPEN_METEO_BASE}?"
        f"latitude={lats}&longitude={lons}"
        f"&hourly={OM_HOURLY_FIELDS}"
        f"&start_date={start_str}&end_date={end_str}"
        f"&timezone=auto"
    )

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "SkyGuardAI/3.0"})
        with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT_S * 3, context=_SSL_CTX) as r:
            raw = r.read()
        responses = json.loads(raw)
    except Exception as e:
        warnings.warn(f"Open-Meteo hourly batch failed: {e}")
        return []

    if isinstance(responses, dict):
        responses = [responses]

    return responses if isinstance(responses, list) else []


# ---------------------------------------------------------------------------
# Public: fetch CURRENT weather snapshot (one row per station)
# ---------------------------------------------------------------------------
def fetch_live_observations(stations: list[dict]) -> Optional[pd.DataFrame]:
    """
    Fetch current weather observations for a list of stations from Open-Meteo.

    Returns
    -------
    pd.DataFrame with columns:
        timestamp, station_id, name, lat, lon, temp, humidity, pressure
    or None on failure.

    Never returns simulated data. On any failure returns None.
    """
    if not stations:
        return None

    valid_stations, invalid_stations = _filter_valid_stations(stations)
    if not valid_stations:
        warnings.warn("No stations with valid coordinates — cannot fetch.")
        return None

    if invalid_stations:
        warnings.warn(
            f"Excluded {len(invalid_stations)} stations with invalid coordinates "
            f"from live fetch."
        )

    # Chunk into batches of _BATCH_SIZE
    records = []
    for i in range(0, len(valid_stations), _BATCH_SIZE):
        batch = valid_stations[i: i + _BATCH_SIZE]
        responses = _fetch_current_batch(batch)

        # Map responses back by position (Open-Meteo preserves order)
        for j, station in enumerate(batch):
            if j >= len(responses):
                warnings.warn(
                    f"No response for station {station['station_id']} in batch."
                )
                continue

            resp = responses[j]
            cur = resp.get("current", {})
            if not cur:
                warnings.warn(
                    f"No 'current' block in response for station {station['station_id']}"
                )
                continue

            ts_str = cur.get("time")
            try:
                ts = pd.Timestamp(ts_str) if ts_str else pd.Timestamp.now()
            except Exception:
                ts = pd.Timestamp.now()

            temp     = cur.get("temperature_2m")
            humidity = cur.get("relative_humidity_2m")
            pressure = cur.get("surface_pressure")

            if temp is None and humidity is None and pressure is None:
                warnings.warn(f"All fields missing for station {station['station_id']}")
                continue

            records.append({
                "timestamp":  ts,
                "station_id": station["station_id"],
                "name":       station["name"],
                "lat":        float(station["lat"]),
                "lon":        float(station["lon"]),
                "temp":       float(temp)     if temp     is not None else float("nan"),
                "humidity":   float(humidity) if humidity is not None else float("nan"),
                "pressure":   float(pressure) if pressure is not None else float("nan"),
            })

    if not records:
        return None

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Public: fetch HOURLY history (many rows per station)
# ---------------------------------------------------------------------------
def fetch_historical_observations(
    stations: list[dict], hours: int = 72
) -> Optional[pd.DataFrame]:
    """
    Fetch recent hourly weather data (past `hours` hours) from Open-Meteo.

    Uses the hourly endpoint → time-series with one row per station per hour.
    Batches requests in chunks of _BATCH_SIZE (≤50 per Open-Meteo limit).
    Filters invalid coordinates before request construction.

    Returns
    -------
    pd.DataFrame with columns:
        timestamp, station_id, name, lat, lon, temp, humidity, pressure
    or None on total failure.
    """
    if not stations:
        return None

    valid_stations, invalid_stations = _filter_valid_stations(stations)
    if not valid_stations:
        warnings.warn("No stations with valid coordinates — cannot fetch historical data.")
        return None

    if invalid_stations:
        warnings.warn(
            f"Excluded {len(invalid_stations)} stations with invalid coordinates "
            f"from historical fetch."
        )

    now       = datetime.now(timezone.utc)
    start     = now - timedelta(hours=hours)
    start_str = start.strftime("%Y-%m-%d")
    end_str   = now.strftime("%Y-%m-%d")
    cutoff    = pd.Timestamp(start)

    all_records = []

    # Chunk into batches of _BATCH_SIZE
    for i in range(0, len(valid_stations), _BATCH_SIZE):
        batch     = valid_stations[i: i + _BATCH_SIZE]
        responses = _fetch_hourly_batch(batch, start_str, end_str)

        for j, station in enumerate(batch):
            if j >= len(responses):
                warnings.warn(
                    f"No response for station {station['station_id']} in hourly batch."
                )
                continue

            resp   = responses[j]
            hourly = resp.get("hourly", {})
            times  = hourly.get("time", [])
            temps  = hourly.get("temperature_2m", [])
            hums   = hourly.get("relative_humidity_2m", [])
            press  = hourly.get("surface_pressure", [])

            n = len(times)
            if n == 0:
                warnings.warn(
                    f"Empty hourly block for station {station['station_id']}"
                )
                continue

            for k in range(n):
                t  = temps[k] if k < len(temps) else None
                rh = hums[k]  if k < len(hums)  else None
                p  = press[k] if k < len(press)  else None

                # Skip completely null rows
                if t is None and rh is None and p is None:
                    continue

                try:
                    ts = pd.Timestamp(times[k])
                except Exception:
                    continue

                all_records.append({
                    "timestamp":  ts,
                    "station_id": station["station_id"],
                    "name":       station["name"],
                    "lat":        float(station["lat"]),
                    "lon":        float(station["lon"]),
                    "temp":       float(t)  if t  is not None else float("nan"),
                    "humidity":   float(rh) if rh is not None else float("nan"),
                    "pressure":   float(p)  if p  is not None else float("nan"),
                })

    if not all_records:
        warnings.warn("No valid hourly records were returned across all batches.")
        return None

    df = pd.DataFrame(all_records)

    # Keep only the requested window (in case Open-Meteo returns extra days)
    if cutoff.tz is not None:
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
        cutoff = cutoff.replace(tzinfo=None)
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = df[df["timestamp"] >= cutoff].copy()
    df = df.sort_values(["station_id", "timestamp"]).reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# Validation helper (unchanged from previous version)
# ---------------------------------------------------------------------------
def validate_live_data(
    df: pd.DataFrame, min_hours: int = 12, min_stations: int = 3
) -> tuple[bool, str, pd.DataFrame]:
    """
    Perform lightweight validation on fetched live/historical data before ML processing.
    Ensures data provenance, deduplicates, and checks sufficient history/spatial coverage.

    Returns: (is_valid, status_message, cleaned_df)
    """
    if df is None or df.empty:
        return False, "Data fetch returned empty results.", df

    df_clean = df.copy()

    # 1. Drop complete duplicates
    df_clean = df_clean.drop_duplicates(subset=["station_id", "timestamp"], keep="last")

    # 2. Check for missing essential columns
    required_cols = ["timestamp", "station_id", "temp", "pressure", "humidity", "lat", "lon"]
    missing = [c for c in required_cols if c not in df_clean.columns]
    if missing:
        return False, f"Malformed records: missing required columns {missing}", df_clean

    # 3. Ensure numeric types for sensors
    for col in ["temp", "pressure", "humidity", "lat", "lon"]:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    # 4. Check spatial coverage
    n_stations = df_clean["station_id"].nunique()
    if n_stations < min_stations:
        return (
            False,
            f"Insufficient spatial coverage: only {n_stations} stations reported "
            f"(need {min_stations}).",
            df_clean,
        )

    # 5. Check temporal history
    ts_min = df_clean["timestamp"].min()
    ts_max = df_clean["timestamp"].max()
    history_hours = (ts_max - ts_min).total_seconds() / 3600.0
    if history_hours < min_hours:
        return (
            False,
            f"Warming Up / Insufficient History: got {history_hours:.1f}h of data, "
            f"need at least {min_hours}h for rolling features.",
            df_clean,
        )

    # 6. Check for excessively stale data (max timestamp older than 12 hours)
    now = pd.Timestamp.now(tz=ts_max.tz) if ts_max.tz else pd.Timestamp.now()
    stale_hours = (now - ts_max).total_seconds() / 3600.0
    if stale_hours > 12:
        return (
            False,
            f"Stale observations: latest data is {stale_hours:.1f} hours old.",
            df_clean,
        )

    return True, "Data validated successfully.", df_clean
