"""
data_source.py — SkyGuard AI Phase 3
======================================
DataSource abstraction: both SimulatedDataSource and ActualAWSDataSource
produce the same normalized observation schema that feeds into the shared
anomaly-detection pipeline (run_pipeline in anomaly_engine.py).

Normalized schema produced by both sources:
    timestamp   datetime64[ns]
    station_id  str
    name        str           (display name)
    lat         float         (WGS-84 decimal degrees)
    lon         float
    temp        float         (°C)
    humidity    float         (%)
    pressure    float         (hPa)

Ground truth (only SimulatedDataSource can provide it):
    timestamp       datetime64[ns]
    station_id      str
    is_anomaly_gt   bool
    true_root_cause str

Design constraints:
  - Ground truth MUST NEVER be passed to the Isolation Forest or any feature.
  - It is returned separately and used only in evaluate_pipeline().
  - ActualAWSDataSource always returns None for ground truth.
"""

from __future__ import annotations

import io
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Expected column ranges for validation
# ---------------------------------------------------------------------------
TEMP_RANGE     = (-50.0, 60.0)    # °C — plausible land-surface air temp
HUMIDITY_RANGE = (0.0, 100.0)     # %
PRESSURE_RANGE = (800.0, 1100.0)  # hPa

# Minimum stations required for spatial features to be meaningful
MIN_STATIONS = 2

# Station coordinates used when the CSV does not supply lat/lon.
# Users can override by including lat/lon columns in their CSV.
# Format: station_id → (lat, lon, display_name)
DEFAULT_STATION_META: dict[str, tuple[float, float, str]] = {
    "AWS-001": (28.6139, 77.2090, "Delhi"),
    "AWS-002": (19.0760, 72.8777, "Mumbai"),
    "AWS-003": (13.0827, 80.2707, "Chennai"),
    "AWS-004": (22.5726, 88.3639, "Kolkata"),
    "AWS-005": (23.2599, 77.4126, "Bhopal"),
    "AWS-006": (26.9124, 75.7873, "Jaipur"),
    "AWS-007": (17.3850, 78.4867, "Hyderabad"),
    "AWS-008": (18.5204, 73.8567, "Pune"),
    "AWS-009": (25.5941, 85.1376, "Patna"),
    "AWS-010": (23.0225, 72.5714, "Ahmedabad"),
    "AWS-011": (26.8467, 80.9462, "Lucknow"),
    "AWS-012": (20.2961, 85.8245, "Bhubaneswar"),
}


# ===========================================================================
# Abstract base
# ===========================================================================

class DataSource(ABC):
    """Base class for all SkyGuard data sources."""

    @abstractmethod
    def get_sensor_df(self) -> pd.DataFrame:
        """Return a normalized sensor DataFrame."""

    @abstractmethod
    def get_ground_truth(self) -> Optional[pd.DataFrame]:
        """Return ground-truth labels if available, else None."""

    @abstractmethod
    def get_source_name(self) -> str:
        """Short human-readable name for display."""

    @abstractmethod
    def get_source_type(self) -> str:
        """'simulated' or 'actual'"""

    def has_ground_truth(self) -> bool:
        return self.get_ground_truth() is not None


# ===========================================================================
# SimulatedDataSource
# ===========================================================================

class SimulatedDataSource(DataSource):
    """
    Wraps the existing data_engine simulator.
    Calls generate_dataset() exactly once on first access; result is cached
    for the lifetime of this object so repeated calls are cheap.
    """

    def __init__(self):
        self._sensor_df: Optional[pd.DataFrame] = None
        self._gt_df: Optional[pd.DataFrame] = None
        self._nbrs: Optional[dict] = None

    def _ensure_loaded(self):
        if self._sensor_df is None:
            from data_engine import generate_dataset, neighbor_map
            self._sensor_df, self._gt_df = generate_dataset()
            self._nbrs = neighbor_map()

    def get_sensor_df(self) -> pd.DataFrame:
        self._ensure_loaded()
        return self._sensor_df.copy()

    def get_ground_truth(self) -> pd.DataFrame:
        self._ensure_loaded()
        return self._gt_df.copy()

    def get_neighbor_map(self) -> dict:
        self._ensure_loaded()
        return self._nbrs

    def get_source_name(self) -> str:
        return "Simulated Data"

    def get_source_type(self) -> str:
        return "simulated"


# ===========================================================================
# ActualAWSDataSource
# ===========================================================================

class ActualAWSDataSource(DataSource):
    """
    Accepts a user-uploaded CSV and normalizes it to the shared schema.

    Required CSV columns: timestamp, station_id, temperature, pressure, humidity
    Optional CSV columns: lat, lon, name

    If lat/lon are missing, they are resolved from DEFAULT_STATION_META
    using station_id. Stations with unknown IDs and no coordinates are
    assigned NaN (those stations will have spatial features = 0).
    """

    # Required source column names (before normalization)
    REQUIRED_COLS = {"timestamp", "station_id", "temperature", "pressure", "humidity"}
    OPTIONAL_COLS = {"lat", "lon", "name"}

    def __init__(self, raw_df: pd.DataFrame):
        self._raw = raw_df.copy()
        self._validated: Optional[pd.DataFrame] = None
        self._errors: list[str] = []
        self._warnings: list[str] = []
        self._validate()

    # ---------------------------------------------------------------- public

    @property
    def validation_errors(self) -> list[str]:
        return list(self._errors)

    @property
    def validation_warnings(self) -> list[str]:
        return list(self._warnings)

    @property
    def is_valid(self) -> bool:
        return len(self._errors) == 0

    def get_sensor_df(self) -> pd.DataFrame:
        if not self.is_valid:
            raise ValueError(
                f"Data failed validation ({len(self._errors)} errors). "
                "Check validation_errors before calling get_sensor_df()."
            )
        return self._validated.copy()

    def get_ground_truth(self) -> None:
        return None

    def get_source_name(self) -> str:
        return "Actual AWS Data (CSV)"

    def get_source_type(self) -> str:
        return "actual"

    def get_neighbor_map(self) -> dict:
        """Build a spatial neighbour map from the validated data (top-4 nearest)."""
        if not self.is_valid:
            return {}
        from data_engine import build_neighbor_map_from_df
        return build_neighbor_map_from_df(self._validated, top_k=4)

    # ---------------------------------------------------------------- validation

    def _validate(self):
        df = self._raw.copy()
        errors, warnings = [], []

        # 1. Required columns
        missing_cols = self.REQUIRED_COLS - set(df.columns.str.lower())
        if missing_cols:
            errors.append(f"Missing required columns: {sorted(missing_cols)}")
            self._errors = errors
            return  # Can't continue without required columns

        # Normalize column names to lowercase
        df.columns = df.columns.str.lower().str.strip()

        # 2. Parse timestamps
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False)
            df["timestamp"] = df["timestamp"].dt.tz_localize(None)  # strip tz for consistency
        except Exception as e:
            errors.append(f"Cannot parse 'timestamp' column as datetime: {e}")

        # 3. Cast numeric columns
        for col, label, (lo, hi) in [
            ("temperature", "Temperature (°C)",    TEMP_RANGE),
            ("pressure",    "Pressure (hPa)",       PRESSURE_RANGE),
            ("humidity",    "Humidity (%)",          HUMIDITY_RANGE),
        ]:
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                n_nan = df[col].isna().sum()
                if n_nan > 0:
                    warnings.append(f"{n_nan} non-numeric values in '{col}' converted to NaN.")
                oob = (~df[col].isna()) & ((df[col] < lo) | (df[col] > hi))
                n_oob = oob.sum()
                if n_oob > 0:
                    warnings.append(
                        f"{n_oob} values in '{col}' are outside plausible range "
                        f"[{lo}, {hi}]. They will be treated as suspect readings."
                    )
            except Exception as e:
                errors.append(f"Cannot process column '{col}': {e}")

        if errors:
            self._errors = errors
            self._warnings = warnings
            return

        # 4. Duplicate (timestamp, station_id) pairs
        dupes = df.duplicated(subset=["timestamp", "station_id"], keep=False).sum()
        if dupes > 0:
            warnings.append(
                f"{dupes} rows have duplicate (timestamp, station_id) combinations. "
                "Duplicates will be deduplicated by keeping the first occurrence."
            )
            df = df.drop_duplicates(subset=["timestamp", "station_id"], keep="first")

        # 5. Station count
        n_stations = df["station_id"].nunique()
        if n_stations < MIN_STATIONS:
            errors.append(
                f"Only {n_stations} distinct station(s) found. "
                f"At least {MIN_STATIONS} are required for spatial anomaly detection."
            )

        # 6. Minimum rows per station (soft warning)
        rows_per_station = df.groupby("station_id").size()
        sparse = rows_per_station[rows_per_station < 24]
        if not sparse.empty:
            warnings.append(
                f"Stations with < 24 observations (rolling features may be less reliable): "
                f"{list(sparse.index)}"
            )

        if errors:
            self._errors = errors
            self._warnings = warnings
            return

        # 7. Normalize to internal schema
        df = df.rename(columns={
            "temperature": "temp",
        })

        # Resolve lat/lon/name
        if "lat" not in df.columns or "lon" not in df.columns:
            df["lat"] = np.nan
            df["lon"] = np.nan
            warnings.append(
                "Columns 'lat' and/or 'lon' not found. "
                "Attempting to resolve coordinates from built-in station metadata. "
                "Unknown station IDs will have coordinates set to NaN — "
                "spatial deviation features will be 0 for those stations."
            )

        if "name" not in df.columns:
            df["name"] = df["station_id"]

        # Fill missing coords from DEFAULT_STATION_META
        for idx, row in df.iterrows():
            sid = str(row["station_id"]).strip()
            if pd.isna(row.get("lat")) or pd.isna(row.get("lon")):
                if sid in DEFAULT_STATION_META:
                    lat, lon, name = DEFAULT_STATION_META[sid]
                    df.at[idx, "lat"] = lat
                    df.at[idx, "lon"] = lon
                    if row["name"] == sid:
                        df.at[idx, "name"] = name
                else:
                    warnings.append(
                        f"Station '{sid}' has no lat/lon in CSV and is not in built-in metadata. "
                        "Spatial features will be 0 for this station."
                    )

        # Ensure float dtypes for coordinates
        df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
        df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

        # Final column selection matching the normalized schema
        keep_cols = ["timestamp", "station_id", "name", "lat", "lon",
                     "temp", "humidity", "pressure"]
        df = df[[c for c in keep_cols if c in df.columns]].reset_index(drop=True)

        self._validated = df
        self._errors = errors
        self._warnings = warnings


# ===========================================================================
# Helpers
# ===========================================================================

def _fallback_neighbor_map(df: pd.DataFrame, top_k: int = 4) -> dict:
    """
    Build a neighbour map from station lat/lon using Euclidean distance.
    This is a fallback used when the actual data source doesn't have a richer
    geographic distance function available.
    """
    stations = df.groupby("station_id")[["lat", "lon"]].first().dropna()
    nbrs = {}
    ids = list(stations.index)
    for sid in ids:
        lat1, lon1 = stations.loc[sid, "lat"], stations.loc[sid, "lon"]
        dists = []
        for other in ids:
            if other == sid:
                continue
            lat2, lon2 = stations.loc[other, "lat"], stations.loc[other, "lon"]
            if pd.isna(lat2) or pd.isna(lon2):
                continue
            d = ((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2) ** 0.5
            dists.append((d, other))
        dists.sort()
        nbrs[sid] = [s for _, s in dists[:top_k]]
    return nbrs


def validate_csv_bytes(content: bytes) -> tuple[ActualAWSDataSource, pd.DataFrame | None]:
    """
    Parse and validate a CSV file from raw bytes (e.g., from st.file_uploader).

    Returns (source, raw_df).
    source.is_valid indicates whether the data can be used.
    """
    try:
        raw_df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        # Return an object that reports the parse error
        dummy = pd.DataFrame(columns=list(ActualAWSDataSource.REQUIRED_COLS))
        src = ActualAWSDataSource(dummy)
        src._errors = [f"Could not parse CSV file: {e}"]
        return src, None
    return ActualAWSDataSource(raw_df), raw_df
