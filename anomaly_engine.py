"""
anomaly_engine.py — SkyGuard AI Phase 2
=========================================
Full anomaly detection pipeline: feature engineering → Isolation Forest →
evidence-based root-cause reasoning → confidence/severity → SHAP → correction → health.

Pipeline:
  1. NaN tracking      — record missing readings BEFORE imputation
  2. Imputation        — forward-fill within station, spatial fallback
  3. Feature engineering
       Temporal:    rolling stats, z-scores, rate-of-change, acceleration,
                    rolling min/max/range, time-of-day encoding, drift slope
       Multivariate: combined deviation magnitude, simultaneous-change score
       Spatial:     deviation from neighbour mean (temp, humidity, pressure),
                    spatial agreement score (non-circular, uses raw value changes)
  4. Isolation Forest  — trained on full feature matrix; produces anomaly_score
  5. Score normalisation → anomaly_score_pct [0-100]
  6. Evidence computation — per-row evidence dict for each evidence type
  7. Root-cause classification — transparent priority-ordered rule engine
  8. Classification confidence — evidence-strength score [0-100]
  9. Severity          — LOW / MEDIUM / HIGH / CRITICAL from score + magnitude
 10. SHAP / explainability — surrogate Random Forest + SHAP TreeExplainer
 11. Corrected value estimate — spatial neighbours first, temporal fallback
 12. Sensor health      — multi-signal health score with Degraded tier

Public API:
  run_pipeline(sensor_df, nbrs)          → (processed_df, iso_model)
  explain_with_shap(processed_df)        → dict[int, list[tuple[str, float]]]
  sensor_health(processed_df)            → pd.DataFrame
  evaluate_pipeline(processed_df, gt_df) → dict  (metrics vs ground truth)
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration — all tunable parameters in one place
# ---------------------------------------------------------------------------
ROLLING_WINDOW     = 6    # hours for rolling statistics
FROZEN_WINDOW      = 4    # hours for frozen-sensor detection (consecutive identical readings)
DRIFT_WINDOW       = 8    # hours for calibration-drift slope estimation
CONTAMINATION = 0.01
                          # Rationale: 9 fault windows / ~864 rows ≈ 5.7%; rounded to 6%
N_ESTIMATORS       = 300  # Isolation Forest trees (more = more stable scores)

# Thresholds used in root-cause rules
SPIKE_DIFF_THRESH   = 3.5  # °C/h or equivalent — rate-of-change for spike detection
SPATIAL_DEV_THRESH  = 3.0  # °C — deviation from neighbour mean to flag as spatially isolated
SPATIAL_AGREE_THRESH = 0.4 # fraction of neighbours showing same-direction change → genuine event
DRIFT_SLOPE_THRESH  = 0.12 # z-score units/h — threshold for drift detection
MULTIVAR_MAG_THRESH = 3.5  # combined z-score magnitude for multivariate inconsistency

# Root-cause string constants — must match ROOT_CAUSE_COLOR in app.py exactly
RC_NORMAL    = "Normal"
RC_SPIKE     = "Sensor Spike / Fault"
RC_FROZEN    = "Frozen / Stuck Sensor"
RC_DRIFT     = "Calibration Drift"
RC_COMM      = "Communication Failure"
RC_MULTIVAR  = "Multivariate Inconsistency"
RC_GENUINE   = "Genuine Weather Event (not a fault)"
RC_UNKNOWN   = "Unclassified Anomaly"

# Feature columns fed into Isolation Forest (22 features)
# Raw sensor values are included because their COMBINED behaviour matters.
# They are NOT the only signal — the engineered features capture temporal
# and spatial context which raw values alone cannot.
FEATURE_COLS = [
    # Raw sensor values (normalised by StandardScaler before IF)
    "temp", "humidity", "pressure",
    # Temporal rate-of-change (1-step differences)
    "feat_temp_diff", "feat_hum_diff", "feat_pres_diff",
    # Second derivative (acceleration) — catches spikes that diff alone may miss
    "feat_temp_accel",
    # Z-scores: deviation from station's own recent rolling mean
    "feat_temp_zscore", "feat_hum_zscore", "feat_pres_zscore",
    # Rolling range (max-min over ROLLING_WINDOW) — captures instability
    "feat_temp_range",
    # Time-of-day encoding (cyclical sine/cosine)
    "feat_hour_sin", "feat_hour_cos",
    # Multivariate composite features
    "feat_multivar_magnitude",      # sqrt(zscore_T² + zscore_H² + zscore_P²)
    "feat_multivar_simultaneous",   # T_diff² + H_diff² + P_diff² — multi-var change
    # Spatial features (deviation from neighbour mean — non-anomaly-based)
    "feat_spatial_temp_dev",
    "feat_spatial_hum_dev",
    "feat_spatial_pres_dev",
    # Frozen-sensor indicator
    "feat_temp_frozen",
    # Calibration drift slope
    "feat_temp_drift_slope",
    # Spatial agreement score (fraction of neighbours changing in same direction)
    "feat_spatial_agreement",
    "temp_residual",
]

assert len(FEATURE_COLS) == 22, f"Expected 22 features, got {len(FEATURE_COLS)}"


# ===========================================================================
# Public API
# ===========================================================================

def run_pipeline(sensor_df: pd.DataFrame, nbrs: dict) -> tuple[pd.DataFrame, IsolationForest]:
    """
    Run the full Phase 2 anomaly detection pipeline.

    Parameters
    ----------
    sensor_df : pd.DataFrame
        Raw sensor observations from generate_dataset() — sensor_df only,
        NOT ground_truth_df. Must have: timestamp, station_id, name, lat, lon,
        temp, humidity, pressure.
    nbrs : dict
        Spatial neighbour map from neighbor_map().

    Returns
    -------
    (processed_df, fitted_iso_model)

    processed_df adds these columns to sensor_df:
        was_nan              — bool: reading was missing before imputation
        feat_*               — 21 engineered features (see FEATURE_COLS)
        anomaly_score_raw    — raw Isolation Forest decision_function output
        anomaly_score_pct    — normalised [0, 100]; higher = more anomalous
        is_anomaly           — bool: IF prediction
        evidence             — dict of per-type evidence signals
        root_cause           — one of the RC_* string constants
        classification_confidence — [0, 100] evidence strength for root-cause label
        severity             — LOW / MEDIUM / HIGH / CRITICAL
        temp_corrected       — estimated true value (raw if normal, imputed if anomalous)
        humidity_corrected   — same
        pressure_corrected   — same
    """
    df = sensor_df.copy().sort_values(["station_id", "timestamp"]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Stage 1: Track NaN positions BEFORE any imputation
    # ------------------------------------------------------------------
    df["was_nan"] = df["temp"].isna() | df["humidity"].isna() | df["pressure"].isna()

    # ------------------------------------------------------------------
    # Stage 2: Impute missing values
    #   Priority 1: forward-fill then back-fill within the same station
    #   Priority 2: spatial mean of all stations at the same timestamp
    # ------------------------------------------------------------------
    for col in ["temp", "humidity", "pressure"]:
        df[col] = df.groupby("station_id")[col].transform(
            lambda x: x.ffill().bfill()
        )
    ts_means = df.groupby("timestamp")[["temp", "humidity", "pressure"]].transform("mean")
    for col in ["temp", "humidity", "pressure"]:
        still_nan = df[col].isna()
        df.loc[still_nan, col] = ts_means.loc[still_nan, col]

    # ------------------------------------------------------------------
    # Stage 3: Feature engineering
    # ------------------------------------------------------------------
    df = _engineer_features(df, nbrs)

    # ------------------------------------------------------------------
    # Stage 4: Isolation Forest
    # Scale features before training — IF is not distance-based but
    # StandardScaler ensures no single feature dominates due to unit differences.
    # ------------------------------------------------------------------
    X_raw = df[FEATURE_COLS].fillna(0).values
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    iso = IsolationForest(
        contamination=CONTAMINATION,
        n_estimators=N_ESTIMATORS,
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(X)

    preds  = iso.predict(X)           # +1 = normal, -1 = anomaly
    scores = iso.decision_function(X) # higher = more normal

    df["anomaly_score_raw"] = scores
    df["is_anomaly"] = (preds == -1) | df["was_nan"]

    # Normalise: anomaly_score_pct higher → more anomalous
    s_min, s_max = scores.min(), scores.max()
    if s_max > s_min:
        df["anomaly_score_pct"] = np.clip(
            (s_max - scores) / (s_max - s_min) * 100, 0, 100
        ).round(1)
    else:
        df["anomaly_score_pct"] = 0.0

    df["confidence"] = df["anomaly_score_pct"]

    # ------------------------------------------------------------------
    # Stage 5: Compute per-row evidence (uses raw features, not IF output)
    # ------------------------------------------------------------------
    df = _compute_evidence(df, nbrs)

    # ------------------------------------------------------------------
    # Stage 6: Root-cause classification
    # ------------------------------------------------------------------
    df["root_cause"] = df.apply(_classify_root_cause, axis=1)

    # Genuine Weather is NOT an anomaly
    df.loc[df["root_cause"] == RC_GENUINE, "is_anomaly"] = False

    # ------------------------------------------------------------------
    # Stage 7: Classification confidence and severity
    # ------------------------------------------------------------------
    df["classification_confidence"] = df.apply(_classification_confidence, axis=1)
    df["severity"]                  = df.apply(_compute_severity, axis=1)

    # ------------------------------------------------------------------
    # Stage 8: Corrected / imputed value estimates
    # ------------------------------------------------------------------
    df = _estimate_corrected_values(df, nbrs)

    # Re-sort chronologically for the UI
    df = df.sort_values(["timestamp", "station_id"]).reset_index(drop=True)

    return df, iso


# ---------------------------------------------------------------------------
def explain_with_shap(processed: pd.DataFrame) -> dict:
    """
    Compute SHAP-based feature-level explanations for anomalous observations.

    Method:
      A surrogate Random Forest classifier is trained to reproduce the Isolation
      Forest's is_anomaly labels. SHAP TreeExplainer is then applied to the
      surrogate RF.

      Note: SHAP is explaining the SURROGATE RF's decision, not Isolation Forest
      directly. This is the standard approach because IF's ensemble structure is
      not compatible with TreeExplainer. The surrogate RF is trained on the same
      features and achieves high agreement with IF labels.

    Returns
    -------
    dict[int, list[tuple[str, float]]]
        Keys = DataFrame index values of anomalous rows with anomaly_score_pct > 50.
        Values = top-5 (feature_name, shap_contribution) sorted by |contribution|.
    """
    # Guard: empty df or ML pipeline columns missing → nothing to explain
    required = {"is_anomaly", "anomaly_score_pct"} | set(FEATURE_COLS)
    if processed is None or processed.empty or not required.issubset(processed.columns):
        return {}

    flagged = processed[processed["is_anomaly"] & (processed["anomaly_score_pct"] > 50)]
    if flagged.empty:
        flagged = processed[processed["is_anomaly"]]
    if flagged.empty:
        return {}

    X_all  = processed[FEATURE_COLS].fillna(0).values
    y_all  = processed["is_anomaly"].astype(int).values
    X_flag = flagged[FEATURE_COLS].fillna(0).values

    # Scale consistently
    scaler = StandardScaler()
    X_all_sc  = scaler.fit_transform(X_all)
    X_flag_sc = scaler.transform(X_flag)

    # Surrogate RF
    rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X_all_sc, y_all)

    result = {}

    try:
        import shap
        explainer = shap.TreeExplainer(rf)
        shap_vals = explainer.shap_values(X_flag_sc)

        # Handle both old (list) and new (ndarray with ndim=3) SHAP API
        if isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
            sv = shap_vals[:, :, 1]        # class-1 (anomaly) slice
        elif isinstance(shap_vals, list):
            sv = np.array(shap_vals[1])
        else:
            sv = np.array(shap_vals)

        for local_i, global_idx in enumerate(flagged.index):
            row_sv = sv[local_i]
            pairs  = sorted(zip(FEATURE_COLS, row_sv.tolist()),
                            key=lambda x: abs(x[1]), reverse=True)[:5]
            result[global_idx] = pairs

    except ImportError:
        # Fallback: RF feature importances × standardised feature value
        fi     = rf.feature_importances_
        X_mean = X_all_sc.mean(axis=0)
        X_std  = X_all_sc.std(axis=0) + 1e-6
        for local_i, global_idx in enumerate(flagged.index):
            row_vals      = X_flag_sc[local_i]
            contributions = fi * (row_vals - X_mean) / X_std
            pairs = sorted(zip(FEATURE_COLS, contributions.tolist()),
                           key=lambda x: abs(x[1]), reverse=True)[:5]
            result[global_idx] = pairs

    return result


# ---------------------------------------------------------------------------
def sensor_health(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "is_anomaly" not in df.columns:
        return pd.DataFrame(columns=[
            "name", "readings", "anomalies", "anomaly_rate", "missing_readings",
            "health_score", "status", "dominant_fault", "trend", "maintenance_risk", "recommendation"
        ])

    def calc_health_for_window(grp, is_recent=True):
        n_readings   = len(grp)
        n_anomalies  = int(grp["is_anomaly"].sum())
        n_genuine    = int((grp["root_cause"] == "Genuine Weather Event (not a fault)").sum()) if "root_cause" in grp.columns else 0
        # is_anomaly already excludes genuine weather events, so n_anomalies equals n_faults
        n_faults     = n_anomalies
        n_missing    = int(grp["was_nan"].sum()) if "was_nan" in grp.columns else 0
        
        rate = round(n_faults / n_readings * 100, 1) if n_readings > 0 else 0.0
        missing_frac = n_missing / n_readings if n_readings > 0 else 0.0

        base_score = 100.0 - (rate * 4.0)
        has_drift = "Calibration Drift" in grp.get("root_cause", pd.Series(dtype=str)).values
        drift_penalty = 15.0 if has_drift else 0.0
        has_freeze = "Frozen / Stuck Sensor" in grp.get("root_cause", pd.Series(dtype=str)).values
        freeze_penalty = 20.0 if has_freeze else 0.0
        missing_penalty = 10.0 * missing_frac

        sev_penalty = 0.0
        if "severity" in grp.columns:
            has_high = grp[grp["is_anomaly"] & (grp["root_cause"] != "Genuine Weather Event (not a fault)")]["severity"].isin(["HIGH", "CRITICAL"]).any()
            sev_penalty = 10.0 if has_high else 0.0

        calculated_health = base_score - drift_penalty - freeze_penalty - missing_penalty - sev_penalty
        health_score = min(100.0, max(0.0, calculated_health))
        return round(health_score, 1), n_faults, n_missing, n_genuine, has_drift, has_freeze

    max_ts = df["timestamp"].max()
    cutoff_recent = max_ts - pd.Timedelta(hours=48)
    cutoff_prev = max_ts - pd.Timedelta(hours=96)
    
    recent = df[df["timestamp"] >= cutoff_recent].copy()
    prev = df[(df["timestamp"] >= cutoff_prev) & (df["timestamp"] < cutoff_recent)].copy()

    rows = []
    for station_id, grp_recent in recent.groupby("station_id"):
        grp_prev = prev[prev["station_id"] == station_id]
        
        cur_health, cur_faults, cur_missing, cur_genuine, cur_drift, cur_freeze = calc_health_for_window(grp_recent)
        prev_health, prev_faults, _, _, _, _ = calc_health_for_window(grp_prev, is_recent=False) if not grp_prev.empty else (cur_health, 0, 0, 0, False, False)
        
        if cur_health >= 80:
            status = "Healthy"
        elif cur_health >= 60:
            status = "Degraded"
        elif cur_health >= 40:
            status = "Warning"
        else:
            status = "Critical"
            
        trend_diff = cur_health - prev_health
        if trend_diff <= -10:
            trend = "DETERIORATING"
        elif trend_diff >= 10:
            trend = "IMPROVING"
        else:
            trend = "UNCHANGED"
            
        # Maintenance Risk
        risk_score = 0
        if trend == "DETERIORATING": risk_score += 2
        if cur_drift: risk_score += 2
        if cur_freeze: risk_score += 3
        if cur_faults > 3: risk_score += 2
        if cur_missing > 5: risk_score += 1
        
        if risk_score >= 5 or cur_health < 40:
            risk = "CRITICAL"
            rec = "Urgent inspection"
        elif risk_score >= 3 or cur_health < 60:
            risk = "HIGH"
            rec = "Schedule preventive maintenance"
        elif risk_score >= 1 or cur_health < 80:
            risk = "MEDIUM"
            rec = "Inspect sensor"
        else:
            risk = "LOW"
            rec = "No action / Monitor"
            
        fault_counts = grp_recent[grp_recent["is_anomaly"] & (grp_recent["root_cause"] != "Genuine Weather Event (not a fault)")]["root_cause"].value_counts()
        dominant = fault_counts.index[0] if not fault_counts.empty else "—"

        n_readings = len(grp_recent)
        rate = round(cur_faults / n_readings * 100, 1) if n_readings > 0 else 0.0

        rows.append({
            "station_id": station_id,
            "name": grp_recent["name"].iloc[0],
            "readings": n_readings,
            "anomalies": cur_faults + cur_genuine,
            "faults": cur_faults,
            "genuine_events": cur_genuine,
            "anomaly_rate": rate,
            "missing_readings": cur_missing,
            "health_score": cur_health,
            "previous_health": prev_health,
            "trend": trend,
            "status": status,
            "dominant_fault": dominant,
            "maintenance_risk": risk,
            "recommendation": rec
        })

    return pd.DataFrame(rows).sort_values("health_score").reset_index(drop=True)

# ---------------------------------------------------------------------------
def evaluate_pipeline(processed_df: pd.DataFrame, ground_truth_df: pd.DataFrame) -> dict:
    """
    Compare pipeline predictions against simulator ground truth.

    IMPORTANT: ground_truth_df must be the one returned by generate_dataset().
    It must NEVER have been used as a pipeline input — it is evaluation-only.

    Returns
    -------
    dict with keys:
        anomaly_detection : dict  (precision, recall, F1, FPR, FNR, support)
        root_cause        : dict  (per-category precision/recall/F1)
        confusion_matrix  : pd.DataFrame (predicted vs true)
        summary           : dict  (total_obs, n_anomalies_gt, n_detected, n_false_alarms)
    """
    # Merge on (timestamp, station_id)
    merged = processed_df.merge(
        ground_truth_df[["timestamp", "station_id", "is_anomaly_gt", "true_root_cause"]],
        on=["timestamp", "station_id"],
        how="inner",
    )
    if merged.empty:
        return {"error": "No matching rows between processed and ground truth."}

    y_true = merged["is_anomaly_gt"].astype(int)
    y_pred = merged["is_anomaly"].astype(int)

    TP = int(((y_true == 1) & (y_pred == 1)).sum())
    FP = int(((y_true == 0) & (y_pred == 1)).sum())
    TN = int(((y_true == 0) & (y_pred == 0)).sum())
    FN = int(((y_true == 1) & (y_pred == 0)).sum())

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    fpr       = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    fnr       = FN / (FN + TP) if (FN + TP) > 0 else 0.0

    # Per-root-cause evaluation (across ALL rows to calculate True FPR/FNR)
    rc_metrics = {}
    for rc_cat in merged["true_root_cause"].dropna().unique():
        if rc_cat == "Normal":
            continue
        mask_gt   = merged["true_root_cause"] == rc_cat
        mask_pred = merged["root_cause"] == rc_cat
        
        tp_rc = int((mask_gt & mask_pred).sum())
        fp_rc = int((~mask_gt & mask_pred).sum())
        fn_rc = int((mask_gt & ~mask_pred).sum())
        tn_rc = int((~mask_gt & ~mask_pred).sum())
        
        prec_rc = tp_rc / (tp_rc + fp_rc) if (tp_rc + fp_rc) > 0 else 0.0
        rec_rc  = tp_rc / (tp_rc + fn_rc) if (tp_rc + fn_rc) > 0 else 0.0
        f1_rc   = (2 * prec_rc * rec_rc / (prec_rc + rec_rc)) if (prec_rc + rec_rc) > 0 else 0.0
        fpr_rc  = fp_rc / (fp_rc + tn_rc) if (fp_rc + tn_rc) > 0 else 0.0
        fnr_rc  = fn_rc / (fn_rc + tp_rc) if (fn_rc + tp_rc) > 0 else 0.0
        
        rc_metrics[rc_cat] = {
            "TP": tp_rc, "FP": fp_rc, "FN": fn_rc, "TN": tn_rc,
            "precision": round(prec_rc, 3),
            "recall":    round(rec_rc,  3),
            "F1":        round(f1_rc,   3),
            "FPR":       round(fpr_rc,  3),
            "FNR":       round(fnr_rc,  3),
            "support":   int(mask_gt.sum()),
        }
        
    anom_rows = merged[merged["is_anomaly_gt"]]

    # Confusion matrix (predicted root_cause vs true_root_cause, anomaly rows only)
    if not anom_rows.empty:
        cats = sorted(set(anom_rows["true_root_cause"].unique()) |
                      set(anom_rows["root_cause"].unique()))
        cm = pd.crosstab(
            anom_rows["root_cause"],
            anom_rows["true_root_cause"],
            rownames=["Predicted"], colnames=["True"],
        ).reindex(index=cats, columns=cats, fill_value=0)
    else:
        cm = pd.DataFrame()

    return {
        "anomaly_detection": {
            "TP": TP, "FP": FP, "TN": TN, "FN": FN,
            "precision": round(precision, 3),
            "recall":    round(recall,    3),
            "F1":        round(f1,        3),
            "FPR":       round(fpr,       3),
            "FNR":       round(fnr,       3),
            "support":   int(y_true.sum()),
        },
        "root_cause": rc_metrics,
        "confusion_matrix": cm,
        "summary": {
            "total_observations":   len(merged),
            "n_anomalies_gt":       int(y_true.sum()),
            "n_detected":           int(y_pred.sum()),
            "n_true_positives":     TP,
            "n_false_alarms":       FP,
            "n_missed":             FN,
        },
    }


# ===========================================================================
# Internal helpers
# ===========================================================================

def _engineer_features(df: pd.DataFrame, nbrs: dict) -> pd.DataFrame:
    """Compute all 21 engineered features. No look-ahead: only past data used."""

    # ---- Temporal features (per station, in chronological order) ----
    def _temporal(grp: pd.DataFrame) -> pd.DataFrame:
        g = grp.sort_values("timestamp").copy()
        n = len(g)

        for col, short in [("temp", "temp"), ("humidity", "hum"), ("pressure", "pres")]:
            # 1-step rate of change
            g[f"feat_{short}_diff"] = g[col].diff().fillna(0)

            # Rolling stats (min_periods=2 to handle short histories)
            rmean = g[col].rolling(ROLLING_WINDOW, min_periods=2).mean().bfill()
            rstd  = g[col].rolling(ROLLING_WINDOW, min_periods=2).std().bfill().fillna(1e-6)

            # Z-score: station-relative deviation (captures contextual anomalies)
            g[f"feat_{short}_zscore"] = ((g[col] - rmean) / (rstd + 1e-6)).fillna(0)

        # Temperature acceleration (second derivative) — spikes show large accel
        g["feat_temp_accel"] = g["feat_temp_diff"].diff().fillna(0)

        # Rolling range of temperature (max-min over ROLLING_WINDOW)
        rmax = g["temp"].rolling(ROLLING_WINDOW, min_periods=2).max().bfill()
        rmin = g["temp"].rolling(ROLLING_WINDOW, min_periods=2).min().bfill()
        g["feat_temp_range"] = (rmax - rmin).fillna(0)

        # Time-of-day cyclical encoding (hour within the 72-h simulation)
        hour_of_day = g["timestamp"].dt.hour.astype(float)
        g["feat_hour_sin"] = np.sin(2 * np.pi * hour_of_day / 24)
        g["feat_hour_cos"] = np.cos(2 * np.pi * hour_of_day / 24)

        # Multivariate composite: magnitude of combined deviation
        # sqrt(z_T² + z_H² + z_P²) — if all three deviate together, this is high
        g["feat_multivar_magnitude"] = np.sqrt(
            g["feat_temp_zscore"] ** 2 +
            g["feat_hum_zscore"]  ** 2 +
            g["feat_pres_zscore"] ** 2
        ).fillna(0)

        # Simultaneous change score: T_diff² + H_diff² + P_diff²
        # Captures moments when multiple variables change at the same time
        g["feat_multivar_simultaneous"] = (
            g["feat_temp_diff"] ** 2 +
            g["feat_hum_diff"]  ** 2 +
            g["feat_pres_diff"] ** 2
        ).fillna(0)

        # Frozen sensor: rolling std of temperature ≈ 0 for FROZEN_WINDOW hours
        frozen_std = g["temp"].rolling(FROZEN_WINDOW, min_periods=FROZEN_WINDOW).std()
        g["feat_temp_frozen"] = (frozen_std < 0.02).fillna(False).astype(float)

        # Calibration drift slope: linear trend of temp z-score over DRIFT_WINDOW
        # Positive slope = temperature slowly reading higher → upward drift
        g["feat_temp_drift_slope"] = (
            g["feat_temp_zscore"]
            .rolling(DRIFT_WINDOW, min_periods=3)
            .apply(_rolling_slope, raw=True)
            .fillna(0)
        )

        return g

    parts = [_temporal(grp) for _, grp in df.groupby("station_id")]
    df = pd.concat(parts).sort_values(["station_id", "timestamp"]).reset_index(drop=True)

    # ---- Spatial features (cross-station, same timestamp) ----
    # Build fast lookup tables indexed by (timestamp, station_id)
    df_indexed = df.set_index(["timestamp", "station_id"])
    temp_lut  = df_indexed["temp"]
    hum_lut   = df_indexed["humidity"]
    pres_lut  = df_indexed["pressure"]
    tdiff_lut = df_indexed["feat_temp_diff"]

    spatial_temp_dev  = np.zeros(len(df))
    spatial_hum_dev   = np.zeros(len(df))
    spatial_pres_dev  = np.zeros(len(df))
    spatial_agreement = np.zeros(len(df))

    for i, row in df.iterrows():
        ts  = row["timestamp"]
        sid = row["station_id"]
        peers = nbrs.get(sid, [])
        if not peers:
            continue

        nbr_temps, nbr_hums, nbr_press = [], [], []
        nbr_diffs = []
        for nbr in peers:
            key = (ts, nbr)
            if key in temp_lut.index:
                t = temp_lut[key]
                h = hum_lut[key]
                p = pres_lut[key]
                d = tdiff_lut[key]
                if not np.isnan(t):
                    nbr_temps.append(t)
                    nbr_hums.append(h)
                    nbr_press.append(p)
                    nbr_diffs.append(d)

        if nbr_temps:
            spatial_temp_dev[i]  = float(row["temp"])     - np.mean(nbr_temps)
            spatial_hum_dev[i]   = float(row["humidity"]) - np.mean(nbr_hums)
            spatial_pres_dev[i]  = float(row["pressure"]) - np.mean(nbr_press)

            # Spatial agreement: what fraction of neighbours changed in same direction?
            # Uses raw diff values — NOT based on IF anomaly flags (avoids circularity)
            own_diff = float(row["feat_temp_diff"])
            if abs(own_diff) > 0.1 and nbr_diffs:
                same_dir = sum(1 for d in nbr_diffs
                               if np.sign(d) == np.sign(own_diff) and abs(d) > 0.1)
                spatial_agreement[i] = same_dir / len(nbr_diffs)
            else:
                spatial_agreement[i] = 0.0

    df["feat_spatial_temp_dev"]  = spatial_temp_dev
    df["feat_spatial_hum_dev"]   = spatial_hum_dev
    df["feat_spatial_pres_dev"]  = spatial_pres_dev
    df["feat_spatial_agreement"] = spatial_agreement

    
    # Diurnal residual
    # Causal Diurnal Residual
    df["hour"] = df["timestamp"].dt.hour
    
    # 1. Historical same-hour mean (strictly prior days via shift(1))
    df["expected_temp"] = df.groupby(["station_id", "hour"])["temp"].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    
    # 2. Causal fallback for Day 1: if no prior days exist, use the current temperature
    # (assuming residual = 0 during the warm-up period)
    df["expected_temp"] = df["expected_temp"].fillna(df["temp"])
    
    df["temp_residual"] = df["temp"] - df["expected_temp"]
    df.drop(columns=["hour"], inplace=True)
    return df


def _rolling_slope(arr: np.ndarray) -> float:
    """Least-squares slope of arr (for rolling.apply)."""
    n = len(arr)
    if n < 3:
        return 0.0
    x  = np.arange(n, dtype=float)
    xm = x.mean()
    ym = arr.mean()
    denom = ((x - xm) ** 2).sum()
    return float(((x - xm) * (arr - ym)).sum() / denom) if denom > 1e-9 else 0.0


def _compute_evidence(df: pd.DataFrame, nbrs: dict) -> pd.DataFrame:
    """
    Compute an evidence dictionary for each row.
    Evidence is computed from engineered features and raw values — NOT from IF output.
    This keeps root-cause reasoning independent from the ML model.
    """

    def _ev(row) -> dict:
        return {
            # Was the original reading missing?
            "missing":    bool(row.get("was_nan", False)),
            # Frozen sensor
            "frozen":     float(row.get("feat_temp_frozen", 0)),
            # Temperature rate-of-change (°C/h)
            "temp_roc":   float(row.get("feat_temp_diff", 0)),
            # Temperature z-score (how many σ from recent behaviour)
            "temp_zscore": float(row.get("feat_temp_zscore", 0)),
            # Spatial deviation from neighbours (°C)
            "spatial_dev": float(row.get("feat_spatial_temp_dev", 0)),
            "spatial_hum_dev": float(row.get("feat_spatial_hum_dev", 0)),
            "spatial_pres_dev": float(row.get("feat_spatial_pres_dev", 0)),
            # Calibration drift slope
            "drift_slope": float(row.get("feat_temp_drift_slope", 0)),
            # Combined deviation magnitude
            "multivar_mag": float(row.get("feat_multivar_magnitude", 0)),
            # Fraction of neighbours changing in same direction (spatial agreement)
            "spatial_agree": float(row.get("feat_spatial_agreement", 0)),
            # Anomaly score from IF
            "anomaly_score_pct": float(row.get("anomaly_score_pct", 0)),
        }

    df["evidence"] = df.apply(_ev, axis=1)
    return df


def _classify_root_cause(row) -> str:
    if not row.get("is_anomaly", False):
        return RC_NORMAL
        
    ev = row.get("evidence", {})

    if ev.get("missing", False):
        return RC_COMM

    if ev.get("frozen", 0) > 0.5:
        return RC_FROZEN

    # Rule 3 — Multivariate Inconsistency
    if (ev.get("multivar_mag", 0) >= MULTIVAR_MAG_THRESH
            and ev.get("spatial_agree", 0) < SPATIAL_AGREE_THRESH):
        return RC_MULTIVAR

    # Rule 4 — Sensor Spike / Fault (Catch extreme ROC early)
    if (abs(ev.get("temp_roc", 0)) >= SPIKE_DIFF_THRESH
            and abs(ev.get("spatial_dev", 0)) >= SPATIAL_DEV_THRESH):
        return RC_SPIKE

    # Rule 5 — Genuine Weather Event:
    # A true regional event affects humidity/pressure significantly.
    if (abs(ev.get("spatial_dev", 0)) >= 2.0 and abs(ev.get("spatial_hum_dev", 0)) >= 5.0):
        return RC_GENUINE
    if (ev.get("spatial_agree", 0) >= 0.6 and abs(ev.get("temp_roc", 0)) >= 1.5 and abs(ev.get("spatial_dev", 0)) < SPATIAL_DEV_THRESH):
        return RC_GENUINE

    # Rule 6 — Calibration Drift
    # If spatial_dev is large but humidity and pressure are NOT deviating widely, it's an isolated temp drift.
    if (abs(ev.get("drift_slope", 0)) >= DRIFT_SLOPE_THRESH and abs(ev.get("temp_zscore", 0)) > 1.2):
        return RC_DRIFT
    if (abs(ev.get("spatial_dev", 0)) >= SPATIAL_DEV_THRESH and abs(ev.get("temp_roc", 0)) < SPIKE_DIFF_THRESH):
        return RC_DRIFT

    # Catch remaining extreme spatial deviations (Spikes)
    if abs(ev.get("spatial_dev", 0)) >= SPATIAL_DEV_THRESH * 1.5:
        return RC_SPIKE

    if row.get("is_anomaly", False):
        return RC_UNKNOWN
        
    return RC_NORMAL


def _classification_confidence(row) -> float:
    """
    Evidence-strength score [0, 100] for the assigned root cause.

    This is NOT a statistical probability. It reflects how strongly the
    observable evidence supports the root-cause classification.
    Methodology: weighted sum of rule-specific evidence signals, normalised to [0, 100].

    Labelled as 'classification_confidence' to distinguish it from anomaly_score_pct.
    """
    if not row["is_anomaly"]:
        return 0.0

    rc = row.get("root_cause", RC_UNKNOWN)
    ev = row.get("evidence", {})
    score = 0.0

    if rc == RC_COMM:
        # Strong: reading was definitively missing
        score = 95.0

    elif rc == RC_FROZEN:
        # Strong: frozen flag is binary; scale by z-score to account for duration
        frozen_strength = min(ev.get("frozen", 0) * 100, 60)
        score = 50.0 + frozen_strength * 0.4

    elif rc == RC_GENUINE:
        # Proportional to spatial agreement fraction and anomaly score
        agree_contrib  = ev.get("spatial_agree", 0) * 50
        score_contrib  = ev.get("anomaly_score_pct", 0) * 0.3
        score = min(agree_contrib + score_contrib, 90.0)

    elif rc == RC_SPIKE:
        # Proportional to |rate-of-change| and |spatial_dev|
        roc_contrib   = min(abs(ev.get("temp_roc", 0)) / SPIKE_DIFF_THRESH * 35, 40)
        spat_contrib  = min(abs(ev.get("spatial_dev", 0)) / SPATIAL_DEV_THRESH * 35, 40)
        score_contrib = ev.get("anomaly_score_pct", 0) * 0.15
        score = min(roc_contrib + spat_contrib + score_contrib, 90.0)

    elif rc == RC_MULTIVAR:
        # Proportional to combined deviation magnitude
        mag_contrib  = min(ev.get("multivar_mag", 0) / MULTIVAR_MAG_THRESH * 60, 70)
        score_contrib = ev.get("anomaly_score_pct", 0) * 0.2
        score = min(mag_contrib + score_contrib, 88.0)

    elif rc == RC_DRIFT:
        # Proportional to drift slope and z-score magnitude
        slope_contrib = min(abs(ev.get("drift_slope", 0)) / DRIFT_SLOPE_THRESH * 40, 50)
        zscore_contrib = min(abs(ev.get("temp_zscore", 0)) / 3.0 * 30, 35)
        score = min(slope_contrib + zscore_contrib, 85.0)

    else:  # RC_UNKNOWN
        # Low confidence: IF flagged it but no pattern matched clearly
        score = ev.get("anomaly_score_pct", 0) * 0.4

    return round(min(max(score, 0.0), 100.0), 1)


def _compute_severity(row) -> str:
    """
    Severity assessment: LOW / MEDIUM / HIGH / CRITICAL.

    Based on anomaly_score_pct + magnitude signals.
    Severity is operationally meaningful (impact, urgency) and is separate
    from both the anomaly score and the classification confidence.
    """
    if not row["is_anomaly"]:
        return "—"

    ev    = row.get("evidence", {})
    score = ev.get("anomaly_score_pct", 0)
    mag   = ev.get("multivar_mag", 0)
    roc   = abs(ev.get("temp_roc", 0))
    rc    = row.get("root_cause", "")

    # Communication failures are always HIGH (data gap = station offline)
    if rc == RC_COMM:
        return "HIGH"

    # Frozen sensors are HIGH (ongoing data corruption)
    if rc == RC_FROZEN:
        return "HIGH"

    # Severity from combined signals
    if score >= 80 or mag >= 5.0 or roc >= 8.0:
        return "CRITICAL"
    elif score >= 60 or mag >= 3.5 or roc >= 4.0:
        return "HIGH"
    elif score >= 35 or mag >= 2.0 or roc >= 2.0:
        return "MEDIUM"
    else:
        return "LOW"


def _estimate_corrected_values(df: pd.DataFrame, nbrs: dict) -> pd.DataFrame:
    """
    Estimate the true underlying value continuously.
    
    Strategy:
    1. Compute rolling mean per station as the temporal baseline.
    2. Compute spatial mean per timestamp as the regional baseline.
    3. The continuous AI corrected estimate is the spatial mean (if neighbours exist),
       falling back to the station's rolling mean.
    """
    df = df.sort_values(["station_id", "timestamp"]).copy()
    
    # 1. Compute rolling mean per station
    for col in ["temp", "humidity", "pressure"]:
        df[f"{col}_rolling"] = df.groupby("station_id")[col].transform(
            lambda x: x.rolling(4, min_periods=1).mean().bfill()
        )
        
    # 2. Compute spatial mean per timestamp
    df_indexed = df.set_index(["timestamp", "station_id"])
    temp_lut = df_indexed["temp"]
    hum_lut  = df_indexed["humidity"]
    pres_lut = df_indexed["pressure"]
    
    spatial_t = []
    spatial_h = []
    spatial_p = []
    
    for _, row in df.iterrows():
        ts = row["timestamp"]
        sid = row["station_id"]
        peers = nbrs.get(sid, [])
        
        nt, nh, np_ = [], [], []
        for p in peers:
            key = (ts, p)
            if key in temp_lut.index:
                t, h, pr = temp_lut[key], hum_lut[key], pres_lut[key]
                if not pd.isna(t): nt.append(t)
                if not pd.isna(h): nh.append(h)
                if not pd.isna(pr): np_.append(pr)
                
        # If neighbours exist, use spatial mean, else fallback to station rolling mean
        if nt:
            spatial_t.append(sum(nt)/len(nt))
        else:
            spatial_t.append(row["temp_rolling"])
            
        if nh:
            spatial_h.append(sum(nh)/len(nh))
        else:
            spatial_h.append(row["humidity_rolling"])
            
        if np_:
            spatial_p.append(sum(np_)/len(np_))
        else:
            spatial_p.append(row["pressure_rolling"])
            
    df["temp_corrected"] = spatial_t
    df["humidity_corrected"] = spatial_h
    df["pressure_corrected"] = spatial_p
    
    # Cleanup
    df.drop(columns=["temp_rolling", "humidity_rolling", "pressure_rolling"], inplace=True)
    return df

