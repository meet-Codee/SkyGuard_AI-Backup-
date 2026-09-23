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
        n_faults     = n_anomalies - n_genuine
        n_missing    = int(grp["was_nan"].sum()) if "was_nan" in grp.columns else 0
        
        rate = round(n_faults / n_readings * 100, 1) if n_readings > 0 else 0.0
        missing_frac = n_missing / n_readings if n_readings > 0 else 0.0

        base_score = 100.0 - rate * 4.0
        has_drift = "Calibration Drift" in grp.get("root_cause", pd.Series(dtype=str)).values
        drift_penalty = 15.0 if has_drift else 0.0
        has_freeze = "Frozen / Stuck Sensor" in grp.get("root_cause", pd.Series(dtype=str)).values
        freeze_penalty = 20.0 if has_freeze else 0.0
        missing_penalty = 10.0 * missing_frac

        sev_penalty = 0.0
        if "severity" in grp.columns:
            has_high = grp[grp["is_anomaly"] & (grp["root_cause"] != "Genuine Weather Event (not a fault)")]["severity"].isin(["HIGH", "CRITICAL"]).any()
            sev_penalty = 10.0 if has_high else 0.0

        health_score = max(0.0, base_score - drift_penalty - freeze_penalty - missing_penalty - sev_penalty)
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
