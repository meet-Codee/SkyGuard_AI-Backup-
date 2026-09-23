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
