import pandas as pd
import numpy as np
from data_engine import generate_dataset, build_neighbor_map_from_df
from anomaly_engine import _engineer_features

def test_causality():
    print("Generating dataset...")
    df_full, _ = generate_dataset()
    nbrs = build_neighbor_map_from_df(df_full)
    
    # 1. Base run on full dataset
    df_base = _engineer_features(df_full.copy(), nbrs)
    
    # Target time to test: Day 2, 14:00 (index ~38 for a specific station)
    target_ts = df_full["timestamp"].unique()[38]
    print(f"Testing causality at timestamp: {target_ts}")
    
    # Get base expected_temp and residual for target time, station 'BLR-01'
    base_row = df_base[(df_base["timestamp"] == target_ts) & (df_base["station_id"] == "BLR-01")].iloc[0]
    base_expected = base_row["expected_temp"]
    base_residual = base_row["temp_residual"]
    
    # 2. Corrupt future data (t > target_ts)
    df_corrupt = df_full.copy()
    future_mask = df_corrupt["timestamp"] > target_ts
    df_corrupt.loc[future_mask, "temp"] += 100.0  # Massive corruption
    
    df_corr_res = _engineer_features(df_corrupt, nbrs)
    corr_row = df_corr_res[(df_corr_res["timestamp"] == target_ts) & (df_corr_res["station_id"] == "BLR-01")].iloc[0]
    
    print("\n--- CAUSALITY TEST RESULTS ---")
    print(f"Base expected_temp: {base_expected:.4f} | Base residual: {base_residual:.4f}")
    print(f"Corr expected_temp: {corr_row['expected_temp']:.4f} | Corr residual: {corr_row['temp_residual']:.4f}")
    
    if np.isclose(base_expected, corr_row['expected_temp']) and np.isclose(base_residual, corr_row['temp_residual']):
        print("PASSED: expected_temp and temp_residual are unaffected by future data.")
    else:
        print("FAILED: FUTURE DATA LEAKAGE DETECTED!")

if __name__ == "__main__":
    test_causality()
