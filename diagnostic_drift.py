import pandas as pd
import numpy as np
from data_engine import generate_dataset, build_neighbor_map_from_df
from anomaly_engine import _engineer_features, run_pipeline

# 1. Run Pipeline
df, gt = generate_dataset()
nbrs = build_neighbor_map_from_df(df)

processed, _ = run_pipeline(df, nbrs)
features_df = _engineer_features(df.copy(), nbrs)

# 2. Merge all data
eval_df = processed[['timestamp', 'station_id', 'is_anomaly', 'anomaly_score_raw', 'anomaly_score_pct', 'root_cause']].copy()
gt_df = pd.DataFrame(gt)
eval_df = eval_df.merge(gt_df, on=['timestamp', 'station_id'], how='left')

final_df = eval_df.merge(features_df, on=['timestamp', 'station_id'], how='inner', suffixes=('', '_feat'))

# 3. Analyze Drift Rows
drift_rows = final_df[final_df['true_root_cause'] == 'Calibration Drift']
normal_rows = final_df[(final_df['is_anomaly_gt'] == False) & (final_df['true_root_cause'] != 'Genuine Weather Event (not a fault)')]

print("=== CALIBRATION DRIFT DIAGNOSTIC ===")
print(f"Total Drift Rows injected: {len(drift_rows)}")

stations = drift_rows['station_id'].unique()
print(f"Stations injected with drift: {list(stations)}")

print("\n--- Drift Row Progression (Kolkata Temp Drift) ---")
kol_drift = drift_rows[drift_rows['station_id'] == '0-20000-0-42807'].sort_values('timestamp')
for idx, row in kol_drift.iterrows():
    print(f"Time: {row['timestamp'].hour:02d}:00 | Temp: {row['temp']:.2f} | res: {row['temp_residual']:.2f} | slope: {row['feat_temp_drift_slope']:.4f} | spatial: {row['feat_spatial_temp_dev']:.2f} | IF: {row['anomaly_score_raw']:.3f} | Anomaly: {row['is_anomaly']}")

print("\n--- Feature Comparisons (Normal vs Drift) ---")
features_to_compare = ['temp_residual', 'feat_temp_drift_slope', 'feat_spatial_temp_dev', 'anomaly_score_raw']
for f in features_to_compare:
    if f in final_df.columns:
        norm_mean = normal_rows[f].mean()
        norm_std = normal_rows[f].std()
        drift_mean = drift_rows[f].mean()
        print(f"{f}:")
        print(f"  Normal: mean={norm_mean:.4f}, std={norm_std:.4f}")
        print(f"  Drift : mean={drift_mean:.4f}")

# IF Score distributions
print("\nIF Score percentiles (Normal):", np.percentile(normal_rows['anomaly_score_raw'], [50, 90, 95, 99]))
print("IF Score percentiles (Drift):", np.percentile(drift_rows['anomaly_score_raw'], [0, 50, 100]))
