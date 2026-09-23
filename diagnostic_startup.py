import time
import json
import pandas as pd
from data_source import SimulatedDataSource
from anomaly_engine import run_pipeline, explain_with_shap, sensor_health
import data_engine

def run_diagnostic():
    t_start = time.perf_counter()
    
    t0 = time.perf_counter()
    with open('stations_india.json', 'r', encoding='utf-8') as f:
        STATIONS = json.load(f)
    t_registry = time.perf_counter() - t0
    
    t0 = time.perf_counter()
    src = SimulatedDataSource()
    sensor_df = src.get_sensor_df()
    t_sim = time.perf_counter() - t0
    
    t0 = time.perf_counter()
    nbrs = src.get_neighbor_map()
    t_nbrs = time.perf_counter() - t0
    
    t0 = time.perf_counter()
    processed, clf = run_pipeline(sensor_df, nbrs)
    t_pipeline = time.perf_counter() - t0
    
    t0 = time.perf_counter()
    try:
        # Note: explain_with_shap trains a Random Forest! This might be slow
        shap_res = explain_with_shap(processed, clf)
    except Exception as e:
        shap_res = str(e)
    t_shap = time.perf_counter() - t0
    
    t0 = time.perf_counter()
    health = sensor_health(processed)
    t_health = time.perf_counter() - t0
    
    t_total = time.perf_counter() - t_start
    
    print("\n[PERFORMANCE]")
    print(f"registry: {t_registry:.3f} sec")
    print(f"simulation generation: {t_sim:.3f} sec")
    print(f"neighbor map: {t_nbrs:.3f} sec")
    print(f"feature eng & Isolation Forest (run_pipeline): {t_pipeline:.3f} sec")
    print(f"SHAP (includes Surrogate RF): {t_shap:.3f} sec")
    print(f"health calculation: {t_health:.3f} sec")
    print(f"TOTAL core processing: {t_total:.3f} sec")

if __name__ == "__main__":
    run_diagnostic()
