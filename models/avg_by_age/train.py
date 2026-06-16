"""
models/avg_by_age/train.py
============================
Space-for-Time Substitution model.
Maps 2023 ages to the historical 2012 average height for that same age cohort.
"""

import os
import csv
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from common.data_utils import load_data, get_kfold_splits
from common.metrics import reuben_metrics

def plot_spatial_error(X_coords, Y_coords, y_test, y_pred, title, output_dir, filename="spatial_error_map.png"):
    fig, ax = plt.subplots(figsize=(6, 5))
    abs_err = np.abs(np.array(y_test) - np.array(y_pred))
    VMIN, VMAX = 0, 15
    cmap = plt.cm.viridis
    
    hb = ax.hexbin(
        X_coords, Y_coords, C=abs_err, reduce_C_function=np.mean,
        gridsize=50, cmap=cmap, vmin=VMIN, vmax=VMAX, linewidths=0.2
    )
    
    mae = abs_err.mean()
    acc = (1 - np.mean(abs_err / (np.array(y_test) + 1e-8))) * 100
    ax.set_title(f"{title}\nMAE = {mae:.2f}m   Acc = {acc:.1f}%", fontsize=10, fontweight='bold')
    ax.set_xlabel("Easting (OS National Grid)", fontsize=8)
    ax.set_ylabel("Northing (OS National Grid)", fontsize=8)
    ax.tick_params(labelsize=7)
    
    cb = fig.colorbar(hb, ax=ax, fraction=0.035, pad=0.04)
    cb.set_label("Mean Absolute Error (m)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, filename)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved spatial error map to {plot_path}")

def run_avg_by_age(df_train, df_test, run_name=""):
    """Runs the Space-for-Time cohort lookup logic for a given train/test split."""
    age_to_height = df_train.groupby(config.AGE_COL)[config.TARGET_COL].mean().sort_index()
    valid_ages = age_to_height.index.values
    
    age_test = df_test[config.AGE_COL].values
    y_test = df_test[config.TARGET_COL].values
    
    snapped_ages = np.array([valid_ages[np.abs(valid_ages - a).argmin()] for a in age_test])
    y_pred_avg_by_age = age_to_height.loc[snapped_ages].values
    
    if run_name:
        print(f"\n--- {run_name} ---")
    metrics = reuben_metrics(y_test, y_pred_avg_by_age, label=run_name)

    # R-squared diagnostic prints for the "unseen plots" experiment
    if "Table 4.2" in run_name:
        print("\n  [R² Diagnostics for Table 4.2]")
        print(f"    Lookup table size : {len(age_to_height)} unique ages")
        print(f"    Lookup table range: {age_to_height.min():.2f}m – {age_to_height.max():.2f}m (mean: {age_to_height.mean():.2f}m)")
        print(f"    Prediction stats  : mean={y_pred_avg_by_age.mean():.2f}, std={y_pred_avg_by_age.std():.2f}")
        print(f"    Test set stats    : mean={y_test.mean():.2f}, std={y_test.std():.2f}")
        # A low prediction variance (std) relative to the test set variance
        # will mechanically suppress the R² score, even if MAE is reasonable.
    
    return metrics, y_pred_avg_by_age

def main():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    all_results = {}

    # --- Experiments using PURGED data (Tables 4.1, 4.3, 4.4) ---
    print("\n\n--- Loading PURGED data for Tables 4.1, 4.3, 4.4 ---")
    df12_purged, df23_purged, _ = load_data(config.DATA_PATH_PURGED)

    # --- Experiment for Table 4.1 (Temporal, Common Plots) ---
    print("\n--- Running Experiment for Table 4.1: Temporal (Common Plots) ---")
    metrics_t1, _ = run_avg_by_age(df12_purged, df23_purged, "Temporal (Table 4.1)")
    for k, v in metrics_t1.items():
        all_results[f"Table4.1_{k}"] = v

    # --- Experiment for Table 4.3 (3-Fold CV within 2012) ---
    print(f"\n--- Running Experiment for Table 4.3: 3-Fold CV within 2012 (Purged) ---")
    cv12_metrics = []
    for fold, (tr_idx, te_idx) in enumerate(get_kfold_splits(len(df12_purged), n_splits=3)):
        m, _ = run_avg_by_age(df12_purged.iloc[tr_idx], df12_purged.iloc[te_idx])
        cv12_metrics.append(m)
    for k in cv12_metrics[0].keys():
        all_results[f"Table4.3_{k}"] = float(np.mean([m[k] for m in cv12_metrics]))

    # --- Experiment for Table 4.4 (3-Fold CV within 2023) ---
    print(f"\n--- Running Experiment for Table 4.4: 3-Fold CV within 2023 (Purged) ---")
    cv23_metrics = []
    for fold, (tr_idx, te_idx) in enumerate(get_kfold_splits(len(df23_purged), n_splits=3)):
        m, _ = run_avg_by_age(df23_purged.iloc[tr_idx], df23_purged.iloc[te_idx])
        cv23_metrics.append(m)
    for k in cv23_metrics[0].keys():
        all_results[f"Table4.4_{k}"] = float(np.mean([m[k] for m in cv23_metrics]))

    # --- Experiment using UNSEEN data (Table 4.2) ---
    print("\n\n--- Loading UNSEEN data for Table 4.2 ---")
    df12_unseen, df23_unseen, _ = load_data(config.DATA_PATH_UNSEEN)
    metrics_t2, y_pred_unseen = run_avg_by_age(df12_unseen, df23_unseen, "Temporal (Table 4.2)")
    for k, v in metrics_t2.items():
        all_results[f"Table4.2_{k}"] = v
        
    pd.DataFrame(list(all_results.items()), columns=["Metric", "Value"]).to_csv(os.path.join(config.OUTPUT_DIR, "results.csv"), index=False)
            
    # Spatial map for the comprehensive "unseen" experiment
    X_coords = df23_unseen["X"].values
    Y_coords = df23_unseen["Y"].values
    plot_spatial_error(X_coords, Y_coords, df23_unseen[config.TARGET_COL].values, y_pred_unseen, "(b) AvgByAge Baseline", config.OUTPUT_DIR)

if __name__ == "__main__":
    main()
