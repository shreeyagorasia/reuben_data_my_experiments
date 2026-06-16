"""
models/chapman_richards/train.py
==================================
Fits the Chapman-Richards growth curve to the 2012 data, evaluates it
on the 2023 data, and saves everything into outputs/chapman_richards/:

  - cr_params.json   : fitted y_max, k, p
                        (the PINN baseline loads this as its physics prior)
                     - Reuben's params:   y_max=46.1126        k=0.0186698      p=1.0175
  - results.json     : evaluation metrics (MAE, MSE, R^2, MRE, Acc)
                     - Reuben: MAE=4.5605  MSE=31.0275  R²=0.2779  Acc=81.57%
  - cr_fit.png       : plot of the fitted curve vs 2012 data

Run with (from this folder):
    python train.py
"""

import os
import json
import csv

import config  # this model's config (also pulls in common settings)

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from common.data_utils import load_data, get_kfold_splits
from common.metrics import reuben_metrics
from model import chapman_richards
from plots import plot_cr_fit, plot_spatial_error

def run_cr(df_train, df_test, run_name=""):
    """Fits the CR model to the training set and evaluates it on the test set."""
    ages_train = df_train[config.AGE_COL].values
    y_train = df_train[config.TARGET_COL].values
    ages_test = df_test[config.AGE_COL].values
    y_test = df_test[config.TARGET_COL].values

    popt, _ = curve_fit(
        chapman_richards,
        ages_train, y_train,
        p0=config.CR_P0, bounds=config.CR_BOUNDS, maxfev=50_000
    )
    cr_ymax, cr_k, cr_p = popt

    y_pred = chapman_richards(ages_test, cr_ymax, cr_k, cr_p)
    
    if run_name:
        print(f"\n--- {run_name} ---")
        print(f"Fitted CR params: y_max={cr_ymax:.4f}  k={cr_k:.7f}  p={cr_p:.4f}")
        
    metrics = reuben_metrics(y_test, y_pred, label=run_name)
    params = {"y_max": float(cr_ymax), "k": float(cr_k), "p": float(cr_p)}
    return metrics, params, y_pred

def main():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    all_results = {}
    all_cr_params = {}

    # --- Experiments using PURGED data (Tables 4.1, 4.3, 4.4) ---
    print("\n\n--- Loading PURGED data for Tables 4.1, 4.3, 4.4 ---")
    df12_purged, df23_purged, _ = load_data(config.DATA_PATH_PURGED)

    # --- Experiment for Table 4.1 (Temporal, Common Plots) ---
    print("\n--- Running Experiment for Table 4.1: Temporal (Common Plots) ---")
    metrics_t1, params_t1, _ = run_cr(df12_purged, df23_purged, "Temporal (Table 4.1)")
    for k, v in metrics_t1.items():
        all_results[f"Table4.1_{k}"] = v
    all_cr_params["Table4.1"] = params_t1  # Save these specifically for the PINN's purged runs

    # --- Experiment for Table 4.3 (3-Fold CV within 2012) ---
    print(f"\n--- Running Experiment for Table 4.3: 3-Fold CV within 2012 (Purged) ---")
    cv12_metrics = []
    for fold, (tr_idx, te_idx) in enumerate(get_kfold_splits(len(df12_purged), n_splits=3)):
        m, _, _ = run_cr(df12_purged.iloc[tr_idx], df12_purged.iloc[te_idx])
        cv12_metrics.append(m)
    for k in cv12_metrics[0].keys():
        all_results[f"Table4.3_{k}"] = float(np.mean([m[k] for m in cv12_metrics]))

    # --- Experiment for Table 4.4 (3-Fold CV within 2023) ---
    print(f"\n--- Running Experiment for Table 4.4: 3-Fold CV within 2023 (Purged) ---")
    cv23_metrics = []
    for fold, (tr_idx, te_idx) in enumerate(get_kfold_splits(len(df23_purged), n_splits=3)):
        m, _, _ = run_cr(df23_purged.iloc[tr_idx], df23_purged.iloc[te_idx])
        cv23_metrics.append(m)
    for k in cv23_metrics[0].keys():
        all_results[f"Table4.4_{k}"] = float(np.mean([m[k] for m in cv23_metrics]))

    # --- Experiment using UNSEEN data (Table 4.2) ---
    print("\n\n--- Loading UNSEEN data for Table 4.2 ---")
    df12_unseen, df23_unseen, _ = load_data(config.DATA_PATH_UNSEEN)
    print(f"\n--- Running Experiment for Table 4.2: Temporal (Unseen Plots) ---")
    metrics_t2, params_t2, y_pred_t2 = run_cr(df12_unseen, df23_unseen, "Temporal (Table 4.2)")
    for k, v in metrics_t2.items():
        all_results[f"Table4.2_{k}"] = v
    all_cr_params["Table4.2"] = params_t2  # Save these for the PINN's unseen run

    # ------------------------------------------------------------
    # Save Outputs
    # ------------------------------------------------------------
    params_path = os.path.join(config.OUTPUT_DIR, "cr_params.json")
    with open(params_path, "w") as f:
        json.dump(all_cr_params, f, indent=2)
    print(f"\nSaved fitted parameters to {params_path}")

    pd.DataFrame(list(all_results.items()), columns=["Metric", "Value"]).to_csv(os.path.join(config.OUTPUT_DIR, "results.csv"), index=False)
    print(f"Saved combined metrics to results.csv")

    # Use the Table 4.2 params to generate the comprehensive visual map
    plot_cr_fit(df12_unseen[config.AGE_COL].values, df12_unseen[config.TARGET_COL].values, 
                params_t2["y_max"], params_t2["k"], params_t2["p"], config.OUTPUT_DIR)

    X_coords = df23_unseen["X"].values
    Y_coords = df23_unseen["Y"].values
    plot_spatial_error(X_coords, Y_coords, df23_unseen[config.TARGET_COL].values, y_pred_t2, "(c) Chapman-Richards", config.OUTPUT_DIR)


if __name__ == "__main__":
    main()
