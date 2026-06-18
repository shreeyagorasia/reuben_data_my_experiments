"""
models/avg_by_age/train.py
============================
Space-for-Time Substitution model.
Maps 2023 ages to the historical 2012 average height for that same age cohort.
"""

import os
import json
import numpy as np
import pandas as pd

import config
from common.data_utils import load_data, get_kfold_splits
from common.metrics import reuben_metrics
from plots import plot_spatial_error, plot_spatial_signed_error


def save_table_4_2_artifacts(df_train, df_test, common_ids, metrics, y_pred):
    """Save the non-neural artifacts needed to audit/recreate Table 4.2."""
    age_to_height = df_train.groupby(config.AGE_COL)[config.TARGET_COL].agg(["mean", "count"]).sort_index()
    age_to_height = age_to_height.rename(columns={"mean": "predicted_top_height", "count": "train_rows"})
    valid_ages = age_to_height.index.values

    age_test = df_test[config.AGE_COL].values
    snapped_ages = np.array([valid_ages[np.abs(valid_ages - a).argmin()] for a in age_test])
    exact_age_match = np.isin(age_test, valid_ages)

    common_id_set = set(common_ids)
    predictions = pd.DataFrame({
        "PLOT_ID": df_test.index.values,
        "X": df_test["X"].values,
        "Y": df_test["Y"].values,
        "AGE": age_test,
        "snapped_age": snapped_ages,
        "age_exact_match": exact_age_match,
        "is_common_2023_plot": [plot_id in common_id_set for plot_id in df_test.index.values],
        "y_true": df_test[config.TARGET_COL].values,
        "y_pred": y_pred,
    })
    predictions["abs_error"] = np.abs(predictions["y_true"] - predictions["y_pred"])
    predictions["signed_error_actual_minus_pred"] = predictions["y_true"] - predictions["y_pred"]

    lookup_path = os.path.join(config.OUTPUT_DIR, "table4_2_age_lookup.csv")
    predictions_path = os.path.join(config.OUTPUT_DIR, "table4_2_predictions.csv")
    config_path = os.path.join(config.OUTPUT_DIR, "config_used.json")

    age_to_height.to_csv(lookup_path, index_label=config.AGE_COL)
    predictions.to_csv(predictions_path, index=False)

    only_2023_mask = ~predictions["is_common_2023_plot"]
    only_2023 = predictions.loc[only_2023_mask]
    no_exact_only_2023 = int((~only_2023["age_exact_match"]).sum())

    config_used = {
        "model_name": config.MODEL_NAME,
        "method": "AvgByAge nearest-age lookup",
        "data_paths": {
            "purged": config.DATA_PATH_PURGED,
            "unseen": config.DATA_PATH_UNSEEN,
        },
        "features_used": [config.AGE_COL],
        "target_col": config.TARGET_COL,
        "random_seed": config.RANDOM_SEED,
        "tables_run": ["4.1", "4.3", "4.4", "4.2"],
        "table4_2": {
            "train_rows_2012_post_purge": int(len(df_train)),
            "test_rows_2023_post_purge": int(len(df_test)),
            "common_plot_count": int(len(common_ids)),
            "unique_2023_only_count": int(only_2023_mask.sum()),
            "age_lookup_unique_ages": int(len(valid_ages)),
            "age_lookup_min": float(valid_ages.min()),
            "age_lookup_max": float(valid_ages.max()),
            "fallback_logic": "nearest AGE by np.abs(valid_ages - age).argmin()",
            "unique_2023_rows_without_exact_age_match": no_exact_only_2023,
            "unique_2023_fraction_without_exact_age_match": (
                float(no_exact_only_2023 / len(only_2023)) if len(only_2023) else 0.0
            ),
            "metrics": metrics,
        },
        "output_files": {
            "results": "results.csv",
            "spatial_error_map": "spatial_error_map.png",
            "spatial_signed_error_map": "spatial_signed_error_map.png",
            "age_lookup": "table4_2_age_lookup.csv",
            "predictions": "table4_2_predictions.csv",
            "config_used": "config_used.json",
        },
    }
    with open(config_path, "w") as f:
        json.dump(config_used, f, indent=2)

    print(f"Saved Table 4.2 age lookup to {lookup_path}")
    print(f"Saved Table 4.2 predictions to {predictions_path}")
    print(f"Saved run details to {config_path}")


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
    df12_unseen, df23_unseen, common_ids_unseen = load_data(config.DATA_PATH_UNSEEN)
    metrics_t2, y_pred_unseen = run_avg_by_age(df12_unseen, df23_unseen, "Temporal (Table 4.2)")
    for k, v in metrics_t2.items():
        all_results[f"Table4.2_{k}"] = v

    X_coords_unseen = df23_unseen["X"].values
    Y_coords_unseen = df23_unseen["Y"].values
    y_test_unseen = df23_unseen[config.TARGET_COL].values
    plot_spatial_error(
        X_coords_unseen,
        Y_coords_unseen,
        y_test_unseen,
        y_pred_unseen,
        "(b) AvgByAge Baseline",
        config.OUTPUT_DIR,
    )
    plot_spatial_signed_error(
        X_coords_unseen,
        Y_coords_unseen,
        y_test_unseen,
        y_pred_unseen,
        "(b) AvgByAge Baseline",
        config.OUTPUT_DIR,
    )
    save_table_4_2_artifacts(
        df12_unseen,
        df23_unseen,
        common_ids_unseen,
        metrics_t2,
        y_pred_unseen,
    )
        
    pd.DataFrame(list(all_results.items()), columns=["Metric", "Value"]).to_csv(os.path.join(config.OUTPUT_DIR, "results.csv"), index=False)
    print(f"Saved combined metrics to results.csv")

if __name__ == "__main__":
    main()
