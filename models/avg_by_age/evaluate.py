"""
models/avg_by_age/evaluate.py
================================
Regenerates spatial plots for the AvgByAge model. Run this locally after
copying outputs/avg_by_age/ from the cluster.

Run with (from this folder):
    python evaluate.py
"""

import numpy as np
import config
from common.data_utils import load_data
from plots import plot_spatial_error, plot_spatial_signed_error


def main():
    df12, df23, _ = load_data(config.DATA_PATH_UNSEEN)

    age_to_height = df12.groupby(config.AGE_COL)[config.TARGET_COL].mean().sort_index()
    valid_ages = age_to_height.index.values

    age_test = df23[config.AGE_COL].values
    y_test = df23[config.TARGET_COL].values
    snapped_ages = np.array([valid_ages[np.abs(valid_ages - a).argmin()] for a in age_test])
    y_pred = age_to_height.loc[snapped_ages].values

    X_coords = df23["X"].values
    Y_coords = df23["Y"].values

    plot_spatial_error(X_coords, Y_coords, y_test, y_pred, "(b) AvgByAge Baseline", config.OUTPUT_DIR)
    plot_spatial_signed_error(X_coords, Y_coords, y_test, y_pred, "(b) AvgByAge Baseline", config.OUTPUT_DIR)


if __name__ == "__main__":
    main()
