"""
models/chapman_richards/evaluate.py
=====================================
Loads cr_params.json and regenerates all plots. Run this locally after
copying outputs/chapman_richards/ from the cluster.

Run with (from this folder):
    python evaluate.py
"""

import os
import json
import numpy as np

import config
from common.data_utils import load_data
from model import chapman_richards
from plots import plot_cr_fit, plot_spatial_error, plot_spatial_signed_error


def main():
    params_path = os.path.join(config.OUTPUT_DIR, "cr_params.json")
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"Could not find {params_path}. Run train.py first.")

    with open(params_path) as f:
        all_cr_params = json.load(f)

    params = all_cr_params.get("Table4.2", all_cr_params)
    print(f"Loaded CR params: {params}")

    df12, df23, _ = load_data(config.DATA_PATH_UNSEEN)

    ages_test = df23[config.AGE_COL].values
    y_test = df23[config.TARGET_COL].values
    y_pred = chapman_richards(ages_test, params["y_max"], params["k"], params["p"])

    X_coords = df23["X"].values
    Y_coords = df23["Y"].values

    plot_cr_fit(df12[config.AGE_COL].values, df12[config.TARGET_COL].values,
                params["y_max"], params["k"], params["p"], config.OUTPUT_DIR)
    plot_spatial_error(X_coords, Y_coords, y_test, y_pred, "(c) Chapman-Richards", config.OUTPUT_DIR)
    plot_spatial_signed_error(X_coords, Y_coords, y_test, y_pred, "(c) Chapman-Richards", config.OUTPUT_DIR)


if __name__ == "__main__":
    main()
