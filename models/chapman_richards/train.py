"""
models/chapman_richards/train.py
==================================
Table 4.1: Fits the Chapman-Richards growth curve to 2012 data,
then predicts 2023 tree heights using only tree AGE.

This is the simplest model in the project — no neural network, just a
mathematical formula:
    H(t) = y_max * (1 - exp(-k * t)) ^ p
where H is height, t is age, and y_max / k / p are learned from the data.

IMPORTANT: Run this FIRST. The PINN needs the fitted parameters saved here
as its "physics prior" (outputs/chapman_richards/cr_params.json).

Run with:
    python train.py
"""

import os
import json
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

import config
from common.data_utils import load_data
from common.metrics import reuben_metrics
from model import chapman_richards
from plots import plot_cr_fit, plot_spatial_error, plot_spatial_signed_error


def main():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. LOAD DATA                                                         #
    # ------------------------------------------------------------------ #
    # "Purged" means: only plots that appear in BOTH 2012 and 2023,
    # with "shrinking" plots (possible sensor noise) removed.
    # df12 = 2012 measurements (used for training)
    # df23 = 2023 measurements (used for testing)
    df12, df23, _ = load_data(config.DATA_PATH_PURGED)

    ages_train   = df12[config.AGE_COL].values
    heights_train = df12[config.TARGET_COL].values
    ages_test    = df23[config.AGE_COL].values
    heights_test  = df23[config.TARGET_COL].values

    # ------------------------------------------------------------------ #
    # 2. FIT THE CURVE                                                     #
    # ------------------------------------------------------------------ #
    # curve_fit finds y_max, k, p that minimise the squared error between
    # the formula and the real 2012 measurements.
    popt, _ = curve_fit(
        chapman_richards,
        ages_train, heights_train,
        p0=config.CR_P0,        # initial guess for the optimiser
        bounds=config.CR_BOUNDS,
        maxfev=50_000,
    )
    cr_ymax, cr_k, cr_p = popt
    print(f"\nFitted CR params:  y_max={cr_ymax:.4f}  k={cr_k:.7f}  p={cr_p:.4f}")

    # ------------------------------------------------------------------ #
    # 3. PREDICT & EVALUATE                                                #
    # ------------------------------------------------------------------ #
    # Plug 2023 ages into the fitted formula to get predicted 2023 heights.
    y_pred = chapman_richards(ages_test, cr_ymax, cr_k, cr_p)

    metrics = reuben_metrics(heights_test, y_pred, label="Chapman-Richards (Table 4.1)")

    # ------------------------------------------------------------------ #
    # 4. SAVE OUTPUTS                                                      #
    # ------------------------------------------------------------------ #
    # cr_params.json is the physics prior loaded by the PINN.
    params = {"Table4.1": {"y_max": float(cr_ymax), "k": float(cr_k), "p": float(cr_p)}}
    with open(os.path.join(config.OUTPUT_DIR, "cr_params.json"), "w") as f:
        json.dump(params, f, indent=2)

    pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"]).to_csv(
        os.path.join(config.OUTPUT_DIR, "results.csv"), index=False
    )

    # Curve plot: shows how well the formula matches the 2012 training data
    plot_cr_fit(ages_train, heights_train, cr_ymax, cr_k, cr_p, config.OUTPUT_DIR)

    # Spatial error maps: colour-coded map of where predictions are most/least accurate
    plot_spatial_error(
        df23["X"].values, df23["Y"].values,
        heights_test, y_pred,
        "(c) Chapman-Richards", config.OUTPUT_DIR,
    )
    plot_spatial_signed_error(
        df23["X"].values, df23["Y"].values,
        heights_test, y_pred,
        "(c) Chapman-Richards", config.OUTPUT_DIR,
    )

    print(f"\nAll outputs saved to: {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
