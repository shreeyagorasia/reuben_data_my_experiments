"""
models/avg_by_age/train.py
============================
Table 4.1: Space-for-Time Substitution baseline.

The idea: "What was the average height of a 30-year-old tree in 2012?
That's probably what a 30-year-old tree in 2023 looks like too."

So we build a simple lookup table:  AGE → average 2012 height
then use it to predict 2023 heights.

No neural network. No fitting. Just a group-by + lookup.

Run with:
    python train.py
"""

import os
import numpy as np
import pandas as pd

import config
from common.data_utils import load_data
from common.metrics import reuben_metrics
from plots import plot_spatial_error, plot_spatial_signed_error


def main():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. LOAD DATA                                                         #
    # ------------------------------------------------------------------ #
    df12, df23, _ = load_data(config.DATA_PATH_PURGED)

    # ------------------------------------------------------------------ #
    # 2. BUILD THE LOOKUP TABLE                                            #
    # ------------------------------------------------------------------ #
    # Group 2012 plots by age, take the mean height for each age bucket.
    # Result: a pandas Series where  age → mean height
    age_to_height = df12.groupby(config.AGE_COL)[config.TARGET_COL].mean()
    valid_ages = age_to_height.index.values

    # ------------------------------------------------------------------ #
    # 3. PREDICT                                                           #
    # ------------------------------------------------------------------ #
    # For each 2023 plot, find the closest age in our lookup table and
    # return the corresponding mean height.
    # (Some 2023 ages may not exist in the 2012 data, so we snap to nearest.)
    ages_test    = df23[config.AGE_COL].values
    heights_test  = df23[config.TARGET_COL].values

    snapped_ages = np.array([valid_ages[np.abs(valid_ages - a).argmin()] for a in ages_test])
    y_pred = age_to_height.loc[snapped_ages].values

    # ------------------------------------------------------------------ #
    # 4. EVALUATE & SAVE                                                   #
    # ------------------------------------------------------------------ #
    metrics = reuben_metrics(heights_test, y_pred, label="AvgByAge (Table 4.1)")

    pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"]).to_csv(
        os.path.join(config.OUTPUT_DIR, "results.csv"), index=False
    )

    # Spatial error maps: colour-coded map of where predictions are most/least accurate
    plot_spatial_error(
        df23["X"].values, df23["Y"].values,
        heights_test, y_pred,
        "(b) AvgByAge Baseline", config.OUTPUT_DIR,
    )
    plot_spatial_signed_error(
        df23["X"].values, df23["Y"].values,
        heights_test, y_pred,
        "(b) AvgByAge Baseline", config.OUTPUT_DIR,
    )

    print(f"\nAll outputs saved to: {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
