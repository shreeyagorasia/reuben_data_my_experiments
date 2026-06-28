"""
Compare where PINN is spatially stronger/weaker than simple baselines.

This script uses Table 4.1 temporal common-plot data and plots:

    PINN relative error - baseline relative error

where:

    relative error = abs(y_true - y_pred) / (abs(y_true) + 1e-8)

Interpretation:
    negative / blue = PINN has lower relative error than the baseline
    positive / red  = PINN has higher relative error than the baseline
    near-zero / white = PINN and the baseline are similarly wrong/right

The panels are:
    1. PINN - Chapman-Richards
       Shows where PINN improves over, or underperforms, a traditional
       age-growth curve.

    2. PINN - AvgByAge
       Shows where PINN's spatial/site/land-use features help beyond a
       simple age-cohort average.

The script prints summary statistics to the terminal:
    mean, min, max, and the fraction of plots where PINN is worse.

It writes:
    outputs/analysis/model_comparison/compare_plot_subtract.png

Usage from repo root:

    python analysis/model_comparison/compare_plot_subtract.py

Optional:

    python analysis/model_comparison/compare_plot_subtract.py --vmax 0.25
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp")
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import config as common_config  # noqa: E402
from common.data_utils import build_feature_arrays, load_data, split_age_column  # noqa: E402
from common.spatial_plots import GRID_SIZE, REDUCE_C_FUNCTION  # noqa: E402


PINN_DIR = PROJECT_ROOT / "models" / "pinn_baseline"
if str(PINN_DIR) not in sys.path:
    sys.path.insert(0, str(PINN_DIR))
import config as pinn_config  # noqa: E402
from model import PINN  # noqa: E402


OUTPUT_DIR = PROJECT_ROOT / "data_exploration" / "model_comparison_output"
PINN_CHECKPOINT = PROJECT_ROOT / "outputs" / "pinn_baseline" / "checkpoint.pt"
CR_PARAMS = PROJECT_ROOT / "outputs" / "chapman_richards" / "cr_params.json"

ERROR_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "blue_white_red_difference", ["#3b6fb6", "#ffffff", "#d94b62"]
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot spatial relative-error differences: PINN minus CR and AvgByAge."
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="Symmetric color limit for relative-error difference. Default uses the robust 95th percentile.",
    )
    return parser.parse_args()


def chapman_richards(t, y_max, k, p):
    return y_max * (1 - np.exp(-k * t)) ** p


def predict_avg_by_age(df_train, df_test):
    age_to_height = df_train.groupby(common_config.AGE_COL)[common_config.TARGET_COL].mean().sort_index()
    valid_ages = age_to_height.index.values
    age_test = df_test[common_config.AGE_COL].values
    snapped_ages = np.array([valid_ages[np.abs(valid_ages - age).argmin()] for age in age_test])
    return age_to_height.loc[snapped_ages].values


def predict_cr(df_test):
    with open(CR_PARAMS) as f:
        all_params = json.load(f)
    params = all_params["Table4.1"]
    return chapman_richards(
        df_test[common_config.AGE_COL].values,
        params["y_max"],
        params["k"],
        params["p"],
    )


def predict_pinn(X_train, X_test):
    checkpoint = torch.load(PINN_CHECKPOINT, map_location="cpu", weights_only=False)
    saved_features = checkpoint.get("feature_list")
    if saved_features is not None and list(saved_features) != list(common_config.FEATURES):
        raise ValueError(
            "PINN checkpoint feature list does not match common_config.FEATURES. "
            "Retrain PINN before using this comparison script."
        )

    _, _, X_test_other, X_test_age, _, _ = split_age_column(X_train, X_test)
    scaler_Xo = checkpoint["scaler_Xo"]
    scaler_age = checkpoint["scaler_age"]
    scaler_y = checkpoint["scaler_y"]
    other_idxs = checkpoint["other_idxs"]

    model = PINN(n_other=len(other_idxs), hidden_size=pinn_config.HIDDEN_SIZE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    Xte_o = torch.tensor(scaler_Xo.transform(X_test_other), dtype=torch.float32)
    Xte_a = torch.tensor(scaler_age.transform(X_test_age), dtype=torch.float32)
    with torch.no_grad():
        y_pred_scaled = model(Xte_o, Xte_a).numpy().reshape(-1, 1)
    return scaler_y.inverse_transform(y_pred_scaled).ravel()


def relative_error(y_true, y_pred):
    return np.abs(y_true - y_pred) / (np.abs(y_true) + 1e-8)


def robust_symmetric_limit(values, user_vmax=None):
    if user_vmax is not None:
        return float(user_vmax)
    robust = float(np.percentile(np.abs(values), 95))
    return max(0.05, robust)


def describe(name, diff):
    print(
        f"{name}: mean={diff.mean():+.4f}, min={diff.min():+.4f}, "
        f"max={diff.max():+.4f}, PINN worse={(diff > 0).mean() * 100:.1f}% of plots"
    )


def panel_summary(diff):
    return (
        f"PINN worse: {(diff > 0).mean() * 100:.1f}% of plots | "
        f"mean diff: {diff.mean():+.3f}"
    )


def plot_difference_maps(x, y, diff_cr, diff_avg, vmax):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharex=True, sharey=True)
    panels = [
        ("PINN - Chapman-Richards", diff_cr),
        ("PINN - AvgByAge", diff_avg),
    ]

    hb = None
    for ax, (title, diff) in zip(axes, panels):
        hb = ax.hexbin(
            x,
            y,
            C=diff,
            reduce_C_function=REDUCE_C_FUNCTION,
            gridsize=GRID_SIZE,
            cmap=ERROR_CMAP,
            vmin=-vmax,
            vmax=vmax,
            linewidths=0.2,
        )
        ax.set_title(f"{title}\n{panel_summary(diff)}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Easting (OS National Grid)", fontsize=9)
        ax.set_ylabel("Northing (OS National Grid)", fontsize=9)
        ax.tick_params(labelsize=8)

    fig.suptitle("Where PINN Improves or Worsens Relative Error", fontsize=15, fontweight="bold", y=1.03)
    fig.text(
        0.5,
        0.97,
        "Colour = PINN relative error minus baseline relative error; blue = PINN better, red = PINN worse.",
        ha="center",
        va="top",
        fontsize=10,
        color="dimgray",
    )
    fig.text(
        0.5,
        0.93,
        "PINN worse % = share of Table 4.1 plots where PINN relative error is higher than the baseline's.",
        ha="center",
        va="top",
        fontsize=10,
        color="dimgray",
    )
    fig.text(
        0.5,
        0.895,
        "Each panel is a spatial hexbin map over OS National Grid coordinates.",
        ha="center",
        va="top",
        fontsize=10,
        color="dimgray",
    )
    cbar = fig.colorbar(hb, ax=axes, fraction=0.035, pad=0.04)
    cbar.set_label("Relative error difference (PINN - baseline)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "compare_plot_subtract.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved subtractive comparison plot to {output_path}")


def main():
    args = parse_args()

    df12, df23, _ = load_data(common_config.DATA_PATH_PURGED)
    X_train, X_test, _, y_true, x_coords, y_coords = build_feature_arrays(df12, df23)

    print("\nGenerating Table 4.1 predictions...")
    y_pred_pinn = predict_pinn(X_train, X_test)
    y_pred_cr = predict_cr(df23)
    y_pred_avg = predict_avg_by_age(df12, df23)

    rel_pinn = relative_error(y_true, y_pred_pinn)
    rel_cr = relative_error(y_true, y_pred_cr)
    rel_avg = relative_error(y_true, y_pred_avg)

    diff_cr = rel_pinn - rel_cr
    diff_avg = rel_pinn - rel_avg

    describe("PINN - Chapman-Richards relative error", diff_cr)
    describe("PINN - AvgByAge relative error", diff_avg)

    vmax = robust_symmetric_limit(np.concatenate([diff_cr, diff_avg]), args.vmax)
    print(f"Using symmetric colour scale: vmin={-vmax:.4f}, vmax={vmax:.4f}, gridsize={GRID_SIZE}")

    plot_difference_maps(x_coords, y_coords, diff_cr, diff_avg, vmax)


if __name__ == "__main__":
    main()
