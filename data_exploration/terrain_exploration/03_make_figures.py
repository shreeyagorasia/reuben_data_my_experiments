"""
Make exploratory terrain diagnostic figures.

Figures:
1. terrain_maps.png
2. residual_alignment.png
3. terrain_residual_scatter.png
4. terrain_binned_errors.png
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data_exploration" / "terrain_exploration_output"
DEFAULT_TABLE = OUTPUT_DIR / "terrain_residual_table.csv"
GRID_SIZE = 90


def parse_args():
    parser = argparse.ArgumentParser(description="Create terrain/residual exploratory figures.")
    parser.add_argument("--input", default=DEFAULT_TABLE, help="terrain_residual_table.csv from step 02.")
    return parser.parse_args()


def robust_limits(values, symmetric=False):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if symmetric:
        vmax = max(0.05, float(np.percentile(np.abs(values), 98)))
        return -vmax, vmax
    return float(np.percentile(values, 2)), float(np.percentile(values, 98))


def add_hexbin(ax, df, value_col, title, cmap, symmetric=False, label=None):
    vmin, vmax = robust_limits(df[value_col], symmetric=symmetric)
    hb = ax.hexbin(
        df["X"],
        df["Y"],
        C=df[value_col],
        reduce_C_function=np.mean,
        gridsize=GRID_SIZE,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidths=0.2,
    )
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Easting (OS National Grid)", fontsize=9)
    ax.set_ylabel("Northing (OS National Grid)", fontsize=9)
    ax.tick_params(labelsize=8)
    cb = plt.colorbar(hb, ax=ax, fraction=0.035, pad=0.04)
    cb.set_label(label or value_col, fontsize=8)


def terrain_maps(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharex=True, sharey=True)
    add_hexbin(axes[0], df, "DEM_ELEVATION", "DEM Elevation", "terrain", label="Elevation (m)")
    add_hexbin(axes[1], df, "DEM_RUGGEDNESS", "DEM Ruggedness", "viridis", label="3x3 local std dev (m)")
    fig.suptitle("Terrain Structure Across Table 4.1 Plots", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "terrain_maps.png", dpi=250, bbox_inches="tight")
    plt.close()


def residual_alignment(df):
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10), sharex=True, sharey=True)
    add_hexbin(
        axes[0, 0],
        df,
        "PINN_SIGNED_REL_ERROR",
        "PINN Signed Relative Error",
        "coolwarm",
        symmetric=True,
        label="(actual - predicted) / actual",
    )
    add_hexbin(
        axes[0, 1],
        df,
        "CR_SIGNED_REL_ERROR",
        "CR Signed Relative Error",
        "coolwarm",
        symmetric=True,
        label="(actual - predicted) / actual",
    )
    add_hexbin(axes[1, 0], df, "DEM_ELEVATION", "DEM Elevation", "terrain", label="Elevation (m)")
    add_hexbin(axes[1, 1], df, "DEM_RUGGEDNESS", "DEM Ruggedness", "viridis", label="3x3 local std dev (m)")
    fig.suptitle("Do Residual Error Patterns Align With Terrain?", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "residual_alignment.png", dpi=250, bbox_inches="tight")
    plt.close()


def scatter_panel(ax, df, x_col, y_col):
    valid = df[[x_col, y_col]].dropna()
    pearson = valid[x_col].corr(valid[y_col], method="pearson")
    spearman = valid[x_col].corr(valid[y_col], method="spearman")
    ax.scatter(valid[x_col], valid[y_col], s=4, alpha=0.2, linewidths=0)
    ax.axhline(0, color="black", lw=0.8)
    grouped = valid.assign(_bin=pd.qcut(valid[x_col], q=20, duplicates="drop")).groupby("_bin", observed=False)
    trend = grouped[[x_col, y_col]].mean()
    ax.plot(trend[x_col], trend[y_col], color="black", lw=1.3)
    ax.set_xlabel(x_col)
    ax.set_ylabel(f"{y_col} (m)")
    ax.set_title(f"{x_col} vs {y_col}\nr={pearson:+.3f}, rho={spearman:+.3f}", fontsize=10)


def scatter_figure(df):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9.5))
    scatter_panel(axes[0, 0], df, "DEM_ELEVATION", "CR_RESIDUAL_2023")
    scatter_panel(axes[0, 1], df, "DEM_ELEVATION", "PINN_RESIDUAL_2023")
    scatter_panel(axes[1, 0], df, "DEM_RUGGEDNESS", "CR_RESIDUAL_2023")
    scatter_panel(axes[1, 1], df, "DEM_RUGGEDNESS", "PINN_RESIDUAL_2023")
    fig.suptitle("Terrain Features vs Model Residuals", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "terrain_residual_scatter.png", dpi=250, bbox_inches="tight")
    plt.close()


def binned_errors(df):
    data = df.copy()
    data["Elevation bin"] = pd.cut(
        data["DEM_ELEVATION"],
        bins=[0, 100, 200, 300, 400, np.inf],
        labels=["0-100m", "100-200m", "200-300m", "300-400m", "400m+"],
        include_lowest=True,
    )
    data["Ruggedness quartile"] = pd.qcut(
        data["DEM_RUGGEDNESS"],
        q=4,
        labels=["Q1 lowest", "Q2", "Q3", "Q4 highest"],
        duplicates="drop",
    )
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
    plots = [
        ("Elevation bin", "PINN_REL_ERROR", "PINN Relative Error by Elevation"),
        ("Elevation bin", "CR_RESIDUAL_2023", "CR Residual by Elevation"),
        ("Ruggedness quartile", "PINN_REL_ERROR", "PINN Relative Error by Ruggedness"),
        ("Ruggedness quartile", "CR_RESIDUAL_2023", "CR Residual by Ruggedness"),
    ]
    for ax, (group_col, value_col, title) in zip(axes.flatten(), plots):
        groups = [sub[value_col].dropna().values for _, sub in data.groupby(group_col, observed=False)]
        labels = [str(label) for label, _ in data.groupby(group_col, observed=False)]
        ax.boxplot(groups, tick_labels=labels, showfliers=False)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_ylabel(value_col)
        ax.tick_params(axis="x", rotation=25)
    fig.suptitle("Binned Terrain Diagnostics", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "terrain_binned_errors.png", dpi=250, bbox_inches="tight")
    plt.close()


def write_figure_manifest():
    manifest = {
        "figures": {
            "terrain_maps.png": "Elevation and ruggedness hexbin maps.",
            "residual_alignment.png": "PINN/CR signed relative error beside elevation/ruggedness.",
            "terrain_residual_scatter.png": "Scatter plots with Pearson/Spearman correlations and binned trend lines.",
            "terrain_binned_errors.png": "Boxplots of error/residual distributions by terrain bins.",
        }
    }
    with open(OUTPUT_DIR / "figure_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table_path = Path(args.input)
    if not table_path.exists():
        raise SystemExit(f"Missing {table_path}. Run 02_residual_diagnostics.py first.")
    df = pd.read_csv(table_path)
    terrain_maps(df)
    residual_alignment(df)
    scatter_figure(df)
    binned_errors(df)
    write_figure_manifest()
    print(f"Saved terrain diagnostic figures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
