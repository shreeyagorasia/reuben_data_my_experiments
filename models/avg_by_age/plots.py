"""
models/avg_by_age/plots.py
=============================
Contains plotting functions for the Avg by Age baseline.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

PINK_WHITE_BLUE = mcolors.LinearSegmentedColormap.from_list(
    "pink_white_blue", ["#e87fa0", "#ffffff", "#3b6fb6"]
)

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

def plot_spatial_signed_error(X_coords, Y_coords, y_test, y_pred, title, output_dir, filename="spatial_signed_error_map.png"):
    fig, ax = plt.subplots(figsize=(6, 5))
    signed_err = np.array(y_test) - np.array(y_pred)
    VMIN, VMAX = -15, 15

    hb = ax.hexbin(
        X_coords, Y_coords, C=signed_err, reduce_C_function=np.mean,
        gridsize=50, cmap=PINK_WHITE_BLUE, vmin=VMIN, vmax=VMAX, linewidths=0.2
    )

    mean_signed = signed_err.mean()
    ax.set_title(f"{title}\nMean Signed Error = {mean_signed:+.2f}m", fontsize=10, fontweight='bold')
    ax.set_xlabel("Easting (OS National Grid)", fontsize=8)
    ax.set_ylabel("Northing (OS National Grid)", fontsize=8)
    ax.tick_params(labelsize=7)

    cb = fig.colorbar(hb, ax=ax, fraction=0.035, pad=0.04)
    cb.set_label("Signed Error, actual - predicted (m)\npink = underpredicted, blue = overpredicted", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, filename)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved spatial signed error map to {plot_path}")
