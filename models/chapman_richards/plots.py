"""
models/chapman_richards/plots.py
==================================
Contains plotting functions for the Chapman-Richards model.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from model import chapman_richards

def plot_cr_fit(ages_2012, y_train, cr_ymax, cr_k, cr_p, output_dir):
    """Plots the fitted Chapman-Richards curve against the 2012 data."""
    # Generate 200 sorted, evenly spaced ages for smooth line plot
    t_range = np.linspace(20, 120, 200)
    plt.figure(figsize=(8, 4))
    # Plot the 2012 points (s=1 makes them small dots, alpha=0.2 makes them semi-transparent)
    plt.scatter(ages_2012, y_train, s=1, alpha=0.2, color="steelblue", label="2012 data")
    # Plot the smooth theoretical curve from fitted parameters
    plt.plot(
        t_range, chapman_richards(t_range, cr_ymax, cr_k, cr_p),
        color="firebrick", lw=2,
        label=f"CR fit: ymax={cr_ymax:.1f} k={cr_k:.4f} p={cr_p:.3f}",
    )
    plt.xlabel("Age (years)")
    plt.ylabel("Top Height (m)")
    plt.title("Chapman-Richards fit on 2012 data")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "cr_fit.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()  # Close the figure to free up memory
    print(f"Saved plot to {plot_path}")

def plot_spatial_error(X_coords, Y_coords, y_test, y_pred, title, output_dir, filename="spatial_error_map.png"):
    """Plots a hexbin map of Mean Absolute Error across spatial coordinates."""
    fig, ax = plt.subplots(figsize=(6, 5))

    abs_err = np.abs(np.array(y_test) - np.array(y_pred))
    VMIN, VMAX = 0, 10
    cmap = plt.cm.viridis

    hb = ax.hexbin(
        X_coords, Y_coords,
        C=abs_err,
        reduce_C_function=np.mean,
        gridsize=50,
        cmap=cmap,
        vmin=VMIN,
        vmax=VMAX,
        linewidths=0.2,
    )

    mae  = abs_err.mean()
    acc  = (1 - np.mean(abs_err / (np.array(y_test) + 1e-8))) * 100

    ax.set_title(f"{title}\nMAE = {mae:.2f}m   Acc = {acc:.1f}%", fontsize=10, fontweight='bold')
    ax.set_xlabel("Easting (OS National Grid)", fontsize=8)
    ax.set_ylabel("Northing (OS National Grid)", fontsize=8)
    ax.tick_params(labelsize=7)

    cb = fig.colorbar(hb, ax=ax, fraction=0.035, pad=0.04)
    cb.set_label("Mean Absolute Error (m)", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    # Standardise border (no red/green highlighting)
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.0)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, filename)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved spatial error map to {plot_path}")