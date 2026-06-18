"""
models/chapman_richards/plots.py
==================================
Contains plotting functions for the Chapman-Richards model.
"""

import os
import numpy as np
os.environ.setdefault("MPLCONFIGDIR", "/tmp")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common.spatial_plots import (
    plot_spatial_error as _plot_spatial_error,
    plot_spatial_signed_error as _plot_spatial_signed_error,
)
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
    return _plot_spatial_error(X_coords, Y_coords, y_test, y_pred, title, output_dir, filename)

def plot_spatial_signed_error(X_coords, Y_coords, y_test, y_pred, title, output_dir, filename="spatial_signed_error_map.png"):
    """Plots a hexbin map of signed error (actual - predicted) across spatial coordinates."""
    return _plot_spatial_signed_error(X_coords, Y_coords, y_test, y_pred, title, output_dir, filename)
