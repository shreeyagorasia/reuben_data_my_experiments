"""
models/pinn_baseline/plots.py
===============================
Contains plotting functions for the PINN baseline model.
"""

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common.spatial_plots import (
    plot_spatial_error as _plot_spatial_error,
    plot_spatial_signed_error as _plot_spatial_signed_error,
)

def plot_training_curve(epoch_log, train_hist, val_hist, output_dir):
    """Plots the training and validation loss over epochs."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(epoch_log, train_hist, marker="o", color="indianred", lw=2,
            markersize=4, label="Train loss (data + physics)")
    ax.plot(epoch_log, val_hist, marker="s", color="steelblue", lw=2,
            markersize=4, linestyle="--", label="Val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (scaled MSE)")
    ax.set_title("PINN Training Convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    curve_path = os.path.join(output_dir, "training_curve.png")
    plt.savefig(curve_path, dpi=150)
    plt.close()  # Close the figure to free up memory
    print(f"Saved training curve to {curve_path}")

def plot_spatial_error(X_coords, Y_coords, y_test, y_pred, title, output_dir, filename="spatial_error_map.png"):
    """Plots a hexbin map of Mean Absolute Error across spatial coordinates."""
    return _plot_spatial_error(X_coords, Y_coords, y_test, y_pred, title, output_dir, filename)

def plot_spatial_signed_error(X_coords, Y_coords, y_test, y_pred, title, output_dir, filename="spatial_signed_error_map.png"):
    """Plots a hexbin map of signed error (actual - predicted) across spatial coordinates."""
    return _plot_spatial_signed_error(X_coords, Y_coords, y_test, y_pred, title, output_dir, filename)
