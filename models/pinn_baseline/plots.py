"""
models/pinn_baseline/plots.py
===============================
Contains plotting functions for the PINN baseline model.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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