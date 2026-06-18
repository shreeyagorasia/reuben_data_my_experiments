"""
models/dnn_baseline/plots.py
==============================
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
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epoch_log, train_hist, marker="o", color="indianred", lw=2, markersize=5, label="Training Loss (with L1)")
    ax.plot(epoch_log, val_hist, marker="s", color="steelblue", lw=2, markersize=5, linestyle="--", label="Validation Loss")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Loss (scaled MSE)", fontsize=11)
    ax.set_title("DNN Training vs. Validation Loss History", fontsize=12, fontweight="bold")
    ax.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=10)
    ax.grid(True, alpha=0.3, linestyle=":")
    plt.tight_layout()
    curve_path = os.path.join(output_dir, "training_curve.png")
    plt.savefig(curve_path, dpi=150)
    plt.close()
    print(f"Saved training curve to {curve_path}")

def plot_spatial_error(X_coords, Y_coords, y_test, y_pred, title, output_dir, filename="spatial_error_map.png"):
    return _plot_spatial_error(X_coords, Y_coords, y_test, y_pred, title, output_dir, filename)

def plot_spatial_signed_error(X_coords, Y_coords, y_test, y_pred, title, output_dir, filename="spatial_signed_error_map.png"):
    return _plot_spatial_signed_error(X_coords, Y_coords, y_test, y_pred, title, output_dir, filename)
