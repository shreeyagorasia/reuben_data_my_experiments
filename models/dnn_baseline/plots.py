"""
models/dnn_baseline/plots.py
==============================
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
    fig, ax = plt.subplots(figsize=(6, 5))
    abs_err = np.abs(np.array(y_test) - np.array(y_pred))
    cmap = plt.cm.viridis
    
    hb = ax.hexbin(X_coords, Y_coords, C=abs_err, reduce_C_function=np.mean,
                   gridsize=50, cmap=cmap, vmin=0, vmax=10, linewidths=0.2)
    
    mae = abs_err.mean()
    acc = (1 - np.mean(abs_err / (np.array(y_test) + 1e-8))) * 100
    ax.set_title(f"{title}\nMAE = {mae:.2f}m   Acc = {acc:.1f}%", fontsize=10, fontweight='bold')
    ax.set_xlabel("Easting", fontsize=8)
    ax.set_ylabel("Northing", fontsize=8)
    fig.colorbar(hb, ax=ax, fraction=0.035, pad=0.04).set_label("Mean Absolute Error (m)", fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()

def plot_spatial_signed_error(X_coords, Y_coords, y_test, y_pred, title, output_dir, filename="spatial_signed_error_map.png"):
    fig, ax = plt.subplots(figsize=(6, 5))
    signed_err = np.array(y_test) - np.array(y_pred)

    hb = ax.hexbin(X_coords, Y_coords, C=signed_err, reduce_C_function=np.mean,
                   gridsize=50, cmap=PINK_WHITE_BLUE, vmin=-10, vmax=10, linewidths=0.2)

    mean_signed = signed_err.mean()
    ax.set_title(f"{title}\nMean Signed Error = {mean_signed:+.2f}m", fontsize=10, fontweight='bold')
    ax.set_xlabel("Easting", fontsize=8)
    ax.set_ylabel("Northing", fontsize=8)
    fig.colorbar(hb, ax=ax, fraction=0.035, pad=0.04).set_label(
        "Signed Error, actual - predicted (m)\npink = underpredicted, blue = overpredicted", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()
