"""
Shared spatial plotting utilities.

All model-specific plotting modules delegate to this file so spatial maps use
the same C arrays, colour scales, colormaps, and metadata checks.
"""

import json
import os
from datetime import datetime, timezone

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp")
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from common.metrics import reuben_metrics


ABS_ERROR_VMIN = 0
ABS_ERROR_VMAX = 10
SIGNED_ERROR_VMIN = -10
SIGNED_ERROR_VMAX = 10
GRID_SIZE = 50
REDUCE_C_FUNCTION = np.mean
REDUCE_C_FUNCTION_NAME = "np.mean"
ABS_ERROR_CMAP = plt.cm.viridis
SIGNED_ERROR_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "pink_white_blue", ["#e87fa0", "#ffffff", "#3b6fb6"]
)


def _metadata_path(output_dir, filename):
    stem, _ = os.path.splitext(filename)
    return os.path.join(output_dir, f"{stem}.json")


def _base_metadata(plot_type, table, data_path_label, title, filename, c_values, y_test, y_pred):
    return {
        "plot_type": plot_type,
        "table": table,
        "data_path_label": data_path_label,
        "data_path": data_path_label,
        "title": title,
        "filename": filename,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_test": int(len(y_test)),
        "y_true_mean": float(np.mean(y_test)),
        "y_pred_mean": float(np.mean(y_pred)),
        "c_mean": float(np.mean(c_values)),
        "c_min": float(np.min(c_values)),
        "c_max": float(np.max(c_values)),
        "reduce_C_function": REDUCE_C_FUNCTION_NAME,
        "gridsize": GRID_SIZE,
    }


def _write_metadata(output_dir, filename, metadata):
    path = _metadata_path(output_dir, filename)
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved spatial plot metadata to {path}")


def plot_spatial_error(
    X_coords,
    Y_coords,
    y_test,
    y_pred,
    title,
    output_dir,
    filename="spatial_error_map.png",
    table="Table4.1",
    data_path_label="purged",
    extra_metadata=None,
):
    """Plot mean absolute error by spatial hexbin with shared settings."""
    y_test = np.asarray(y_test)
    y_pred = np.asarray(y_pred)
    abs_err = np.abs(y_test - y_pred)
    cmap = ABS_ERROR_CMAP

    print(
        f"[plot_spatial_error] {title}: "
        f"C=abs(y_test-y_pred), mean={abs_err.mean():.4f}, "
        f"min={abs_err.min():.4f}, max={abs_err.max():.4f}, "
        f"vmin={ABS_ERROR_VMIN}, vmax={ABS_ERROR_VMAX}, "
        f"cmap={cmap.name}, reduce_C_function={REDUCE_C_FUNCTION_NAME}, "
        f"table={table}, data={data_path_label}"
    )

    metrics = reuben_metrics(y_test, y_pred, label=f"{title} plot")

    fig, ax = plt.subplots(figsize=(6, 5))
    hb = ax.hexbin(
        X_coords,
        Y_coords,
        C=abs_err,
        reduce_C_function=REDUCE_C_FUNCTION,
        gridsize=GRID_SIZE,
        cmap=cmap,
        vmin=ABS_ERROR_VMIN,
        vmax=ABS_ERROR_VMAX,
        linewidths=0.2,
    )

    ax.set_title(
        f"{title}\nMAE = {metrics['mae']:.2f}m   Acc = {metrics['acc']:.1f}%",
        fontsize=10,
        fontweight="bold",
    )
    ax.set_xlabel("Easting (OS National Grid)", fontsize=8)
    ax.set_ylabel("Northing (OS National Grid)", fontsize=8)
    ax.tick_params(labelsize=7)

    cb = fig.colorbar(hb, ax=ax, fraction=0.035, pad=0.04)
    cb.set_label("Mean Absolute Error (m)", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.0)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, filename)
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved spatial error map to {plot_path}")

    metadata = _base_metadata(
        "absolute_error", table, data_path_label, title, filename, abs_err, y_test, y_pred
    )
    metadata.update({
        "vmin": ABS_ERROR_VMIN,
        "vmax": ABS_ERROR_VMAX,
        "cmap": cmap.name,
        "metric_values": metrics,
    })
    if extra_metadata:
        metadata.update(extra_metadata)
    _write_metadata(output_dir, filename, metadata)


def plot_spatial_signed_error(
    X_coords,
    Y_coords,
    y_test,
    y_pred,
    title,
    output_dir,
    filename="spatial_signed_error_map.png",
    table="Table4.1",
    data_path_label="purged",
    extra_metadata=None,
):
    """Plot signed error by spatial hexbin with shared settings."""
    y_test = np.asarray(y_test)
    y_pred = np.asarray(y_pred)
    signed_err = y_test - y_pred
    cmap = SIGNED_ERROR_CMAP

    print(
        f"[plot_spatial_signed_error] {title}: "
        f"C=y_test-y_pred, mean={signed_err.mean():.4f}, "
        f"min={signed_err.min():.4f}, max={signed_err.max():.4f}, "
        f"vmin={SIGNED_ERROR_VMIN}, vmax={SIGNED_ERROR_VMAX}, "
        f"cmap={cmap.name}, reduce_C_function={REDUCE_C_FUNCTION_NAME}, "
        f"table={table}, data={data_path_label}"
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    hb = ax.hexbin(
        X_coords,
        Y_coords,
        C=signed_err,
        reduce_C_function=REDUCE_C_FUNCTION,
        gridsize=GRID_SIZE,
        cmap=cmap,
        vmin=SIGNED_ERROR_VMIN,
        vmax=SIGNED_ERROR_VMAX,
        linewidths=0.2,
    )

    ax.set_title(
        f"{title}\nMean Signed Error = {signed_err.mean():+.2f}m",
        fontsize=10,
        fontweight="bold",
    )
    ax.set_xlabel("Easting (OS National Grid)", fontsize=8)
    ax.set_ylabel("Northing (OS National Grid)", fontsize=8)
    ax.tick_params(labelsize=7)

    cb = fig.colorbar(hb, ax=ax, fraction=0.035, pad=0.04)
    cb.set_label(
        "Signed Error, actual - predicted (m)\npink = underpredicted, blue = overpredicted",
        fontsize=8,
    )
    cb.ax.tick_params(labelsize=7)

    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.0)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, filename)
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved spatial signed error map to {plot_path}")

    metadata = _base_metadata(
        "signed_error", table, data_path_label, title, filename, signed_err, y_test, y_pred
    )
    metadata.update({
        "vmin": SIGNED_ERROR_VMIN,
        "vmax": SIGNED_ERROR_VMAX,
        "cmap": cmap.name,
        "metric_values": reuben_metrics(y_test, y_pred, label=f"{title} signed-plot"),
    })
    if extra_metadata:
        metadata.update(extra_metadata)
    _write_metadata(output_dir, filename, metadata)
