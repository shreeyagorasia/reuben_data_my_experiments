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


REL_ERROR_VMIN = 0
REL_ERROR_VMAX = 0.5
REL_ERROR_CAP = 0.5
SIGNED_REL_ERROR_VMIN = -1.0
SIGNED_REL_ERROR_VMAX = 1.0
GRID_SIZE = 50
REDUCE_C_FUNCTION = np.mean
REDUCE_C_FUNCTION_NAME = "np.mean"
REL_ERROR_CMAP = plt.cm.viridis
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
    """Plot capped mean relative error by spatial hexbin with shared settings."""
    y_test = np.asarray(y_test)
    y_pred = np.asarray(y_pred)
    abs_err = np.abs(y_test - y_pred)
    rel_err = abs_err / (np.abs(y_test) + 1e-8)
    capped_rel_err = np.minimum(rel_err, REL_ERROR_CAP)
    cmap = REL_ERROR_CMAP

    print(
        f"[plot_spatial_error] {title}: "
        f"C=min(abs(y_test-y_pred)/(abs(y_test)+1e-8), {REL_ERROR_CAP}), "
        f"mean={capped_rel_err.mean():.4f}, "
        f"min={capped_rel_err.min():.4f}, max={capped_rel_err.max():.4f}, "
        f"uncapped_mean={rel_err.mean():.4f}, uncapped_max={rel_err.max():.4f}, "
        f"vmin={REL_ERROR_VMIN}, vmax={REL_ERROR_VMAX}, "
        f"cmap={cmap.name}, reduce_C_function={REDUCE_C_FUNCTION_NAME}, "
        f"table={table}, data={data_path_label}"
    )

    metrics = reuben_metrics(y_test, y_pred, label=f"{title} plot")

    fig, ax = plt.subplots(figsize=(6, 5))
    hb = ax.hexbin(
        X_coords,
        Y_coords,
        C=capped_rel_err,
        reduce_C_function=REDUCE_C_FUNCTION,
        gridsize=GRID_SIZE,
        cmap=cmap,
        vmin=REL_ERROR_VMIN,
        vmax=REL_ERROR_VMAX,
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
    cb.set_label("Mean Relative Error (capped at 0.5)", fontsize=8)
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
        "relative_error_capped", table, data_path_label, title, filename, capped_rel_err, y_test, y_pred
    )
    metadata.update({
        "vmin": REL_ERROR_VMIN,
        "vmax": REL_ERROR_VMAX,
        "relative_error_cap": REL_ERROR_CAP,
        "uncapped_relative_error_mean": float(np.mean(rel_err)),
        "uncapped_relative_error_min": float(np.min(rel_err)),
        "uncapped_relative_error_max": float(np.max(rel_err)),
        "c_description": "min(abs(y_true - y_pred) / (abs(y_true) + 1e-8), 0.5)",
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
    """Plot signed relative error by spatial hexbin with shared settings."""
    y_test = np.asarray(y_test)
    y_pred = np.asarray(y_pred)
    signed_err = y_test - y_pred
    signed_rel_err = signed_err / (np.abs(y_test) + 1e-8)
    cmap = SIGNED_ERROR_CMAP

    print(
        f"[plot_spatial_signed_error] {title}: "
        f"C=(y_test-y_pred)/(abs(y_test)+1e-8), mean={signed_rel_err.mean():.4f}, "
        f"min={signed_rel_err.min():.4f}, max={signed_rel_err.max():.4f}, "
        f"mean_signed_m={signed_err.mean():.4f}, "
        f"vmin={SIGNED_REL_ERROR_VMIN}, vmax={SIGNED_REL_ERROR_VMAX}, "
        f"cmap={cmap.name}, reduce_C_function={REDUCE_C_FUNCTION_NAME}, "
        f"table={table}, data={data_path_label}"
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    hb = ax.hexbin(
        X_coords,
        Y_coords,
        C=signed_rel_err,
        reduce_C_function=REDUCE_C_FUNCTION,
        gridsize=GRID_SIZE,
        cmap=cmap,
        vmin=SIGNED_REL_ERROR_VMIN,
        vmax=SIGNED_REL_ERROR_VMAX,
        linewidths=0.2,
    )

    ax.set_title(
        f"{title}\nMean Signed Rel. Error = {signed_rel_err.mean():+.3f}",
        fontsize=10,
        fontweight="bold",
    )
    ax.set_xlabel("Easting (OS National Grid)", fontsize=8)
    ax.set_ylabel("Northing (OS National Grid)", fontsize=8)
    ax.tick_params(labelsize=7)

    cb = fig.colorbar(hb, ax=ax, fraction=0.035, pad=0.04)
    cb.set_label(
        "Signed Relative Error, (actual - predicted) / actual\npink = underpredicted, blue = overpredicted",
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
        "signed_relative_error", table, data_path_label, title, filename, signed_rel_err, y_test, y_pred
    )
    metadata.update({
        "vmin": SIGNED_REL_ERROR_VMIN,
        "vmax": SIGNED_REL_ERROR_VMAX,
        "c_description": "(y_true - y_pred) / (abs(y_true) + 1e-8)",
        "mean_signed_error_m": float(np.mean(signed_err)),
        "min_signed_error_m": float(np.min(signed_err)),
        "max_signed_error_m": float(np.max(signed_err)),
        "cmap": cmap.name,
        "metric_values": reuben_metrics(y_test, y_pred, label=f"{title} signed-plot"),
    })
    if extra_metadata:
        metadata.update(extra_metadata)
    _write_metadata(output_dir, filename, metadata)
