"""Plot wrappers for the AvgByAge baseline."""

from common.spatial_plots import (
    plot_spatial_error as _plot_spatial_error,
    plot_spatial_signed_error as _plot_spatial_signed_error,
)

def plot_spatial_error(X_coords, Y_coords, y_test, y_pred, title, output_dir, filename="spatial_error_map.png"):
    return _plot_spatial_error(X_coords, Y_coords, y_test, y_pred, title, output_dir, filename)

def plot_spatial_signed_error(X_coords, Y_coords, y_test, y_pred, title, output_dir, filename="spatial_signed_error_map.png"):
    return _plot_spatial_signed_error(X_coords, Y_coords, y_test, y_pred, title, output_dir, filename)
