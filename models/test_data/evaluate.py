"""
Create spatial plots for the observed 2023 test-data control.

The control prediction is y_pred = y_true for the Table 4.1 temporal
experiment. This gives a zero-error reference layer for comparison figures
such as PINN - TestData and AvgByAge - TestData.
"""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import config as common_config  # noqa: E402
from common.data_utils import build_feature_arrays, load_data  # noqa: E402
from common.spatial_plots import plot_spatial_error, plot_spatial_signed_error  # noqa: E402
from common.metrics import reuben_metrics  # noqa: E402

import config  # noqa: E402


def main():
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df12, df23, _ = load_data(common_config.DATA_PATH_PURGED)
    _, _, _, y_true, x_coords, y_coords = build_feature_arrays(df12, df23)
    y_pred = y_true.copy()

    metrics = reuben_metrics(y_true, y_pred, "TestData control")

    with open(config.OUTPUT_DIR / "config_used.json", "w") as f:
        json.dump(
            {
                "model_name": config.MODEL_NAME,
                "method": "Observed 2023 test-data control; y_pred = y_true",
                "table": "Table4.1",
                "data_path_label": "purged",
                "n_test": int(len(y_true)),
                "metrics": metrics,
                "outputs": {
                    "spatial_error_map": "spatial_error_map.png",
                    "spatial_error_metadata": "spatial_error_map.json",
                    "spatial_signed_error_map": "spatial_signed_error_map.png",
                    "spatial_signed_error_metadata": "spatial_signed_error_map.json",
                },
            },
            f,
            indent=2,
        )

    plot_spatial_error(
        x_coords,
        y_coords,
        y_true,
        y_pred,
        "Test Data Control",
        config.OUTPUT_DIR,
        extra_metadata={"control_type": "observed_test_data", "short_name": "TestData"},
    )
    plot_spatial_signed_error(
        x_coords,
        y_coords,
        y_true,
        y_pred,
        "Test Data Control",
        config.OUTPUT_DIR,
        extra_metadata={"control_type": "observed_test_data", "short_name": "TestData"},
    )


if __name__ == "__main__":
    main()

