"""
Settings for the observed 2023 test-data control.

This is not a trained model. It uses the Table 4.1 2023 observed heights as
both y_true and y_pred so the output maps represent the zero-error reference.
"""

from pathlib import Path


MODEL_NAME = "test_data"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / MODEL_NAME

