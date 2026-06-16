"""
models/dnn_baseline/config.py
==============================
Settings specific to the DNN model.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from common.config import DATA_PATH_PURGED, DATA_PATH_UNSEEN, FEATURES, TARGET_COL, AGE_COL, RANDOM_SEED  # noqa: E402

# ----------------------------------------------------------------
# Model ID and output location
# ----------------------------------------------------------------
MODEL_NAME = "dnn_baseline"
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", MODEL_NAME)

HIDDEN_SIZE = 128
LAMBDA_L1 = 1e-5
EPOCHS = 1000
BATCH_SIZE = 512
LEARNING_RATE = 1e-4
EARLY_STOP_PATIENCE = 50
VAL_SPLIT = 0.33
