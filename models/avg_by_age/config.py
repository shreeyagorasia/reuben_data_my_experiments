"""
models/avg_by_age/config.py
=============================
Settings specific to the Avg by Age baseline.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from common.config import DATA_PATH_PURGED, DATA_PATH_UNSEEN, FEATURES, TARGET_COL, AGE_COL, RANDOM_SEED  # noqa: E402

MODEL_NAME = "avg_by_age"
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", MODEL_NAME)
