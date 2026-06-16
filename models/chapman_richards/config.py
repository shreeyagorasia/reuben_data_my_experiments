"""
models/chapman_richards/config.py
===================================
Settings specific to the Chapman-Richards baseline.

Shared settings (data path, feature list, etc.) come from
common/config.py and are re-imported here so train.py only needs
`import config` to get everything it needs.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path to allow global imports (like 'common.config')
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT) 

from common.config import DATA_PATH_PURGED, DATA_PATH_UNSEEN, FEATURES, TARGET_COL, AGE_COL, RANDOM_SEED  # noqa: E402

# ----------------------------------------------------------------
# Model ID and output location
# ----------------------------------------------------------------
MODEL_NAME = "chapman_richards"
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", MODEL_NAME)

# ----------------------------------------------------------------
# Chapman-Richards curve_fit settings
# ----------------------------------------------------------------
# H(t) = y_max * (1 - exp(-k * t)) ^ p
#
# CR_P0 is the starting guess for curve_fit, CR_BOUNDS keeps the
# fitted parameters within sensible ranges.
CR_P0 = [46.1, 0.0187, 1.017]
CR_BOUNDS = ([30, 0.001, 0.5], [100, 0.1, 3.0])
