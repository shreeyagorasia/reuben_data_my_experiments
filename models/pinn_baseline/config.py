"""
models/pinn_baseline/config.py
================================
Settings specific to the PINN baseline.

Shared settings (data path, feature list, etc.) come from
common/config.py and are re-imported here so train.py only needs
`import config` to get everything it needs.
"""

import os
import sys
from pathlib import Path

# Add the project root to sys.path so "common" can be imported
# regardless of which directory this script is run from.
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from common.config import DATA_PATH_PURGED, DATA_PATH_UNSEEN, FEATURES, TARGET_COL, AGE_COL, RANDOM_SEED  # noqa: E402

# ----------------------------------------------------------------
# Model ID and output location
# ----------------------------------------------------------------
MODEL_NAME = "pinn_baseline"
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", MODEL_NAME)

# Where to find the Chapman-Richards baseline's fitted parameters.
# These are used as the "physics prior" for this PINN.
# -> run models/chapman_richards/train.py BEFORE this model.
CR_PARAMS_PATH = os.path.join(PROJECT_ROOT, "outputs", "chapman_richards", "cr_params.json")

# ----------------------------------------------------------------
# PINN architecture & training hyperparameters
# ----------------------------------------------------------------
HIDDEN_SIZE = 128             # width of each hidden layer (Reuben: 128)

LAMBDA_PH = 1.0                # weight on the physics loss term
LAMBDA_L1 = 1e-5                # weight on the L1 weight-penalty term

EPOCHS = 5 #FIND NO EPOCHS RUEBEN USED and CHANGE to that                  # max number of training epochs
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EARLY_STOP_PATIENCE = 50       # stop if val loss hasn't improved for this many epochs

VAL_SPLIT = 0.33               # fraction of 2012 data used for validation
