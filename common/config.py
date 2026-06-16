"""
common/config.py
=================
Settings shared by ALL models: where the data lives, which columns
are used as features/target, and the random seed.

Model-specific settings (hyperparameters, output folder, etc.) live
in each model's own models/<model_name>/config.py instead -- see
e.g. models/pinn_baseline/config.py.
"""

import os

# PROJECT_ROOT = the top-level "Reuben_data_my_experiments" folder
# (the folder that contains common/, data/, models/, outputs/, ...).
# Computed automatically so paths work no matter where you run scripts from.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Path to the input CSV. Put your data file in data/ and update the
# filename below if it differs.
DATA_PATH_PURGED = os.path.join(
    PROJECT_ROOT, "data", "Aber_1223_conjoin_plot_purged_expanded_encoded.csv"
)
DATA_PATH_UNSEEN = os.path.join(
    PROJECT_ROOT, "data", "Aber_1223_conjoin_expanded_encoded_duplicate_plots_removed.csv"
)

# ----------------------------------------------------------------
# DATA COLUMNS (Reuben's Table D.2)
# ----------------------------------------------------------------
# X, Y, AGE, CULTIVATION, and PRIMARY LAND USE (one-hot encoded).
# PLOT_ID is excluded -- it was only used for matching plots between
# years and carries no extra information beyond X and Y.
FEATURES = [
    "X", "Y", "AGE",
    "CULTIVATN_MOUNDING", "CULTIVATN_NO CULTIVATION",
    "PRILANDUSE_Dead High Forest", "PRILANDUSE_Failed",
    "PRILANDUSE_Felled", "PRILANDUSE_High Forest",
    "PRILANDUSE_Open", "PRILANDUSE_Open Water",
    "PRILANDUSE_Other Built Facility",
    "PRILANDUSE_Partially Intruded Broadleaf",
    "PRILANDUSE_Quarries", "PRILANDUSE_Residential",
    "PRILANDUSE_Unplantable or bare",
    "PRILANDUSE_Unplanted streamsides",
    "PRILANDUSE_Windblow",
]

TARGET_COL = "TOP_HEIGHT"   # what we are predicting (tree top height, in metres)
AGE_COL = "AGE"             # used both as a feature and for physics-based models

# ----------------------------------------------------------------
# REPRODUCIBILITY
# ----------------------------------------------------------------
RANDOM_SEED = 42
