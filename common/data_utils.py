"""
common/data_utils.py
=====================
Helper functions for loading the plot data and turning it into the
arrays / tensors models need. Shared by every model.

The overall idea:
  - 2012 data is used for TRAINING (it's the "before" state)
  - 2023 data is used for TESTING  (it's the "after" / target state)
  - We only keep plots that exist in BOTH years, so we can compare
    "what a plot looked like in 2012" to "what it grew into by 2023".
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold

from common import config


def load_data(path=None):
    """Load the CSV and split it into 2012 / 2023 dataframes.

    Returns
    -------
    df12, df23 : DataFrames indexed by PLOT_ID, for 2012 and 2023
    common_ids : PLOT_IDs that appear in both years
    """
    path = path or config.DATA_PATH
    df = pd.read_csv(path)

    df12 = df[df["YEAR"] == 2012].set_index("PLOT_ID")
    df23 = df[df["YEAR"] == 2023].set_index("PLOT_ID")

    print(f"\n--- Data Loading & Purging Steps for {os.path.basename(path)} ---")
    print(f"  Raw 2012 plots : {len(df12):,}")
    print(f"  Raw 2023 plots : {len(df23):,}")

    # 1. Identify common plots and purge "shrunk" plots (negative growth)
    # Reuben treats these as sensor noise/errors and removes them.
    common_ids_raw = df12.index.intersection(df23.index)
    heights_12 = df12.loc[common_ids_raw, config.TARGET_COL]
    heights_23 = df23.loc[common_ids_raw, config.TARGET_COL]
    
    shrunk_ids = common_ids_raw[heights_23 < heights_12]
    print(f"  Noise purged   : {len(shrunk_ids):,} common plots removed (negative growth)")

    # 2. Drop the noisy plots from BOTH dataframes
    df12 = df12.drop(index=shrunk_ids)
    df23 = df23.drop(index=shrunk_ids)

    # Sort the index to ensure completely deterministic ordering for train_test_split
    common_ids = df12.index.intersection(df23.index).sort_values()

    print(f"  Final 2012     : {len(df12):,} plots")
    print(f"  Final 2023     : {len(df23):,} plots")
    print(f"  Final common   : {len(common_ids):,} plots")

    return df12, df23, common_ids


def build_feature_arrays(df12, df23):
    """Build the X (features) and y (target) arrays used by all models.

    X_train / y_train come from 2012 (inputs + the height we start from).
    X_test  / y_test  come from 2023 (the height we are trying to predict).

    Both X_train and X_test use the SAME set of features (config.FEATURES),
    just measured/recorded in different years.
    """
    features = config.FEATURES

    # Boolean one-hot columns need to become 0/1 integers for
    # sklearn / torch to work with them.
    bool_feats = [c for c in features if df12[c].dtype == bool]

    X_train_df = df12[features].copy()
    X_test_df = df23[features].copy()

    for c in bool_feats:
        X_train_df[c] = X_train_df[c].astype(int)
        X_test_df[c] = X_test_df[c].astype(int)

    X_train = X_train_df.values
    X_test = X_test_df.values

    y_train = df12[config.TARGET_COL].values
    y_test = df23[config.TARGET_COL].values

    # Saved separately in case you want to make spatial error plots later.
    X_coords = df23["X"].values
    Y_coords = df23["Y"].values

    print(f"Features   : {X_train.shape[1]}")
    print(f"y_train    : mean={y_train.mean():.2f}m  ({len(y_train):,} plots)")
    print(f"y_test     : mean={y_test.mean():.2f}m  ({len(y_test):,} plots)")

    return X_train, X_test, y_train, y_test, X_coords, Y_coords


def split_age_column(X_train, X_test):
    """Pull the AGE column out of the feature matrix.

    Physics-informed models need AGE on its own so they can compute
    d(predicted height) / d(AGE) with autograd.

    Returns
    -------
    X_train_other, X_train_age, X_test_other, X_test_age : arrays
    age_idx     : the column index AGE had in config.FEATURES
    other_idxs  : the column indices of all the OTHER features
    """
    age_idx = config.FEATURES.index(config.AGE_COL)
    other_idxs = [i for i in range(len(config.FEATURES)) if i != age_idx]

    X_train_other = X_train[:, other_idxs]
    X_train_age = X_train[:, age_idx].reshape(-1, 1)
    X_test_other = X_test[:, other_idxs]
    X_test_age = X_test[:, age_idx].reshape(-1, 1)

    print(f"Main tensor  : {X_train_other.shape[1]} features (all except AGE)")
    print(f"Age tensor   : 1 feature (AGE, used separately for autograd)")

    return X_train_other, X_train_age, X_test_other, X_test_age, age_idx, other_idxs


def make_scalers(X_train_other, X_train_age, y_train):
    """Fit StandardScalers using ONLY the 2012 ("train") data.

    Using only 2012 stats to scale everything (including the 2023 test
    set) avoids "leaking" information about the 2023 outcomes into the
    scaling step.
    """
    scaler_Xo = StandardScaler().fit(X_train_other)
    scaler_age = StandardScaler().fit(X_train_age)
    scaler_y = StandardScaler().fit(y_train.reshape(-1, 1))

    sigma_y = float(scaler_y.scale_[0])
    sigma_age = float(scaler_age.scale_[0])
    print(f"Scaling:  sigma_y={sigma_y:.4f}  sigma_age={sigma_age:.4f}  "
          f"chain-rule factor={sigma_y / sigma_age:.4f}")

    return scaler_Xo, scaler_age, scaler_y


def train_val_split(n_samples, val_fraction, seed=None):
    """Return (train_indices, val_indices) for the 2012 data.

    val_fraction and seed are passed in explicitly (rather than read
    from a shared config) because different models may want different
    validation splits.
    """
    seed = config.RANDOM_SEED if seed is None else seed
    idx = np.arange(n_samples)
    tr_idx, val_idx = train_test_split(idx, test_size=val_fraction, random_state=seed)
    return tr_idx, val_idx

def get_kfold_splits(n_samples, n_splits=3, seed=None):
    """Yields (train_idx, test_idx) for K-Fold cross validation."""
    seed = config.RANDOM_SEED if seed is None else seed
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return kf.split(np.zeros(n_samples))


def to_tensors(idx_arr, X_other, X_age, y, scaler_Xo, scaler_age, scaler_y):
    """Select rows `idx_arr`, scale them, and convert to torch tensors.

    Used to build the train and validation tensors from the 2012 data.
    """
    import torch
    Xo = torch.tensor(scaler_Xo.transform(X_other[idx_arr]), dtype=torch.float32)
    Xa = torch.tensor(scaler_age.transform(X_age[idx_arr]), dtype=torch.float32)
    y_scaled = scaler_y.transform(y[idx_arr].reshape(-1, 1)).ravel()
    yt = torch.tensor(y_scaled, dtype=torch.float32)
    return Xo, Xa, yt
