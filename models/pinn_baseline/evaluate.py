"""
models/pinn_baseline/evaluate.py
==================================
Optional. Loads outputs/pinn_baseline/checkpoint.pt and re-runs it on
the 2023 test set. Useful for checking results again later without
retraining, e.g. after copying the checkpoint to another machine.

Run with (from this folder):
    python evaluate.py
"""

import os
import config  # this model's config (also pulls in common settings)

import torch

from common.data_utils import load_data, build_feature_arrays, split_age_column
from common.metrics import reuben_metrics
from model import PINN
from plots import plot_spatial_error


def main():
    # 1. Load data and rebuild the test features
    # ------------------------------------------------------------
    df12, df23, common_ids = load_data(config.DATA_PATH_UNSEEN)
    X_train, X_test, y_train, y_test, X_coords, Y_coords = build_feature_arrays(df12, df23)
    _, _, X_test_other, X_test_age, _, _ = split_age_column(X_train, X_test)

    # 2. Load the checkpoint
    # ------------------------------------------------------------
    ckpt_path = os.path.join(config.OUTPUT_DIR, "checkpoint.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Could not find {ckpt_path}. Run train.py first.")

    # weights_only=False is needed because the checkpoint also contains
    # sklearn StandardScaler objects (not just model weights). Only load
    # checkpoints you created yourself / trust.
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    scaler_Xo = checkpoint["scaler_Xo"]
    scaler_age = checkpoint["scaler_age"]
    scaler_y = checkpoint["scaler_y"]
    other_idxs = checkpoint["other_idxs"]

    print(f"Loaded checkpoint from {ckpt_path}")

    # 3. Rebuild the model and load its weights
    # ------------------------------------------------------------
    pinn = PINN(n_other=len(other_idxs))
    pinn.load_state_dict(checkpoint["model_state"])
    pinn.eval()

    # 4. Scale the test inputs using the SAVED scalers
    # ------------------------------------------------------------
    # Why scale here? We must apply the exact same transformation (mean, std)
    # learned from the 2012 data to the 2023 test data for the model's predictions to be valid.
    #
    # Variable names:
    # Xte_o : Test features (other than AGE) scaled as tensors
    # Xte_a : Test AGE feature scaled as a tensor
    Xte_o = torch.tensor(scaler_Xo.transform(X_test_other), dtype=torch.float32)
    Xte_a = torch.tensor(scaler_age.transform(X_test_age), dtype=torch.float32)

    # 5. Predict and evaluate
    # ------------------------------------------------------------
    with torch.no_grad():
        y_pred_scaled = pinn(Xte_o, Xte_a).numpy().reshape(-1, 1)
        y_pred = scaler_y.inverse_transform(y_pred_scaled).ravel()

    reuben_metrics(y_test, y_pred, "PINN (reloaded)")

    # 6. Plot spatial error map
    # ------------------------------------------------------------
    plot_spatial_error(X_coords, Y_coords, y_test, y_pred, 
                       title="(g) PINN Baseline", 
                       output_dir=config.OUTPUT_DIR)

if __name__ == "__main__":
    main()
