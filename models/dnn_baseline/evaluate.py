"""
models/dnn_baseline/evaluate.py
==================================
Optional. Loads outputs/dnn_baseline/checkpoint.pt and re-runs it on
the 2023 test set. Useful for checking results again later without
retraining, e.g. after copying the checkpoint to another machine.

Run with (from this folder):
    python evaluate.py
"""

import os
import config  # this model's config (also pulls in common settings)

import torch

from common.data_utils import load_data, build_feature_arrays
from common.metrics import reuben_metrics
from model import DNN
from plots import plot_spatial_error, plot_spatial_signed_error


def main():
    # Auto-detect GPU (CUDA for cluster, MPS for Mac, or CPU fallback)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # 1. Load data and rebuild the test features
    # ------------------------------------------------------------
    df12, df23, common_ids = load_data(config.DATA_PATH_UNSEEN)
    X_train, X_test, y_train, y_test, X_coords, Y_coords = build_feature_arrays(df12, df23)

    # 2. Load the checkpoint
    # ------------------------------------------------------------
    ckpt_path = os.path.join(config.OUTPUT_DIR, "checkpoint.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Could not find {ckpt_path}. Run train.py first.")

    # weights_only=False is needed because the checkpoint also contains
    # sklearn StandardScaler objects (not just model weights). Only load
    # checkpoints you created yourself / trust.
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    scaler_X = checkpoint["scaler_X"]
    scaler_y = checkpoint["scaler_y"]
    saved_metrics = checkpoint.get("final_metrics")

    print(f"Loaded checkpoint from {ckpt_path}")

    # Scalers are positional -- if config.FEATURES has been reordered since
    # this checkpoint was trained, the scaler would silently apply the wrong
    # mean/std to the wrong columns. Catch that here instead of failing silently.
    saved_feature_list = checkpoint.get("feature_list")
    if saved_feature_list is not None and list(saved_feature_list) != list(config.FEATURES):
        raise ValueError(
            "Feature columns have changed since this checkpoint was trained.\n"
            f"  Checkpoint was trained on : {saved_feature_list}\n"
            f"  config.FEATURES is now    : {config.FEATURES}\n"
            "Predictions would silently use the wrong scaler columns -- retrain before evaluating."
        )

    # 3. Rebuild the model and load its weights
    # ------------------------------------------------------------
    dnn = DNN(n_features=X_test.shape[1]).to(device)
    dnn.load_state_dict(checkpoint["model_state"])
    dnn.eval()

    # 4. Scale the test inputs using the SAVED scaler
    # ------------------------------------------------------------
    # Why scale here? We must apply the exact same transformation (mean, std)
    # learned from the 2012 data to the 2023 test data for the model's predictions to be valid.
    Xte = torch.tensor(scaler_X.transform(X_test), dtype=torch.float32).to(device)

    # 5. Predict and evaluate
    # ------------------------------------------------------------
    with torch.no_grad():
        y_pred_scaled = dnn(Xte).cpu().numpy().reshape(-1, 1)
        y_pred = scaler_y.inverse_transform(y_pred_scaled).ravel()

    fresh_metrics = reuben_metrics(y_test, y_pred, "DNN (reloaded)")

    # Compare metrics at training time (saved in the checkpoint) against
    # what we just recomputed, to catch any drift between runs.
    if saved_metrics is not None:
        print("\nMetrics comparison (saved at training time vs. freshly recomputed):")
        print(f"  {'Metric':<6} {'Saved':>10} {'Fresh':>10}")
        for k in fresh_metrics:
            print(f"  {k:<6} {saved_metrics.get(k, float('nan')):>10.4f} {fresh_metrics[k]:>10.4f}")
    else:
        print("\n(No 'final_metrics' found in checkpoint -- skipping saved-vs-fresh comparison.)")

    # 6. Plot spatial error maps
    # ------------------------------------------------------------
    plot_spatial_error(X_coords, Y_coords, y_test, y_pred,
                       title="(f) DNN Temporal Error (Unseen)",
                       output_dir=config.OUTPUT_DIR)
    plot_spatial_signed_error(X_coords, Y_coords, y_test, y_pred,
                       title="(f) DNN Temporal Error (Unseen)",
                       output_dir=config.OUTPUT_DIR)

if __name__ == "__main__":
    main()
