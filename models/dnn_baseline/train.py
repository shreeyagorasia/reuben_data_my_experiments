"""
models/dnn_baseline/train.py
==============================
Table 4.1: Trains a standard Deep Neural Network (DNN) to predict 2023 tree
heights from 2012 plot features (including AGE, species, density, etc.).

Unlike the Chapman-Richards model, this network learns from ALL features,
not just age — but it has no built-in knowledge of tree biology.

Run with:
    python train.py
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

import config
from common.data_utils import load_data, build_feature_arrays, train_val_split
from common.metrics import reuben_metrics
from model import DNN
from plots import plot_spatial_error, plot_spatial_signed_error, plot_training_curve


def train_model(X_train, y_train, X_test, device):
    """
    Scales data, builds the DNN, trains it, and returns predictions on X_test.

    Parameters
    ----------
    X_train : numpy array, shape (n_train, n_features)  — 2012 feature matrix
    y_train : numpy array, shape (n_train,)              — 2012 heights (labels)
    X_test  : numpy array, shape (n_test,  n_features)  — 2023 feature matrix
    device  : torch.device (cpu / mps / cuda)

    Returns
    -------
    y_pred      : numpy array of predicted 2023 heights (in metres)
    epoch_log   : list of epoch numbers that were logged
    train_hist  : list of training losses (one per logged epoch)
    val_hist    : list of validation losses (one per logged epoch)
    """

    # ------------------------------------------------------------------ #
    # STEP 1 — SCALE                                                       #
    # ------------------------------------------------------------------ #
    # Fit scalers on 2012 TRAINING data only (never the 2023 test data).
    # This stops the model from "cheating" by learning the 2023 statistics.
    scaler_X = StandardScaler().fit(X_train)
    scaler_y = StandardScaler().fit(y_train.reshape(-1, 1))

    # ------------------------------------------------------------------ #
    # STEP 2 — VALIDATION SPLIT                                           #
    # ------------------------------------------------------------------ #
    # Hold out a slice of the 2012 data as a validation set.
    # We watch validation loss to decide when to stop training early.
    tr_idx, val_idx = train_val_split(len(X_train), val_fraction=config.VAL_SPLIT)

    # Scale and convert everything to PyTorch tensors
    Xtr  = torch.tensor(scaler_X.transform(X_train[tr_idx]),  dtype=torch.float32)
    ytr  = torch.tensor(scaler_y.transform(y_train[tr_idx].reshape(-1, 1)).ravel(), dtype=torch.float32)
    Xval = torch.tensor(scaler_X.transform(X_train[val_idx]), dtype=torch.float32).to(device)
    yval = torch.tensor(scaler_y.transform(y_train[val_idx].reshape(-1, 1)).ravel(), dtype=torch.float32).to(device)
    Xte  = torch.tensor(scaler_X.transform(X_test),           dtype=torch.float32).to(device)

    # DataLoader shuffles the training data into mini-batches each epoch
    train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=config.BATCH_SIZE, shuffle=True)

    # ------------------------------------------------------------------ #
    # STEP 3 — BUILD MODEL                                                 #
    # ------------------------------------------------------------------ #
    dnn      = DNN(n_features=X_train.shape[1]).to(device)
    optimiser = torch.optim.Adam(dnn.parameters(), lr=config.LEARNING_RATE)
    # ReduceLROnPlateau: automatically lowers learning rate when val loss stalls
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode="min", factor=0.8, patience=10)
    loss_fn   = nn.MSELoss()

    # ------------------------------------------------------------------ #
    # STEP 4 — TRAINING LOOP                                               #
    # ------------------------------------------------------------------ #
    train_hist, val_hist, epoch_log = [], [], []
    best_val    = float("inf")
    patience_ctr = 0

    print(f"\n  {'Epoch':>4} | {'Train loss':>12} | {'Val loss':>12}")
    print("  " + "-" * 36)

    for epoch in range(config.EPOCHS):
        dnn.train()
        running, n_batches = 0.0, 0

        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimiser.zero_grad()
            pred = dnn(Xb)
            # L1 regularisation: lightly penalises large weights to avoid overfitting
            l1   = sum(p.abs().sum() for p in dnn.parameters())
            loss = loss_fn(pred, yb) + config.LAMBDA_L1 * l1
            loss.backward()
            optimiser.step()
            running  += loss.item()
            n_batches += 1

        avg_train = running / n_batches

        dnn.eval()
        with torch.no_grad():
            val_loss = loss_fn(dnn(Xval), yval).item()

        scheduler.step(val_loss)

        # Early stopping: quit if validation loss hasn't improved in a while
        if val_loss < best_val - 1e-5:
            best_val     = val_loss
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= config.EARLY_STOP_PATIENCE:
                epoch_log.append(epoch + 1); train_hist.append(avg_train); val_hist.append(val_loss)
                print(f"  Early stopping at epoch {epoch + 1}")
                break

        if (epoch + 1) % 10 == 0:
            epoch_log.append(epoch + 1); train_hist.append(avg_train); val_hist.append(val_loss)
            print(f"  {epoch + 1:>4}  |  {avg_train:>12.6f}  |  {val_loss:>12.6f}", flush=True)

    if not epoch_log:
        epoch_log.append(epoch + 1); train_hist.append(avg_train); val_hist.append(val_loss)

    # ------------------------------------------------------------------ #
    # STEP 5 — PREDICT ON TEST SET                                         #
    # ------------------------------------------------------------------ #
    dnn.eval()
    with torch.no_grad():
        # Undo the y-scaling so we get predictions back in metres
        y_pred = scaler_y.inverse_transform(dnn(Xte).cpu().numpy().reshape(-1, 1)).ravel()

    return y_pred, epoch_log, train_hist, val_hist, dnn, optimiser, scaler_X, scaler_y


def main():
    np.random.seed(config.RANDOM_SEED)
    torch.manual_seed(config.RANDOM_SEED)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"\nUsing device: {device}")

    # ------------------------------------------------------------------ #
    # 1. LOAD DATA                                                         #
    # ------------------------------------------------------------------ #
    df12, df23, _ = load_data(config.DATA_PATH_PURGED)
    X_train, X_test, y_train, y_test, X_coords, Y_coords = build_feature_arrays(df12, df23)

    # ------------------------------------------------------------------ #
    # 2. TRAIN & PREDICT                                                   #
    # ------------------------------------------------------------------ #
    print("\n--- Training DNN (Table 4.1: Temporal, Common Plots) ---")
    y_pred, ep_log, tr_hist, val_hist, dnn, optimiser, scaler_X, scaler_y = train_model(
        X_train, y_train, X_test, device
    )

    # ------------------------------------------------------------------ #
    # 3. EVALUATE                                                          #
    # ------------------------------------------------------------------ #
    metrics = reuben_metrics(y_test, y_pred, label="DNN Baseline (Table 4.1)")

    # ------------------------------------------------------------------ #
    # 4. SAVE OUTPUTS                                                      #
    # ------------------------------------------------------------------ #
    # Checkpoint: saved weights + scalers so you can reload the model later
    torch.save(
        {
            "model_state":     dnn.state_dict(),
            "optimizer_state": optimiser.state_dict(),
            "scaler_X":        scaler_X,
            "scaler_y":        scaler_y,
            "feature_list":    config.FEATURES,
            "final_metrics":   metrics,
        },
        os.path.join(config.OUTPUT_DIR, "checkpoint.pt"),
    )

    pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"]).to_csv(
        os.path.join(config.OUTPUT_DIR, "results.csv"), index=False
    )

    with open(os.path.join(config.OUTPUT_DIR, "config_used.json"), "w") as f:
        json.dump(
            {
                "epochs_max": config.EPOCHS,
                "epochs_run": ep_log[-1] if ep_log else 0,
                "batch_size": config.BATCH_SIZE,
                "learning_rate": config.LEARNING_RATE,
                "lambda_l1": config.LAMBDA_L1,
                "early_stop_patience": config.EARLY_STOP_PATIENCE,
                "val_split": config.VAL_SPLIT,
                "hidden_size": config.HIDDEN_SIZE,
                "features": config.FEATURES,
                "final_metrics": metrics,
            },
            f, indent=2,
        )

    plot_training_curve(ep_log, tr_hist, val_hist, config.OUTPUT_DIR)
    plot_spatial_error(X_coords, Y_coords, y_test, y_pred,
                       "(f) DNN Temporal Error (Common)", config.OUTPUT_DIR)
    plot_spatial_signed_error(X_coords, Y_coords, y_test, y_pred,
                               "(f) DNN Temporal Error (Common)", config.OUTPUT_DIR)

    print(f"\nAll outputs saved to: {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
