"""
models/pinn_baseline/train.py
===============================
Table 4.1: Trains a Physics-Informed Neural Network (PINN) to predict 2023
tree heights from 2012 features (AGE + plot characteristics).

What makes this different from the plain DNN:
  The loss function has two parts:
    1. Data loss   — the usual "how wrong are my predictions?" MSE.
    2. Physics loss — "does my predicted GROWTH RATE (dHeight/dAge) match
                       the Chapman-Richards curve?"  (computed with autograd)
  These are weighted and summed:  loss = data_loss + λ_phys * physics_loss

IMPORTANT: Run models/chapman_richards/train.py FIRST.
           This script loads outputs/chapman_richards/cr_params.json.

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

import config
from common.data_utils import (
    load_data,
    build_feature_arrays,
    split_age_column,
    make_scalers,
    train_val_split,
    to_tensors,
)
from common.metrics import reuben_metrics
from model import PINN, pinn_loss
from plots import plot_spatial_error, plot_spatial_signed_error, plot_training_curve


def load_cr_params():
    """Load the Chapman-Richards parameters saved by chapman_richards/train.py."""
    if not os.path.exists(config.CR_PARAMS_PATH):
        raise FileNotFoundError(
            f"Cannot find {config.CR_PARAMS_PATH}.\n"
            f"Run models/chapman_richards/train.py first."
        )
    with open(config.CR_PARAMS_PATH) as f:
        all_params = json.load(f)
    # The CR script saves params per-table; we want the Table 4.1 (purged) ones.
    cr_params = all_params.get("Table4.1", all_params)
    print(f"\nLoaded CR prior: {cr_params}")
    return cr_params


def train_model(X_train, y_train, X_test, cr_params, device):
    """
    Scales data, trains the PINN, and returns predictions on X_test.

    The PINN needs AGE separated from other features so that autograd can
    compute d(prediction)/d(AGE) for the physics loss.

    Parameters
    ----------
    X_train   : numpy array (n_train, n_features) — 2012 feature matrix
    y_train   : numpy array (n_train,)             — 2012 heights (labels)
    X_test    : numpy array (n_test,  n_features) — 2023 feature matrix
    cr_params : dict with y_max, k, p — physics prior from Chapman-Richards
    device    : torch.device

    Returns
    -------
    y_pred     : numpy array of predicted 2023 heights (metres)
    epoch_log  : list of epoch numbers that were logged
    train_hist : list of training losses per logged epoch
    val_hist   : list of validation losses per logged epoch
    """

    # ------------------------------------------------------------------ #
    # STEP 1 — SPLIT AGE OUT OF THE FEATURE MATRIX                        #
    # ------------------------------------------------------------------ #
    # AGE needs its own tensor because we differentiate through it.
    # X_other = all features except AGE
    # X_age   = just the AGE column, shape (n, 1)
    X_tr_other, X_tr_age, X_te_other, X_te_age, age_idx, other_idxs = \
        split_age_column(X_train, X_test)

    # ------------------------------------------------------------------ #
    # STEP 2 — SCALE                                                       #
    # ------------------------------------------------------------------ #
    # Fit scalers on 2012 training data only (avoids data leakage).
    # Separate scalers for "other" features, age, and the target height.
    scaler_Xo, scaler_age, scaler_y = make_scalers(X_tr_other, X_tr_age, y_train)

    # The physics loss chain rule needs these scaling factors
    sigma_y   = float(scaler_y.scale_[0])
    sigma_age = float(scaler_age.scale_[0])
    age_mean  = float(scaler_age.mean_[0])

    # ------------------------------------------------------------------ #
    # STEP 3 — VALIDATION SPLIT & TENSORS                                  #
    # ------------------------------------------------------------------ #
    tr_idx, val_idx = train_val_split(len(X_train), val_fraction=config.VAL_SPLIT)

    Xtr_o, Xtr_a, ytr   = to_tensors(tr_idx,  X_tr_other, X_tr_age, y_train, scaler_Xo, scaler_age, scaler_y)
    Xval_o, Xval_a, yval = to_tensors(val_idx, X_tr_other, X_tr_age, y_train, scaler_Xo, scaler_age, scaler_y)

    Xtr_o, Xtr_a, ytr    = Xtr_o.to(device), Xtr_a.to(device), ytr.to(device)
    Xval_o, Xval_a, yval = Xval_o.to(device), Xval_a.to(device), yval.to(device)

    # Scale and move the 2023 test features to the GPU
    Xte_o = torch.tensor(scaler_Xo.transform(X_te_other), dtype=torch.float32).to(device)
    Xte_a = torch.tensor(scaler_age.transform(X_te_age),  dtype=torch.float32).to(device)

    loader = DataLoader(TensorDataset(Xtr_o, Xtr_a, ytr), batch_size=config.BATCH_SIZE, shuffle=True)

    # ------------------------------------------------------------------ #
    # STEP 4 — BUILD MODEL                                                 #
    # ------------------------------------------------------------------ #
    pinn      = PINN(n_other=len(other_idxs)).to(device)
    optimiser = torch.optim.Adam(pinn.parameters(), lr=config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode="min", factor=0.8, patience=10)
    mse_fn    = nn.MSELoss()

    # ------------------------------------------------------------------ #
    # STEP 5 — TRAINING LOOP                                               #
    # ------------------------------------------------------------------ #
    train_hist, val_hist, epoch_log = [], [], []
    best_val     = float("inf")
    patience_ctr = 0

    print(f"\n  {'Epoch':>4} | {'Train loss':>12} | {'Val loss':>12}")
    print("  " + "-" * 36)

    for epoch in range(config.EPOCHS):
        pinn.train()
        running, n_batches = 0.0, 0

        for Xb_o, Xb_a, yb in loader:
            Xb_o, Xb_a, yb = Xb_o.to(device), Xb_a.to(device), yb.to(device)
            optimiser.zero_grad()

            # KEY STEP: requires_grad=True on the age tensor lets PyTorch
            # compute d(prediction)/d(AGE) automatically (autograd).
            t_age = Xb_a.clone().requires_grad_(True)
            pred  = pinn(Xb_o, t_age)

            # Physics-informed loss = data MSE + CR growth-rate consistency
            data_loss, phys_loss = pinn_loss(
                pred, t_age, yb, mse_fn, cr_params, sigma_y, sigma_age, age_mean
            )
            # L1 regularisation: lightly penalises large weights
            l1 = sum(p.abs().sum() for p in pinn.parameters())

            loss = data_loss + config.LAMBDA_PH * phys_loss + config.LAMBDA_L1 * l1
            loss.backward()
            optimiser.step()
            running  += (data_loss + config.LAMBDA_PH * phys_loss).item()
            n_batches += 1

        avg_train = running / n_batches

        pinn.eval()
        with torch.no_grad():
            val_loss = mse_fn(pinn(Xval_o, Xval_a), yval).item()

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

    if not epoch_log or epoch_log[-1] != (epoch + 1):
        epoch_log.append(epoch + 1); train_hist.append(avg_train); val_hist.append(val_loss)

    # ------------------------------------------------------------------ #
    # STEP 6 — PREDICT ON TEST SET                                         #
    # ------------------------------------------------------------------ #
    pinn.eval()
    with torch.no_grad():
        y_pred_scaled = pinn(Xte_o, Xte_a).cpu().numpy().reshape(-1, 1)
        y_pred = scaler_y.inverse_transform(y_pred_scaled).ravel()

    return y_pred, epoch_log, train_hist, val_hist, pinn, optimiser, scaler_Xo, scaler_age, scaler_y


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
    # 1. LOAD CR PARAMS (physics prior) & DATA                            #
    # ------------------------------------------------------------------ #
    cr_params = load_cr_params()

    df12, df23, _ = load_data(config.DATA_PATH_PURGED)
    X_train, X_test, y_train, y_test, X_coords, Y_coords = build_feature_arrays(df12, df23)

    # ------------------------------------------------------------------ #
    # 2. TRAIN & PREDICT                                                   #
    # ------------------------------------------------------------------ #
    print("\n--- Training PINN (Table 4.1: Temporal, Common Plots) ---")
    y_pred, ep_log, tr_hist, val_hist, pinn, optimiser, scaler_Xo, scaler_age, scaler_y = \
        train_model(X_train, y_train, X_test, cr_params, device)

    # ------------------------------------------------------------------ #
    # 3. EVALUATE                                                          #
    # ------------------------------------------------------------------ #
    metrics = reuben_metrics(y_test, y_pred, label="PINN Baseline (Table 4.1)")

    # ------------------------------------------------------------------ #
    # 4. SAVE OUTPUTS                                                      #
    # ------------------------------------------------------------------ #
    torch.save(
        {
            "model_state":     pinn.state_dict(),
            "optimizer_state": optimiser.state_dict(),
            "scaler_Xo":       scaler_Xo,
            "scaler_age":      scaler_age,
            "scaler_y":        scaler_y,
            "feature_list":    config.FEATURES,
            "cr_params":       cr_params,
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
                "epochs_max":          config.EPOCHS,
                "epochs_run":          ep_log[-1] if ep_log else 0,
                "batch_size":          config.BATCH_SIZE,
                "learning_rate":       config.LEARNING_RATE,
                "lambda_ph":           config.LAMBDA_PH,
                "lambda_l1":           config.LAMBDA_L1,
                "early_stop_patience": config.EARLY_STOP_PATIENCE,
                "val_split":           config.VAL_SPLIT,
                "hidden_size":         config.HIDDEN_SIZE,
                "features":            config.FEATURES,
                "cr_params_used":      cr_params,
                "final_metrics":       metrics,
            },
            f, indent=2,
        )

    plot_training_curve(ep_log, tr_hist, val_hist, config.OUTPUT_DIR)
    plot_spatial_error(X_coords, Y_coords, y_test, y_pred,
                       "(g) PINN Baseline (Common)", config.OUTPUT_DIR)
    plot_spatial_signed_error(X_coords, Y_coords, y_test, y_pred,
                               "(g) PINN Baseline (Common)", config.OUTPUT_DIR)

    print(f"\nAll outputs saved to: {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
