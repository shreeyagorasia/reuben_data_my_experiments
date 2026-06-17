"""
models/pinn_baseline/train.py
===============================
Trains the physics-informed neural network (PINN) to predict 2023 tree heights from 2012 features (+ AGE),
using the Chapman-Richards as a physics prior.

Requires outputs/chapman_richards/cr_params.json -- run models/chapman_richards/train.py first.

Saves everything into outputs/pinn_baseline/:
  - checkpoint.pt        : trained model + scalers + history (Table 4.2 only)
  - config_used.json      : snapshot of the hyperparameters for this run (Table 4.2 only)
  - results.json          : evaluation metrics
                          -     Reuben: "MAE": 4.0424, "MSE": 24.1570, "R²": 0.4378, "MRE": 0.1539, "Acc%": 84.61

  - training_curve.png    : train/val loss over training (Table 4.2 only)

Run with (from this folder):
    python train.py                 # runs all 4 tables (4.1, 4.3, 4.4, 4.2), like before
    python train.py --table 4.1     # runs just Table 4.1
    python train.py --table 4.3     # runs just the Table 4.3 3-fold CV
    python train.py --table 4.4     # runs just the Table 4.4 3-fold CV
    python train.py --table 4.2     # runs just Table 4.2 (also saves checkpoint/plots)

Each run merges its results into the existing outputs/pinn_baseline/results.csv
rather than overwriting the rows from other tables.
"""

import os
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "12355")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import json
import csv
import argparse

import config  # this model's config (also pulls in common settings)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from common.data_utils import (
    load_data,
    build_feature_arrays,
    split_age_column,
    make_scalers,
    train_val_split,
    to_tensors,
    get_kfold_splits,
)
from common.metrics import reuben_metrics
from model import PINN, pinn_loss

def run_training(X_tr_full, y_tr_full, X_te_full, y_te_full, cr_params, device, run_name=""):
    """Handles scaling, splitting, and training the PINN for a single given subset of data."""
    # 1. FEATURE SPLITTING & SCALING:
    # Separate the 'AGE' column from the other features. This is essential for the PINN
    # because we need to compute the derivative of the output with respect to AGE.
    (X_tr_o, X_tr_a, X_te_o, X_te_a, age_idx, other_idxs) = split_age_column(X_tr_full, X_te_full)
    # Create separate scalers for other features, age, and the target.
    # They are all fit on the 2012 training data only to prevent leakage.
    scaler_Xo, scaler_age, scaler_y = make_scalers(X_tr_o, X_tr_a, y_tr_full)

    # These scaling factors are needed for the chain rule in the physics loss calculation.
    sigma_y = float(scaler_y.scale_[0])
    sigma_age = float(scaler_age.scale_[0])
    age_mean = float(scaler_age.mean_[0])

    # 2. VALIDATION SPLIT: Create an internal validation set for early stopping.
    tr_idx, val_idx = train_val_split(len(X_tr_full), val_fraction=config.VAL_SPLIT)

    # 3. TENSOR CONVERSION: Create scaled tensors for train and validation sets.
    # Note the separate tensors for 'other' features (Xtr_o) and 'age' (Xtr_a).
    Xtr_o, Xtr_a, ytr = to_tensors(tr_idx, X_tr_o, X_tr_a, y_tr_full, scaler_Xo, scaler_age, scaler_y)
    Xval_o, Xval_a, yval = to_tensors(val_idx, X_tr_o, X_tr_a, y_tr_full, scaler_Xo, scaler_age, scaler_y)

    # Move validation tensors to the GPU.
    Xtr_o, Xtr_a, ytr = Xtr_o.to(device), Xtr_a.to(device), ytr.to(device)
    Xval_o, Xval_a, yval = Xval_o.to(device), Xval_a.to(device), yval.to(device)

    # Scale and move the final test set tensors to the GPU.
    Xte_o = torch.tensor(scaler_Xo.transform(X_te_o), dtype=torch.float32).to(device)
    Xte_a = torch.tensor(scaler_age.transform(X_te_a), dtype=torch.float32).to(device)

    # DataLoader will handle batching and shuffling of the three training tensors.
    loader = DataLoader(TensorDataset(Xtr_o, Xtr_a, ytr), batch_size=config.BATCH_SIZE, shuffle=True)

    # 4. MODEL SETUP
    pinn = PINN(n_other=len(other_idxs)).to(device)
    optimiser = torch.optim.Adam(pinn.parameters(), lr=config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode="min", factor=0.8, patience=10)
    mse_fn = nn.MSELoss()

    # 5. TRAINING LOOP
    train_hist, val_hist, epoch_log = [], [], []
    best_val = float("inf")
    patience_ctr = 0

    if run_name:
        print(f"\n--- {run_name} ---")
        print(f"  {'Epoch':>4} | {'Train loss':>12} | {'Val loss':>12}")
        print("-" * 38)

    for epoch in range(config.EPOCHS):
        pinn.train()
        running, n_batches = 0.0, 0
        for Xb_o, Xb_a, yb in loader:
            Xb_o, Xb_a, yb = Xb_o.to(device), Xb_a.to(device), yb.to(device)
            optimiser.zero_grad()

            # THIS IS THE KEY FOR THE PHYSICS LOSS:
            # The age tensor for the batch (`Xb_a`) must have `requires_grad=True`.
            # This tells PyTorch to track operations on it for automatic differentiation.
            t_age = Xb_a.clone().requires_grad_(True)
            pred = pinn(Xb_o, t_age)

            # Calculate the custom PINN loss, which returns both the data and physics components.
            data_loss, phys_loss = pinn_loss(
                pred, t_age, yb, mse_fn, cr_params, sigma_y, sigma_age, age_mean
            )
            # L1 regularization term
            l1 = sum(p.abs().sum() for p in pinn.parameters())
            # The final loss is a weighted sum of the three components.
            loss = data_loss + config.LAMBDA_PH * phys_loss + config.LAMBDA_L1 * l1

            loss.backward()
            optimiser.step()
            running += (data_loss + config.LAMBDA_PH * phys_loss).item()
            n_batches += 1

        avg_train = running / n_batches
        pinn.eval()
        with torch.no_grad():
            # Validation loss is just standard MSE on the validation set.
            # The physics loss is only used for training the model.
            val_loss = mse_fn(pinn(Xval_o, Xval_a), yval).item()

        scheduler.step(val_loss)

        # Early stopping logic
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= config.EARLY_STOP_PATIENCE:
                epoch_log.append(epoch + 1); train_hist.append(avg_train); val_hist.append(val_loss)
                if run_name: print(f"    Early stopping at epoch {epoch + 1}")
                break

        if (epoch + 1) % 10 == 0:
            epoch_log.append(epoch + 1); train_hist.append(avg_train); val_hist.append(val_loss)
            if run_name: print(f"  {epoch + 1:>4}  |  {avg_train:>12.6f}  |  {val_loss:>12.6f}", flush=True)

    if not epoch_log or epoch_log[-1] != (epoch + 1):
        epoch_log.append(epoch + 1); train_hist.append(avg_train); val_hist.append(val_loss)

    # 6. FINAL EVALUATION
    pinn.eval()
    with torch.no_grad():
        # Get scaled predictions and inverse_transform them back to metres.
        y_pred_scaled = pinn(Xte_o, Xte_a).cpu().numpy().reshape(-1, 1)
        y_pred = scaler_y.inverse_transform(y_pred_scaled).ravel()

    # Calculate final performance metrics.
    metrics = reuben_metrics(y_te_full, y_pred, label=run_name if run_name else "CV Fold")

    return metrics, epoch_log, train_hist, val_hist, y_pred, pinn, optimiser, scaler_Xo, scaler_age, scaler_y, other_idxs, age_idx


def load_cr_params():
    if not os.path.exists(config.CR_PARAMS_PATH):
        raise FileNotFoundError(
            f"Could not find {config.CR_PARAMS_PATH}.\n"
            f"Run models/chapman_richards/train.py first."
        )
    with open(config.CR_PARAMS_PATH) as f:
        all_cr_params = json.load(f)

    # Accommodate both the new refactored multi-key dict and backwards compatibility
    cr_params_purged = all_cr_params.get("Table4.1", all_cr_params)
    cr_params_unseen = all_cr_params.get("Table4.2", all_cr_params)
    print(f"\nLoaded CR prior (Purged): {cr_params_purged}")
    print(f"Loaded CR prior (Unseen): {cr_params_unseen}")
    return cr_params_purged, cr_params_unseen


def run_table_4_1(cr_params_purged, device):
    print("\n\n--- Loading PURGED data for Table 4.1 ---")
    df12, df23, _ = load_data(config.DATA_PATH_PURGED)
    X_tr, X_te, y_tr, y_te, X_coords, Y_coords = build_feature_arrays(df12, df23)

    print("\n--- Running Experiment for Table 4.1: Temporal (Common Plots) ---")
    (metrics, ep_log, tr_hist, val_hist, y_pred, pinn, optimiser,
     scaler_Xo, scaler_age, scaler_y, other_idxs, age_idx) = run_training(
        X_tr, y_tr, X_te, y_te, cr_params_purged, device, "Temporal (Table 4.1)"
    )

    table_results = {f"Table4.1_{k}": v for k, v in metrics.items()}
    extras = {
        "metrics": metrics,
        "ep_log": ep_log, "tr_hist": tr_hist, "val_hist": val_hist,
        "y_pred": y_pred, "y_te": y_te, "pinn": pinn, "optimiser": optimiser,
        "scaler_Xo": scaler_Xo, "scaler_age": scaler_age, "scaler_y": scaler_y,
        "other_idxs": other_idxs, "age_idx": age_idx,
        "X_coords": X_coords, "Y_coords": Y_coords,
        "cr_params": cr_params_purged,
    }
    return table_results, extras


def run_table_4_3(cr_params_purged, device):
    print("\n\n--- Loading PURGED data for Table 4.3 ---")
    df12, df23, _ = load_data(config.DATA_PATH_PURGED)
    X_tr, _, y_tr, _, _, _ = build_feature_arrays(df12, df23)

    print("\n--- Running Experiment for Table 4.3: 3-Fold CV within 2012 (Purged) ---")
    cv_metrics = []
    cv_test_plot_ids = []
    for fold, (tr_idx, te_idx) in enumerate(get_kfold_splits(len(X_tr), n_splits=3)):
        print(f"  Training Fold {fold+1}/3...")
        m, *_ = run_training(X_tr[tr_idx], y_tr[tr_idx], X_tr[te_idx], y_tr[te_idx], cr_params_purged, device)
        cv_metrics.append(m)
        cv_test_plot_ids.append(df12.index.values[te_idx].tolist())
    metrics_out = {f"Table4.3_{k}": float(np.mean([m[k] for m in cv_metrics])) for k in cv_metrics[0].keys()}
    return metrics_out, cv_test_plot_ids


def run_table_4_4(cr_params_purged, device):
    print("\n\n--- Loading PURGED data for Table 4.4 ---")
    df12, df23, _ = load_data(config.DATA_PATH_PURGED)
    _, X_te, _, y_te, _, _ = build_feature_arrays(df12, df23)

    print("\n--- Running Experiment for Table 4.4: 3-Fold CV within 2023 (Purged) ---")
    cv_metrics = []
    cv_test_plot_ids = []
    for fold, (tr_idx, te_idx) in enumerate(get_kfold_splits(len(X_te), n_splits=3)):
        print(f"  Training Fold {fold+1}/3...")
        m, *_ = run_training(X_te[tr_idx], y_te[tr_idx], X_te[te_idx], y_te[te_idx], cr_params_purged, device)
        cv_metrics.append(m)
        cv_test_plot_ids.append(df23.index.values[te_idx].tolist())
    metrics_out = {f"Table4.4_{k}": float(np.mean([m[k] for m in cv_metrics])) for k in cv_metrics[0].keys()}
    return metrics_out, cv_test_plot_ids


def run_table_4_2(cr_params_unseen, device):
    print("\n\n--- Loading UNSEEN data for Table 4.2 ---")
    df12, df23, _ = load_data(config.DATA_PATH_UNSEEN)
    X_tr, X_te, y_tr, y_te, X_coords, Y_coords = build_feature_arrays(df12, df23)

    print("\n--- Running Experiment for Table 4.2: Temporal (Unseen Plots) ---")
    (metrics, ep_log, tr_hist, val_hist, y_pred, pinn, optimiser,
     scaler_Xo, scaler_age, scaler_y, other_idxs, age_idx) = run_training(
        X_tr, y_tr, X_te, y_te, cr_params_unseen, device, "Temporal (Table 4.2)"
    )

    table_results = {f"Table4.2_{k}": v for k, v in metrics.items()}
    extras = {
        "metrics": metrics,
        "ep_log": ep_log, "tr_hist": tr_hist, "val_hist": val_hist,
        "y_pred": y_pred, "y_te": y_te, "pinn": pinn, "optimiser": optimiser,
        "scaler_Xo": scaler_Xo, "scaler_age": scaler_age, "scaler_y": scaler_y,
        "other_idxs": other_idxs, "age_idx": age_idx,
        "X_coords": X_coords, "Y_coords": Y_coords,
        "cr_params": cr_params_unseen,
    }
    return table_results, extras


def load_existing_results():
    """Read outputs/pinn_baseline/results.csv if it exists, so individual
    table runs can be merged in without clobbering the other tables' rows."""
    path = os.path.join(config.OUTPUT_DIR, "results.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        return dict(zip(df["Metric"], df["Value"]))
    return {}


def save_results(results):
    path = os.path.join(config.OUTPUT_DIR, "results.csv")
    pd.DataFrame(list(results.items()), columns=["Metric", "Value"]).to_csv(path, index=False)
    print(f"\nSaved combined multi-experiment metrics to {path}")


def main():
    parser = argparse.ArgumentParser(description="Train the PINN baseline on one or more of Reuben's 4 tables.")
    parser.add_argument(
        "--table", choices=["4.1", "4.2", "4.3", "4.4", "all"], default="all",
        help="Which experiment to run (default: all -- runs every table, same as the original behaviour)."
    )
    args = parser.parse_args()
    tables = ["4.1", "4.3", "4.4", "4.2"] if args.table == "all" else [args.table]

    np.random.seed(config.RANDOM_SEED)
    torch.manual_seed(config.RANDOM_SEED)

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    cr_params_purged, cr_params_unseen = load_cr_params()

    results = load_existing_results()
    table_4_1_extras = None
    table_4_2_extras = None
    cv_test_plot_ids_4_3 = None
    cv_test_plot_ids_4_4 = None

    if "4.1" in tables:
        t1_results, table_4_1_extras = run_table_4_1(cr_params_purged, device)
        results.update(t1_results)
    if "4.3" in tables:
        t3_results, cv_test_plot_ids_4_3 = run_table_4_3(cr_params_purged, device)
        results.update(t3_results)
    if "4.4" in tables:
        t4_results, cv_test_plot_ids_4_4 = run_table_4_4(cr_params_purged, device)
        results.update(t4_results)
    if "4.2" in tables:
        t2_results, table_4_2_extras = run_table_4_2(cr_params_unseen, device)
        results.update(t2_results)

    save_results(results)

    # ------------------------------------------------------------
    # Checkpoint / plots / config snapshot are now ONLY based on Table 4.1.
    # They will only be generated if that experiment was run.
    # ------------------------------------------------------------
    if table_4_1_extras is not None:
        e = table_4_1_extras
        checkpoint = {
            "model_state": e["pinn"].state_dict(),
            "optimizer_state": e["optimiser"].state_dict(),
            "history": {
                "train_loss": e["tr_hist"],
                "val_loss": e["val_hist"],
                "epochs": e["ep_log"],
            },
            "feature_list": config.FEATURES,
            "other_idxs": e["other_idxs"],
            "age_idx": e["age_idx"],
            "scaler_Xo": e["scaler_Xo"],
            "scaler_age": e["scaler_age"],
            "scaler_y": e["scaler_y"],
            "cr_params": e["cr_params"],
            "final_metrics": e["metrics"],
            "cv_test_plot_ids": {
                "Table4.3": cv_test_plot_ids_4_3,
                "Table4.4": cv_test_plot_ids_4_4,
            },
        }
        ckpt_path = os.path.join(config.OUTPUT_DIR, "checkpoint.pt")
        torch.save(checkpoint, ckpt_path)
        print(f"\nSaved checkpoint to {ckpt_path}")

        config_used = {
            "hidden_size": config.HIDDEN_SIZE,
            "lambda_ph": config.LAMBDA_PH,
            "lambda_l1": config.LAMBDA_L1,
            "epochs_max": config.EPOCHS,
            "epochs_run": e["ep_log"][-1] if e["ep_log"] else 0,
            "batch_size": config.BATCH_SIZE,
            "learning_rate": config.LEARNING_RATE,
            "early_stop_patience": config.EARLY_STOP_PATIENCE,
            "val_split": config.VAL_SPLIT,
            "random_seed": config.RANDOM_SEED,
        }
        config_path = os.path.join(config.OUTPUT_DIR, "config_used.json")
        with open(config_path, "w") as f:
            json.dump(config_used, f, indent=2)


if __name__ == "__main__":
    main()