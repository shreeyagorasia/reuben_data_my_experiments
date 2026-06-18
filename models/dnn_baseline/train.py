"""
models/dnn_baseline/train.py
==============================
Trains the Deep Neural Network (DNN) baseline.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

import config
from common.data_utils import load_data, build_feature_arrays, train_val_split, get_kfold_splits
from common.metrics import reuben_metrics
from model import DNN
from plots import plot_spatial_error, plot_spatial_signed_error, plot_training_curve

def run_training(X_tr_full, y_tr_full, X_te_full, y_te_full, device, run_name=""):
    """Handles scaling, splitting, and training for a single run."""
    # 1. SCALING: Fit scalers on the TRAINING data (2012) ONLY.
    # This prevents data leakage from the test set's statistics (mean, std).
    scaler_X = StandardScaler().fit(X_tr_full)
    scaler_y = StandardScaler().fit(y_tr_full.reshape(-1, 1))
    
    # 2. VALIDATION SPLIT: Create an internal validation set from the training data.
    # This is used for early stopping to prevent overfitting.
    tr_idx, val_idx = train_val_split(len(X_tr_full), val_fraction=config.VAL_SPLIT)
    
    # 3. TENSOR CONVERSION: Scale the data and convert to PyTorch tensors.
    # Training data is kept on the CPU and moved to the GPU in batches by the DataLoader.
    Xtr_t = torch.tensor(scaler_X.transform(X_tr_full[tr_idx]), dtype=torch.float32)
    ytr_t = torch.tensor(scaler_y.transform(y_tr_full[tr_idx].reshape(-1, 1)).ravel(), dtype=torch.float32)
    # Validation and Test data are moved to the GPU immediately since they are used all at once.
    Xval_t = torch.tensor(scaler_X.transform(X_tr_full[val_idx]), dtype=torch.float32).to(device)
    yval_t = torch.tensor(scaler_y.transform(y_tr_full[val_idx].reshape(-1, 1)).ravel(), dtype=torch.float32).to(device)
    Xte = torch.tensor(scaler_X.transform(X_te_full), dtype=torch.float32).to(device)
    
    # The DataLoader handles batching, shuffling, and moving data to the GPU.
    train_loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=config.BATCH_SIZE, shuffle=True)
    
    # 4. MODEL SETUP: Instantiate the model, optimizer, scheduler, and loss function.
    dnn = DNN(n_features=X_tr_full.shape[1]).to(device)
    optimiser = torch.optim.Adam(dnn.parameters(), lr=config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=0.8, patience=10)
    loss_fn = nn.MSELoss()
    
    # 5. TRAINING LOOP
    train_hist, val_hist, epoch_log = [], [], []
    best_val = float('inf')
    patience_ctr = 0
    
    print(f"  {'Epoch':>4} | {'Train loss':>12} | {'Val loss':>12}")
    print("-" * 38)

    for epoch in range(config.EPOCHS):
        # Set model to training mode
        dnn.train()
        running, batch_count = 0.0, 0
        for Xb, yb in train_loader:
            # Move batch to the GPU
            Xb, yb = Xb.to(device), yb.to(device)
            # Reset gradients from previous batch
            optimiser.zero_grad()
            # Forward pass: get predictions
            pred = dnn(Xb)
            # L1 regularization term: penalizes large weights to prevent overfitting
            l1 = sum(p.abs().sum() for p in dnn.parameters())
            # Calculate loss: standard MSE + L1 penalty
            loss = loss_fn(pred, yb) + config.LAMBDA_L1 * l1
            # Backward pass: compute gradients
            loss.backward()
            # Update weights
            optimiser.step()
            # Accumulate loss for logging
            running += loss.item()
            batch_count += 1
            
        avg_train = running / batch_count
        # After each epoch, evaluate on the validation set
        dnn.eval()
        with torch.no_grad():
            val_loss = loss_fn(dnn(Xval_t), yval_t).item()
            
        # Adjust learning rate based on validation loss
        scheduler.step(val_loss)

        # Early stopping logic: if validation loss doesn't improve for a
        # number of epochs (patience), stop training.
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= config.EARLY_STOP_PATIENCE:
                epoch_log.append(epoch + 1); train_hist.append(avg_train); val_hist.append(val_loss)
                print(f"    Early stopping at epoch {epoch + 1} - val_loss plateaued")
                break
                
        if (epoch + 1) % 10 == 0:
            epoch_log.append(epoch + 1); train_hist.append(avg_train); val_hist.append(val_loss)
            print(f"  {epoch + 1:>4}  |  {avg_train:>12.6f}  |  {val_loss:>12.6f}", flush=True)

    if not epoch_log:
        epoch_log.append(epoch + 1); train_hist.append(avg_train); val_hist.append(val_loss)
        
    # 6. FINAL EVALUATION: Use the trained model to predict on the test set.
    dnn.eval()
    with torch.no_grad():
        # Get scaled predictions and then inverse_transform them back to original units (metres).
        y_pred = scaler_y.inverse_transform(dnn(Xte).cpu().numpy().reshape(-1, 1)).ravel()
    
    # We only print the metrics output if it's the main temporal run
    if run_name:
        print(f"\n--- {run_name} ---")
    # Calculate final performance metrics (MAE, R², etc.)
    metrics = reuben_metrics(y_te_full, y_pred, label=run_name)

    return metrics, epoch_log, train_hist, val_hist, y_pred, dnn, optimiser, scaler_X, scaler_y

def main():
    np.random.seed(config.RANDOM_SEED)
    torch.manual_seed(config.RANDOM_SEED)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # Auto-detect GPU (CUDA for cluster, MPS for Mac, or CPU fallback)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    all_results = {}

    # --- Experiments using PURGED data (Tables 4.1, 4.3, 4.4) ---
    print("\n\n--- Loading PURGED data for Tables 4.1, 4.3, 4.4 ---")
    df12_purged, df23_purged, _ = load_data(config.DATA_PATH_PURGED)
    X_train_purged, X_test_purged, y_train_purged, y_test_purged, X_coords_purged, Y_coords_purged = build_feature_arrays(
        df12_purged, df23_purged
    )

    # --- Experiment for Table 4.1 (Temporal, Common Plots) ---
    print("\n--- Running Experiment for Table 4.1: Temporal (Common Plots) ---")
    metrics_t1, ep_log_t1, tr_hist_t1, val_hist_t1, y_pred_t1, dnn_t1, optimiser_t1, scaler_X_t1, scaler_y_t1 = run_training(
        X_train_purged, y_train_purged, X_test_purged, y_test_purged, device, "Temporal (Table 4.1)"
    )
    for k, v in metrics_t1.items():
        all_results[f"Table4.1_{k}"] = v

    # --- Experiment for Table 4.3 (3-Fold CV within 2012) ---
    print(f"\n--- Running Experiment for Table 4.3: 3-Fold CV within 2012 (Purged) ---")
    cv12_metrics = []
    cv43_test_plot_ids = []
    for fold, (tr_idx, te_idx) in enumerate(get_kfold_splits(len(X_train_purged), n_splits=3)):
        print(f"  Training Fold {fold+1}/3...")
        m, *_ = run_training(
            X_train_purged[tr_idx], y_train_purged[tr_idx],
            X_train_purged[te_idx], y_train_purged[te_idx], device
        )
        cv12_metrics.append(m)
        cv43_test_plot_ids.append(df12_purged.index.values[te_idx].tolist())

    for k in cv12_metrics[0].keys():
        all_results[f"Table4.3_{k}"] = float(np.mean([m[k] for m in cv12_metrics]))

    # --- Experiment for Table 4.4 (3-Fold CV within 2023) ---
    print(f"\n--- Running Experiment for Table 4.4: 3-Fold CV within 2023 (Purged) ---")
    cv23_metrics = []
    cv44_test_plot_ids = []
    for fold, (tr_idx, te_idx) in enumerate(get_kfold_splits(len(X_test_purged), n_splits=3)):
        print(f"  Training Fold {fold+1}/3...")
        m, *_ = run_training(
            X_test_purged[tr_idx], y_test_purged[tr_idx],
            X_test_purged[te_idx], y_test_purged[te_idx], device
        )
        cv23_metrics.append(m)
        cv44_test_plot_ids.append(df23_purged.index.values[te_idx].tolist())

    for k in cv23_metrics[0].keys():
        all_results[f"Table4.4_{k}"] = float(np.mean([m[k] for m in cv23_metrics]))

    # --- Experiment using UNSEEN data (Table 4.2) ---
    print("\n\n--- Loading UNSEEN data for Table 4.2 ---")
    df12_unseen, df23_unseen, _ = load_data(config.DATA_PATH_UNSEEN)
    X_train_unseen, X_test_unseen, y_train_unseen, y_test_unseen, X_coords_unseen, Y_coords_unseen = build_feature_arrays(
        df12_unseen, df23_unseen
    )
    print(f"\n--- Running Experiment for Table 4.2: Temporal (Unseen Plots) ---")
    metrics_t2, ep_log, tr_hist, val_hist, y_pred_unseen, dnn_t2, optimiser_t2, scaler_X_t2, scaler_y_t2 = run_training(
        X_train_unseen, y_train_unseen, X_test_unseen, y_test_unseen, device, "Temporal (Table 4.2)"
    )
    for k, v in metrics_t2.items():
        all_results[f"Table4.2_{k}"] = v

    # Checkpoint is now based on Table 4.1
    checkpoint = {
        "model_state": dnn_t1.state_dict(),
        "optimizer_state": optimiser_t1.state_dict(),
        "history": {
            "train_loss": tr_hist_t1,
            "val_loss": val_hist_t1,
            "epochs": ep_log_t1,
        },
        "feature_list": config.FEATURES,
        "scaler_X": scaler_X_t1,
        "scaler_y": scaler_y_t1,
        "final_metrics": metrics_t1,
        "cv_test_plot_ids": {
            "Table4.3": cv43_test_plot_ids,
            "Table4.4": cv44_test_plot_ids,
        },
    }
    ckpt_path = os.path.join(config.OUTPUT_DIR, "checkpoint.pt")
    torch.save(checkpoint, ckpt_path)
    print(f"\nSaved checkpoint to {ckpt_path}")

    plot_training_curve(ep_log_t1, tr_hist_t1, val_hist_t1, config.OUTPUT_DIR)
    plot_spatial_error(
        X_coords_purged,
        Y_coords_purged,
        y_test_purged,
        y_pred_t1,
        "(f) DNN Temporal Error (Common)",
        config.OUTPUT_DIR,
    )
    plot_spatial_signed_error(
        X_coords_purged,
        Y_coords_purged,
        y_test_purged,
        y_pred_t1,
        "(f) DNN Temporal Error (Common)",
        config.OUTPUT_DIR,
    )

    config_used = {
        "model_name": config.MODEL_NAME,
        "architecture": {
            "type": "DNN",
            "input_features": len(config.FEATURES),
            "hidden_size": config.HIDDEN_SIZE,
            "hidden_layers": 3,
            "activation": "LeakyReLU",
            "output_size": 1,
        },
        "training": {
            "epochs_max": config.EPOCHS,
            "epochs_run_table4_1": ep_log_t1[-1] if ep_log_t1 else 0,
            "batch_size": config.BATCH_SIZE,
            "learning_rate": config.LEARNING_RATE,
            "optimizer": "Adam",
            "scheduler": {
                "type": "ReduceLROnPlateau",
                "mode": "min",
                "factor": 0.8,
                "patience": 10,
            },
            "lambda_l1": config.LAMBDA_L1,
            "early_stop_patience": config.EARLY_STOP_PATIENCE,
            "val_split": config.VAL_SPLIT,
            "random_seed": config.RANDOM_SEED,
            "device": str(device),
        },
        "data_paths": {
            "purged": config.DATA_PATH_PURGED,
            "unseen": config.DATA_PATH_UNSEEN,
        },
        "feature_list": config.FEATURES,
        "target_col": config.TARGET_COL,
        "tables_run": ["4.1", "4.3", "4.4", "4.2"],
        "data_counts": {
            "purged_2012_rows": int(len(df12_purged)),
            "purged_2023_rows": int(len(df23_purged)),
            "unseen_2012_rows": int(len(df12_unseen)),
            "unseen_2023_rows": int(len(df23_unseen)),
        },
        "final_metrics": {
            "Table4.1": metrics_t1,
            "Table4.2": metrics_t2,
            "Table4.3": {k: all_results[f"Table4.3_{k}"] for k in metrics_t1.keys()},
            "Table4.4": {k: all_results[f"Table4.4_{k}"] for k in metrics_t1.keys()},
        },
        "checkpoint_basis": "Table4.1",
        "output_files": {
            "results": "results.csv",
            "checkpoint": "checkpoint.pt",
            "config_used": "config_used.json",
            "training_curve": "training_curve.png",
            "spatial_error_map": "spatial_error_map.png (Table 4.1 temporal common plots)",
            "spatial_error_metadata": "spatial_error_map.json",
            "spatial_signed_error_map": "spatial_signed_error_map.png (Table 4.1 temporal common plots)",
            "spatial_signed_error_metadata": "spatial_signed_error_map.json",
        },
    }
    config_path = os.path.join(config.OUTPUT_DIR, "config_used.json")
    with open(config_path, "w") as f:
        json.dump(config_used, f, indent=2)
    print(f"Saved run details to {config_path}")

    # --- Save ALL metrics to the results CSV ---
    results_df = pd.DataFrame(list(all_results.items()), columns=["Metric", "Value"])
    results_df.to_csv(os.path.join(config.OUTPUT_DIR, "results.csv"), index=False)
    print(f"\nSaved combined multi-experiment metrics to results.csv")

if __name__ == "__main__":
    main()
