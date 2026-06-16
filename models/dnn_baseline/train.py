"""
models/dnn_baseline/train.py
==============================
Trains the Deep Neural Network (DNN) baseline.
"""

import os
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
from plots import plot_training_curve, plot_spatial_error

def run_training(X_tr_full, y_tr_full, X_te_full, y_te_full, device, run_name=""):
    """Handles scaling, splitting, and training for a single run."""
    scaler_X = StandardScaler().fit(X_tr_full)
    scaler_y = StandardScaler().fit(y_tr_full.reshape(-1, 1))
    
    # Create an internal validation set for Early Stopping
    tr_idx, val_idx = train_val_split(len(X_tr_full), val_fraction=config.VAL_SPLIT)
    
    Xtr_t = torch.tensor(scaler_X.transform(X_tr_full[tr_idx]), dtype=torch.float32)
    ytr_t = torch.tensor(scaler_y.transform(y_tr_full[tr_idx].reshape(-1, 1)).ravel(), dtype=torch.float32)
    Xval_t = torch.tensor(scaler_X.transform(X_tr_full[val_idx]), dtype=torch.float32).to(device)
    yval_t = torch.tensor(scaler_y.transform(y_tr_full[val_idx].reshape(-1, 1)).ravel(), dtype=torch.float32).to(device)
    Xte = torch.tensor(scaler_X.transform(X_te_full), dtype=torch.float32).to(device)
    
    train_loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=config.BATCH_SIZE, shuffle=True)
    
    dnn = DNN(n_features=X_tr_full.shape[1]).to(device)
    optimiser = torch.optim.Adam(dnn.parameters(), lr=config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=0.8, patience=10)
    loss_fn = nn.MSELoss()
    
    train_hist, val_hist, epoch_log = [], [], []
    best_val = float('inf')
    patience_ctr = 0
    
    print(f"  {'Epoch':>4} | {'Train loss':>12} | {'Val loss':>12}")
    print("-" * 38)

    for epoch in range(config.EPOCHS):
        dnn.train()
        running, batch_count = 0.0, 0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimiser.zero_grad()
            pred = dnn(Xb)
            l1 = sum(p.abs().sum() for p in dnn.parameters())
            loss = loss_fn(pred, yb) + config.LAMBDA_L1 * l1
            loss.backward()
            optimiser.step()
            running += loss.item()
            batch_count += 1
            
        avg_train = running / batch_count
        dnn.eval()
        with torch.no_grad():
            val_loss = loss_fn(dnn(Xval_t), yval_t).item()
            
        scheduler.step(val_loss)
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= config.EARLY_STOP_PATIENCE:
                epoch_log.append(epoch + 1); train_hist.append(avg_train); val_hist.append(val_loss)
                print(f"    Early stopping at epoch {epoch + 1} - val_loss plateaued")
                break
                
        if (epoch + 1) % 50 == 0:
            epoch_log.append(epoch + 1); train_hist.append(avg_train); val_hist.append(val_loss)
            print(f"  {epoch + 1:>4}  |  {avg_train:>12.6f}  |  {val_loss:>12.6f}")

    if not epoch_log:
        epoch_log.append(epoch + 1); train_hist.append(avg_train); val_hist.append(val_loss)
        
    dnn.eval()
    with torch.no_grad():
        y_pred = scaler_y.inverse_transform(dnn(Xte).cpu().numpy().reshape(-1, 1)).ravel()
    
    # We only print the metrics output if it's the main temporal run
    if run_name:
        print(f"\n--- {run_name} ---")
    metrics = reuben_metrics(y_te_full, y_pred, label=run_name)
    
    return metrics, epoch_log, train_hist, val_hist, y_pred

def main():
    np.random.seed(config.RANDOM_SEED)
    torch.manual_seed(config.RANDOM_SEED)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # Auto-detect and use Apple Silicon (M1/M2) GPU if available
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    all_results = {}

    # --- Experiments using PURGED data (Tables 4.1, 4.3, 4.4) ---
    print("\n\n--- Loading PURGED data for Tables 4.1, 4.3, 4.4 ---")
    df12_purged, df23_purged, _ = load_data(config.DATA_PATH_PURGED)
    X_train_purged, X_test_purged, y_train_purged, y_test_purged, _, _ = build_feature_arrays(
        df12_purged, df23_purged
    )

    # --- Experiment for Table 4.1 (Temporal, Common Plots) ---
    print("\n--- Running Experiment for Table 4.1: Temporal (Common Plots) ---")
    metrics_t1, _, _, _, _ = run_training(
        X_train_purged, y_train_purged, X_test_purged, y_test_purged, device, "Temporal (Table 4.1)"
    )
    for k, v in metrics_t1.items():
        all_results[f"Table4.1_{k}"] = v

    # --- Experiment for Table 4.3 (3-Fold CV within 2012) ---
    print(f"\n--- Running Experiment for Table 4.3: 3-Fold CV within 2012 (Purged) ---")
    cv12_metrics = []
    for fold, (tr_idx, te_idx) in enumerate(get_kfold_splits(len(X_train_purged), n_splits=3)):
        print(f"  Training Fold {fold+1}/3...")
        m, _, _, _, _ = run_training(
            X_train_purged[tr_idx], y_train_purged[tr_idx], 
            X_train_purged[te_idx], y_train_purged[te_idx], device
        )
        cv12_metrics.append(m)
        
    for k in cv12_metrics[0].keys():
        all_results[f"Table4.3_{k}"] = float(np.mean([m[k] for m in cv12_metrics]))

    # --- Experiment for Table 4.4 (3-Fold CV within 2023) ---
    print(f"\n--- Running Experiment for Table 4.4: 3-Fold CV within 2023 (Purged) ---")
    cv23_metrics = []
    for fold, (tr_idx, te_idx) in enumerate(get_kfold_splits(len(X_test_purged), n_splits=3)):
        print(f"  Training Fold {fold+1}/3...")
        m, _, _, _, _ = run_training(
            X_test_purged[tr_idx], y_test_purged[tr_idx], 
            X_test_purged[te_idx], y_test_purged[te_idx], device
        )
        cv23_metrics.append(m)
        
    for k in cv23_metrics[0].keys():
        all_results[f"Table4.4_{k}"] = float(np.mean([m[k] for m in cv23_metrics]))

    # --- Experiment using UNSEEN data (Table 4.2) ---
    print("\n\n--- Loading UNSEEN data for Table 4.2 ---")
    df12_unseen, df23_unseen, _ = load_data(config.DATA_PATH_UNSEEN)
    X_train_unseen, X_test_unseen, y_train_unseen, y_test_unseen, X_coords_unseen, Y_coords_unseen = build_feature_arrays(
        df12_unseen, df23_unseen
    )
    print(f"\n--- Running Experiment for Table 4.2: Temporal (Unseen Plots) ---")
    metrics_t2, ep_log, tr_hist, val_hist, y_pred_unseen = run_training(
        X_train_unseen, y_train_unseen, X_test_unseen, y_test_unseen, device, "Temporal (Table 4.2)"
    )
    for k, v in metrics_t2.items():
        all_results[f"Table4.2_{k}"] = v
        
    # We only save plots for the "unseen" temporal experiment, as it's the most comprehensive
    plot_training_curve(ep_log, tr_hist, val_hist, config.OUTPUT_DIR)
    plot_spatial_error(X_coords_unseen, Y_coords_unseen, y_test_unseen, y_pred_unseen, 
                       "(f) DNN Temporal Error (Unseen)", config.OUTPUT_DIR)

    # --- Save ALL metrics to the results CSV ---
    results_df = pd.DataFrame(list(all_results.items()), columns=["Metric", "Value"])
    results_df.to_csv(os.path.join(config.OUTPUT_DIR, "results.csv"), index=False)
    print(f"\nSaved combined multi-experiment metrics to results.csv")

if __name__ == "__main__":
    main()
