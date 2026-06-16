"""
common/metrics.py
==================
A single function for evaluating model predictions, using the same
metric definitions as Reuben's dissertation. Used by every model so
results are directly comparable.
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def reuben_metrics(y_true, y_pred, label=""):
    """Print and return MAE, MSE, R^2, MRE and "accuracy".

    MRE (mean relative error) = mean( |actual - predicted| / actual )   [Eq. 4.1]
    Acc = (1 - MRE) * 100
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Small epsilon (1e-8) avoids division by zero if a true value is 0.
    mre = np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + 1e-8))
    acc = (1 - mre) * 100

    print(f"  {label:<28} MAE={mae:.4f}  MSE={mse:.4f}  R²={r2:.4f}  "
          f"MRE={mre:.4f}  Acc={acc:.2f}%")

    return {"mae": float(mae), "mse": float(mse), "r2": float(r2),
            "mre": float(mre), "acc": float(acc)}
