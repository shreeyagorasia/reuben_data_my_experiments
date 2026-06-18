"""
Build the terrain/residual analysis table and numeric diagnostics.

Uses Table 4.1 2023 rows, adds CR and PINN predictions/residuals, then saves
correlations and binned summaries for DEM_ELEVATION and DEM_RUGGEDNESS.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp")
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import config as common_config  # noqa: E402
from common.data_utils import build_feature_arrays, load_data, split_age_column  # noqa: E402


PINN_DIR = PROJECT_ROOT / "models" / "pinn_baseline"
if str(PINN_DIR) not in sys.path:
    sys.path.insert(0, str(PINN_DIR))
import config as pinn_config  # noqa: E402
from model import PINN  # noqa: E402


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "analysis" / "terrain_exploration"
DEFAULT_INPUT = OUTPUT_DIR / "terrain_augmented_purged.csv"
DEFAULT_TABLE = OUTPUT_DIR / "terrain_residual_table.csv"
TERRAIN_COLS = ["DEM_ELEVATION", "DEM_RUGGEDNESS"]
CR_PARAMS = PROJECT_ROOT / "outputs" / "chapman_richards" / "cr_params.json"
PINN_CHECKPOINT = PROJECT_ROOT / "outputs" / "pinn_baseline" / "checkpoint.pt"


def parse_args():
    parser = argparse.ArgumentParser(description="Join terrain features to CR/PINN residual diagnostics.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="DEM-augmented purged CSV.")
    parser.add_argument("--output", default=DEFAULT_TABLE, help="Output terrain residual table.")
    return parser.parse_args()


def chapman_richards(t, y_max, k, p):
    return y_max * (1 - np.exp(-k * t)) ** p


def predict_cr(df23):
    with open(CR_PARAMS) as f:
        params = json.load(f)["Table4.1"]
    return chapman_richards(df23[common_config.AGE_COL].values, params["y_max"], params["k"], params["p"])


def predict_pinn(X_train, X_test):
    checkpoint = torch.load(PINN_CHECKPOINT, map_location="cpu", weights_only=False)
    saved_features = checkpoint.get("feature_list")
    if saved_features is not None and list(saved_features) != list(common_config.FEATURES):
        raise ValueError("PINN checkpoint feature list does not match common_config.FEATURES.")

    _, _, X_test_other, X_test_age, _, _ = split_age_column(X_train, X_test)
    model = PINN(n_other=len(checkpoint["other_idxs"]), hidden_size=pinn_config.HIDDEN_SIZE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    Xte_o = torch.tensor(checkpoint["scaler_Xo"].transform(X_test_other), dtype=torch.float32)
    Xte_a = torch.tensor(checkpoint["scaler_age"].transform(X_test_age), dtype=torch.float32)
    with torch.no_grad():
        y_pred_scaled = model(Xte_o, Xte_a).numpy().reshape(-1, 1)
    return checkpoint["scaler_y"].inverse_transform(y_pred_scaled).ravel()


def require_terrain_columns(path):
    if not Path(path).exists():
        raise SystemExit(
            f"Missing {path}.\n"
            "Run 01_build_terrain_features.py first, or pass --input to an existing DEM-augmented CSV."
        )
    header = pd.read_csv(path, nrows=0).columns
    missing = [col for col in TERRAIN_COLS if col not in header]
    if missing:
        raise SystemExit(
            f"Missing terrain columns {missing} in {path}.\n"
            "Run 01_build_terrain_features.py first, or pass --input to an existing DEM-augmented CSV."
        )


def assert_static_terrain(df):
    changed = {
        col: int((df.groupby("PLOT_ID")[col].nunique(dropna=False) > 1).sum())
        for col in TERRAIN_COLS
    }
    problems = {col: n for col, n in changed.items() if n}
    if problems:
        raise ValueError(f"Terrain columns should be static per plot but changed across years: {problems}")


def build_analysis_table(data_path):
    raw_df = pd.read_csv(data_path)
    assert_static_terrain(raw_df)
    df12, df23, _ = load_data(data_path)
    X_train, X_test, _, y_true, _, _ = build_feature_arrays(df12, df23)

    cr_pred = predict_cr(df23)
    pinn_pred = predict_pinn(X_train, X_test)
    table = df23.reset_index()[["PLOT_ID", "X", "Y", common_config.AGE_COL, common_config.TARGET_COL] + TERRAIN_COLS].copy()
    table = table.rename(columns={common_config.AGE_COL: "AGE_2023", common_config.TARGET_COL: "TOP_HEIGHT_2023"})
    table["CR_PRED_2023"] = cr_pred
    table["CR_RESIDUAL_2023"] = y_true - cr_pred
    table["CR_SIGNED_REL_ERROR"] = table["CR_RESIDUAL_2023"] / (np.abs(y_true) + 1e-8)
    table["CR_REL_ERROR"] = np.abs(table["CR_RESIDUAL_2023"]) / (np.abs(y_true) + 1e-8)
    table["PINN_PRED_2023"] = pinn_pred
    table["PINN_RESIDUAL_2023"] = y_true - pinn_pred
    table["PINN_SIGNED_REL_ERROR"] = table["PINN_RESIDUAL_2023"] / (np.abs(y_true) + 1e-8)
    table["PINN_REL_ERROR"] = np.abs(table["PINN_RESIDUAL_2023"]) / (np.abs(y_true) + 1e-8)
    return table


def correlation_summary(table):
    targets = [
        "CR_RESIDUAL_2023",
        "PINN_RESIDUAL_2023",
        "PINN_REL_ERROR",
        "CR_REL_ERROR",
        "X",
        "Y",
    ]
    rows = []
    for terrain_col in TERRAIN_COLS:
        for target in targets:
            valid = table[[terrain_col, target]].dropna()
            rows.append({
                "terrain_feature": terrain_col,
                "target": target,
                "n": int(len(valid)),
                "pearson_r": float(valid[terrain_col].corr(valid[target], method="pearson")),
                "spearman_r": float(valid[terrain_col].corr(valid[target], method="spearman")),
            })
    return pd.DataFrame(rows)


def binned_summary(table):
    data = table.copy()
    data["ELEVATION_BIN"] = pd.cut(
        data["DEM_ELEVATION"],
        bins=[0, 100, 200, 300, 400, np.inf],
        labels=["0-100m", "100-200m", "200-300m", "300-400m", "400m+"],
        include_lowest=True,
    )
    data["RUGGEDNESS_QUARTILE"] = pd.qcut(
        data["DEM_RUGGEDNESS"],
        q=4,
        labels=["Q1 lowest", "Q2", "Q3", "Q4 highest"],
        duplicates="drop",
    )
    rows = []
    for group_col in ["ELEVATION_BIN", "RUGGEDNESS_QUARTILE"]:
        for group, sub in data.groupby(group_col, observed=False):
            rows.append({
                "group_type": group_col,
                "group": str(group),
                "n": int(len(sub)),
                "cr_residual_mean": float(sub["CR_RESIDUAL_2023"].mean()),
                "pinn_residual_mean": float(sub["PINN_RESIDUAL_2023"].mean()),
                "pinn_rel_error_mean": float(sub["PINN_REL_ERROR"].mean()),
                "cr_rel_error_mean": float(sub["CR_REL_ERROR"].mean()),
            })
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    data_path = Path(args.input)
    output_path = Path(args.output)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    require_terrain_columns(data_path)
    table = build_analysis_table(data_path)
    table.to_csv(output_path, index=False)

    correlations = correlation_summary(table)
    correlations.to_csv(OUTPUT_DIR / "terrain_correlations.csv", index=False)
    with open(OUTPUT_DIR / "terrain_correlations.json", "w") as f:
        json.dump(correlations.to_dict(orient="records"), f, indent=2)

    binned = binned_summary(table)
    binned.to_csv(OUTPUT_DIR / "terrain_binned_summary.csv", index=False)
    with open(OUTPUT_DIR / "terrain_binned_summary.json", "w") as f:
        json.dump(binned.to_dict(orient="records"), f, indent=2)

    print(f"Saved terrain residual table to {output_path}")
    print(f"Saved correlations to {OUTPUT_DIR / 'terrain_correlations.csv'}")
    print(correlations.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    print(f"Saved binned summary to {OUTPUT_DIR / 'terrain_binned_summary.csv'}")


if __name__ == "__main__":
    main()
