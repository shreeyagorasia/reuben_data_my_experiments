"""
Quantify exploratory residual signal without training a new nonlinear model.

This script uses the existing CR and PINN residuals as targets and asks:

1. How much residual variance can simple feature groups explain linearly?
2. Do DEM features still explain residuals after coordinates are included?
3. Which individual features have the strongest univariate association?

This is diagnostic analysis, not a new prediction model.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data_exploration" / "residual_feature_importance_output"
DEFAULT_INPUT = OUTPUT_DIR / "residual_feature_table.csv"

FEATURE_GROUPS = {
    "coordinates": ["X", "Y"],
    "age": ["AGE"],
    "cultivation": ["CULTIVATN_MOUNDING", "CULTIVATN_NO CULTIVATION"],
    "land_use": [
        "PRILANDUSE_Dead High Forest",
        "PRILANDUSE_Failed",
        "PRILANDUSE_Felled",
        "PRILANDUSE_High Forest",
        "PRILANDUSE_Open",
        "PRILANDUSE_Open Water",
        "PRILANDUSE_Other Built Facility",
        "PRILANDUSE_Partially Intruded Broadleaf",
        "PRILANDUSE_Quarries",
        "PRILANDUSE_Residential",
        "PRILANDUSE_Unplantable or bare",
        "PRILANDUSE_Unplanted streamsides",
        "PRILANDUSE_Windblow",
    ],
    "dem": ["DEM_ELEVATION", "DEM_RUGGEDNESS"],
}

FEATURE_SETS = {
    "DEM only": FEATURE_GROUPS["dem"],
    "Coordinates only": FEATURE_GROUPS["coordinates"],
    "Existing PINN features": (
        FEATURE_GROUPS["coordinates"] + FEATURE_GROUPS["age"] + FEATURE_GROUPS["cultivation"] + FEATURE_GROUPS["land_use"]
    ),
    "Existing without coordinates": FEATURE_GROUPS["age"] + FEATURE_GROUPS["cultivation"] + FEATURE_GROUPS["land_use"],
    "Coordinates + DEM": FEATURE_GROUPS["coordinates"] + FEATURE_GROUPS["dem"],
    "All features + DEM": (
        FEATURE_GROUPS["coordinates"]
        + FEATURE_GROUPS["age"]
        + FEATURE_GROUPS["cultivation"]
        + FEATURE_GROUPS["land_use"]
        + FEATURE_GROUPS["dem"]
    ),
}

TARGETS = ["CR_RESIDUAL_2023", "PINN_RESIDUAL_2023"]


def parse_args():
    parser = argparse.ArgumentParser(description="Linear diagnostic feature-set analysis for CR/PINN residuals.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--ridge-alpha",
        type=float,
        default=1.0,
        help="Ridge regularization for multi-feature diagnostic fits. Use 0 for ordinary linear regression.",
    )
    return parser.parse_args()


def make_estimator(alpha):
    if alpha == 0:
        return make_pipeline(StandardScaler(), LinearRegression())
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))


def cv_feature_set_metrics(df, features, target, args):
    X = df[features].to_numpy()
    y = df[target].to_numpy()
    kf = KFold(n_splits=args.folds, shuffle=True, random_state=42)
    preds = np.empty_like(y, dtype=float)
    fold_rows = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
        estimator = make_estimator(args.ridge_alpha)
        estimator.fit(X[train_idx], y[train_idx])
        fold_pred = estimator.predict(X[test_idx])
        preds[test_idx] = fold_pred
        fold_rows.append({
            "target": target,
            "feature_set": None,
            "fold": fold,
            "r2": float(r2_score(y[test_idx], fold_pred)),
            "mae": float(mean_absolute_error(y[test_idx], fold_pred)),
            "rmse": float(mean_squared_error(y[test_idx], fold_pred) ** 0.5),
        })

    cv_r2 = float(r2_score(y, preds))
    overall = {
        "target": target,
        "feature_set": None,
        "n_features": len(features),
        "cv_r2": cv_r2,
        "cv_mae": float(mean_absolute_error(y, preds)),
        "cv_rmse": float(mean_squared_error(y, preds) ** 0.5),
        "unexplained_fraction": float(max(0.0, 1.0 - cv_r2)),
    }
    return overall, fold_rows


def feature_set_results(df, args):
    overall_rows = []
    fold_rows = []
    for target in TARGETS:
        for name, features in FEATURE_SETS.items():
            overall, folds = cv_feature_set_metrics(df, features, target, args)
            overall["feature_set"] = name
            for row in folds:
                row["feature_set"] = name
            overall_rows.append(overall)
            fold_rows.extend(folds)
    return pd.DataFrame(overall_rows), pd.DataFrame(fold_rows)


def univariate_associations(df):
    feature_cols = FEATURE_SETS["All features + DEM"]
    rows = []
    for target in TARGETS:
        for feature in feature_cols:
            valid = df[[feature, target]].dropna()
            if valid[feature].nunique() <= 1 or valid[target].nunique() <= 1:
                pearson = np.nan
                spearman = np.nan
            else:
                pearson = valid[feature].corr(valid[target], method="pearson")
                spearman = valid[feature].corr(valid[target], method="spearman")
            rows.append({
                "target": target,
                "feature": feature,
                "group": group_for_feature(feature),
                "pearson_r": float(pearson) if pd.notna(pearson) else np.nan,
                "spearman_r": float(spearman) if pd.notna(spearman) else np.nan,
                "abs_pearson_r": float(abs(pearson)) if pd.notna(pearson) else np.nan,
            })
    return pd.DataFrame(rows)


def coefficients_for_all_features(df, args):
    features = FEATURE_SETS["All features + DEM"]
    rows = []
    for target in TARGETS:
        X = df[features].to_numpy()
        y = df[target].to_numpy()
        estimator = make_estimator(args.ridge_alpha)
        estimator.fit(X, y)
        linear = estimator.named_steps["ridge"] if args.ridge_alpha != 0 else estimator.named_steps["linearregression"]
        for feature, coef in zip(features, linear.coef_):
            rows.append({
                "target": target,
                "feature": feature,
                "group": group_for_feature(feature),
                "standardized_coefficient": float(coef),
                "abs_standardized_coefficient": float(abs(coef)),
            })
    return pd.DataFrame(rows)


def group_for_feature(feature):
    for group, features in FEATURE_GROUPS.items():
        if feature in features:
            return group
    raise KeyError(feature)


def incremental_dem_summary(feature_sets):
    rows = []
    for target in TARGETS:
        target_rows = feature_sets.set_index("feature_set")
        dem_only = float(target_rows[(target_rows["target"] == target)].loc["DEM only", "cv_r2"])
        coords_only = float(target_rows[(target_rows["target"] == target)].loc["Coordinates only", "cv_r2"])
        coords_dem = float(target_rows[(target_rows["target"] == target)].loc["Coordinates + DEM", "cv_r2"])
        existing = float(target_rows[(target_rows["target"] == target)].loc["Existing PINN features", "cv_r2"])
        all_features = float(target_rows[(target_rows["target"] == target)].loc["All features + DEM", "cv_r2"])
        rows.append({
            "target": target,
            "dem_only_cv_r2": dem_only,
            "coordinates_only_cv_r2": coords_only,
            "dem_gain_over_coordinates": coords_dem - coords_only,
            "existing_pinn_features_cv_r2": existing,
            "dem_gain_over_existing_pinn_features": all_features - existing,
            "all_features_plus_dem_cv_r2": all_features,
            "unexplained_after_all_features_plus_dem": max(0.0, 1.0 - all_features),
        })
    return pd.DataFrame(rows)


def plot_feature_sets(feature_sets):
    for target in TARGETS:
        data = feature_sets[feature_sets["target"] == target].sort_values("cv_r2")
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.barh(data["feature_set"], data["cv_r2"], color="#4c78a8")
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("Cross-validated linear R2 on residual target")
        ax.set_title(f"Linear Residual Signal Explained: {target}")
        for i, value in enumerate(data["cv_r2"]):
            ax.text(value, i, f" {value:+.3f}", va="center", fontsize=8)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"linear_feature_set_cv_r2_{target.lower()}.png", dpi=250, bbox_inches="tight")
        plt.close()


def plot_top_associations(associations):
    for target in TARGETS:
        data = associations[associations["target"] == target].sort_values("abs_pearson_r").tail(12)
        fig, ax = plt.subplots(figsize=(8, 5.5))
        ax.barh(data["feature"], data["pearson_r"], color="#72b7b2")
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("Pearson r with residual target")
        ax.set_title(f"Strongest Univariate Associations: {target}")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"top_univariate_associations_{target.lower()}.png", dpi=250, bbox_inches="tight")
        plt.close()


def main():
    args = parse_args()
    input_path = Path(args.input)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not input_path.exists():
        raise SystemExit(f"Missing {input_path}. Run 01_prepare_residual_feature_table.py first.")

    df = pd.read_csv(input_path)
    feature_sets, fold_metrics = feature_set_results(df, args)
    associations = univariate_associations(df)
    coefficients = coefficients_for_all_features(df, args)
    incremental = incremental_dem_summary(feature_sets)

    feature_sets.to_csv(OUTPUT_DIR / "linear_feature_set_cv_metrics.csv", index=False)
    fold_metrics.to_csv(OUTPUT_DIR / "linear_feature_set_fold_metrics.csv", index=False)
    associations.to_csv(OUTPUT_DIR / "univariate_residual_associations.csv", index=False)
    coefficients.to_csv(OUTPUT_DIR / "linear_standardized_coefficients.csv", index=False)
    incremental.to_csv(OUTPUT_DIR / "incremental_dem_summary.csv", index=False)

    plot_feature_sets(feature_sets)
    plot_top_associations(associations)

    summary = {
        "purpose": "Exploratory linear residual signal analysis, not a new prediction model.",
        "targets": TARGETS,
        "ridge_alpha": args.ridge_alpha,
        "feature_set_cv_metrics": feature_sets.to_dict(orient="records"),
        "incremental_dem_summary": incremental.to_dict(orient="records"),
    }
    with open(OUTPUT_DIR / "linear_residual_signal_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved linear residual-signal outputs to {OUTPUT_DIR}")
    print("\nFeature-set CV metrics:")
    print(feature_sets.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    print("\nIncremental DEM summary:")
    print(incremental.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))


if __name__ == "__main__":
    main()
