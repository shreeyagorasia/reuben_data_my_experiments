"""
Prepare a feature matrix for exploratory residual-importance analysis.

Joins the terrain residual table from analysis/terrain_exploration to the
2023 rows of the DEM-augmented Table 4.1 CSV so existing predictors and DEM
features can be tested against CR/PINN residuals.
"""

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TERRAIN_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "analysis" / "terrain_exploration"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "analysis" / "residual_feature_importance"
DEFAULT_TERRAIN_CSV = TERRAIN_OUTPUT_DIR / "terrain_augmented_purged.csv"
DEFAULT_RESIDUAL_CSV = TERRAIN_OUTPUT_DIR / "terrain_residual_table.csv"
DEFAULT_OUTPUT = OUTPUT_DIR / "residual_feature_table.csv"

BASE_FEATURES = [
    "X",
    "Y",
    "AGE",
    "CULTIVATN_MOUNDING",
    "CULTIVATN_NO CULTIVATION",
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
]
DEM_FEATURES = ["DEM_ELEVATION", "DEM_RUGGEDNESS"]
TARGETS = [
    "CR_RESIDUAL_2023",
    "PINN_RESIDUAL_2023",
    "CR_REL_ERROR",
    "PINN_REL_ERROR",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Build residual feature table for exploratory importance analysis.")
    parser.add_argument("--terrain-csv", default=DEFAULT_TERRAIN_CSV)
    parser.add_argument("--residual-csv", default=DEFAULT_RESIDUAL_CSV)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main():
    args = parse_args()
    terrain_csv = Path(args.terrain_csv)
    residual_csv = Path(args.residual_csv)
    output = Path(args.output)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not terrain_csv.exists():
        raise SystemExit(f"Missing {terrain_csv}. Run analysis/terrain_exploration/01_build_terrain_features.py first.")
    if not residual_csv.exists():
        raise SystemExit(f"Missing {residual_csv}. Run analysis/terrain_exploration/02_residual_diagnostics.py first.")

    terrain = pd.read_csv(terrain_csv)
    residuals = pd.read_csv(residual_csv)
    rows_2023 = terrain[terrain["YEAR"] == 2023].copy()

    required_features = BASE_FEATURES + DEM_FEATURES
    missing = [col for col in required_features if col not in rows_2023.columns]
    if missing:
        raise SystemExit(f"Missing feature columns in {terrain_csv}: {missing}")

    residual_cols = ["PLOT_ID"] + TARGETS
    missing_targets = [col for col in residual_cols if col not in residuals.columns]
    if missing_targets:
        raise SystemExit(f"Missing residual columns in {residual_csv}: {missing_targets}")

    table = rows_2023[["PLOT_ID"] + required_features].merge(
        residuals[residual_cols],
        on="PLOT_ID",
        how="inner",
        validate="one_to_one",
    )
    table.to_csv(output, index=False)
    print(f"Saved residual feature table to {output}")
    print(f"Rows: {len(table):,}; features: {len(required_features)}; targets: {TARGETS}")


if __name__ == "__main__":
    main()

