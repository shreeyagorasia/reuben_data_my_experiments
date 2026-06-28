"""
Build a DEM-augmented Table 4.1 CSV.

Samples Copernicus GLO-30 elevation once per PLOT_ID using the 2012 plot
coordinate, derives a simple 3x3 local ruggedness value, then joins those
static terrain values back onto both 2012 and 2023 rows.
"""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "Aber_1223_conjoin_plot_purged_expanded_encoded.csv"
OUTPUT_DIR = PROJECT_ROOT / "data_exploration" / "terrain_exploration_output"
DEFAULT_OUTPUT = OUTPUT_DIR / "terrain_augmented_purged.csv"
DEFAULT_TILE_DIR = OUTPUT_DIR / "dem_tiles"
BASE_URL = "https://copernicus-dem-30m.s3.amazonaws.com"


def parse_args():
    parser = argparse.ArgumentParser(description="Sample Copernicus GLO-30 DEM features at plot locations.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input 2012/2023 CSV without or with DEM columns.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output CSV with DEM_ELEVATION and DEM_RUGGEDNESS.")
    parser.add_argument("--tile-dir", default=DEFAULT_TILE_DIR, help="Where downloaded DEM tiles are cached.")
    parser.add_argument("--buffer-deg", type=float, default=0.01, help="AOI padding in degrees around plot extent.")
    return parser.parse_args()


def require_geo_deps():
    try:
        import rasterio
        import rasterio.transform
        import rasterio.windows
        from pyproj import Transformer
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing geospatial dependency. Install these in the conda env first:\n"
            "  conda run -n diss_experiments python -m pip install rasterio pyproj\n"
            f"\nOriginal import error: {exc}"
        ) from None
    return rasterio, Transformer


def tile_name(lat_val, lon_val):
    lat_floor = math.floor(lat_val)
    lon_floor = math.floor(lon_val)
    ns = "N" if lat_floor >= 0 else "S"
    ew = "E" if lon_floor >= 0 else "W"
    return f"Copernicus_DSM_COG_10_{ns}{abs(lat_floor):02d}_00_{ew}{abs(lon_floor):03d}_00_DEM"


def open_tile(tile, tile_dir, rasterio):
    url_stream = f"/vsicurl/{BASE_URL}/{tile}/{tile}.tif"
    try:
        src = rasterio.open(url_stream)
        _ = src.profile
        return src
    except Exception:
        import requests

        tile_dir.mkdir(parents=True, exist_ok=True)
        local_path = tile_dir / f"{tile}.tif"
        if not local_path.exists():
            url = f"{BASE_URL}/{tile}/{tile}.tif"
            with requests.get(url, stream=True, timeout=120) as response:
                response.raise_for_status()
                with open(local_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1 << 20):
                        if chunk:
                            f.write(chunk)
        return rasterio.open(local_path)


def anchor_plots(df, transformer):
    anchor = (
        df[df["YEAR"] == 2012]
        .drop_duplicates(subset="PLOT_ID")[["PLOT_ID", "X", "Y"]]
        .reset_index(drop=True)
    )
    lon, lat = transformer.transform(anchor["X"].values, anchor["Y"].values)
    anchor["LON"] = lon
    anchor["LAT"] = lat
    anchor["TILE"] = [tile_name(la, lo) for la, lo in zip(lat, lon)]
    return anchor


def sample_tile(anchor, tile, buffer_deg, tile_dir, rasterio):
    mask = (anchor["TILE"] == tile).values
    sub_lon = anchor.loc[mask, "LON"].values
    sub_lat = anchor.loc[mask, "LAT"].values

    print(f"Reading AOI for {tile} covering {mask.sum():,} plots ...")
    with open_tile(tile, tile_dir, rasterio) as src:
        west, east = sub_lon.min() - buffer_deg, sub_lon.max() + buffer_deg
        south, north = sub_lat.min() - buffer_deg, sub_lat.max() + buffer_deg
        window = rasterio.windows.from_bounds(west, south, east, north, src.transform)
        window = window.round_offsets(op="floor").round_lengths(op="ceil")
        arr = src.read(1, window=window).astype("float64")
        win_transform = rasterio.windows.transform(window, src.transform)
        nodata = src.nodata

    if nodata is not None:
        arr[arr == nodata] = np.nan

    filled = np.nan_to_num(arr, nan=np.nanmean(arr))
    mean = uniform_filter(filled, size=3, mode="nearest")
    mean_sq = uniform_filter(filled ** 2, size=3, mode="nearest")
    local_std = np.sqrt(np.clip(mean_sq - mean ** 2, 0, None))

    rows, cols = rasterio.transform.rowcol(win_transform, sub_lon, sub_lat)
    rows = np.clip(np.asarray(rows), 0, arr.shape[0] - 1)
    cols = np.clip(np.asarray(cols), 0, arr.shape[1] - 1)
    return mask, arr[rows, cols], local_std[rows, cols]


def main():
    args = parse_args()
    rasterio, Transformer = require_geo_deps()

    input_path = Path(args.input)
    output_path = Path(args.output)
    tile_dir = Path(args.tile_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    transformer = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
    anchor = anchor_plots(df, transformer)

    print(f"Anchored {len(anchor):,} plots to 2012 coordinates")
    print(f"Lon: {anchor['LON'].min():.4f} to {anchor['LON'].max():.4f}")
    print(f"Lat: {anchor['LAT'].min():.4f} to {anchor['LAT'].max():.4f}")
    print(f"Tile(s): {sorted(anchor['TILE'].unique())}")

    elevations = np.full(len(anchor), np.nan)
    ruggedness = np.full(len(anchor), np.nan)
    for tile in sorted(anchor["TILE"].unique()):
        mask, tile_elev, tile_rugged = sample_tile(anchor, tile, args.buffer_deg, tile_dir, rasterio)
        elevations[mask] = tile_elev
        ruggedness[mask] = tile_rugged

    anchor["DEM_ELEVATION"] = elevations
    anchor["DEM_RUGGEDNESS"] = ruggedness
    terrain_cols = ["DEM_ELEVATION", "DEM_RUGGEDNESS"]
    augmented = df.drop(columns=[col for col in terrain_cols if col in df.columns]).merge(
        anchor[["PLOT_ID"] + terrain_cols],
        on="PLOT_ID",
        how="left",
    )

    augmented.to_csv(output_path, index=False)
    static_check = augmented.groupby("PLOT_ID")["DEM_ELEVATION"].nunique(dropna=False)
    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "n_rows": int(len(augmented)),
        "n_plots": int(augmented["PLOT_ID"].nunique()),
        "missing_elevation": int(augmented["DEM_ELEVATION"].isna().sum()),
        "plots_with_multiple_elevations": int((static_check > 1).sum()),
        "tiles": sorted(anchor["TILE"].unique()),
        "terrain_summary": augmented[terrain_cols].describe().to_dict(),
    }
    with open(output_path.parent / "terrain_feature_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved DEM-augmented CSV to {output_path}")
    print(f"Missing elevation values: {summary['missing_elevation']} / {summary['n_rows']}")
    print(f"Plots with >1 distinct DEM_ELEVATION across years: {summary['plots_with_multiple_elevations']}")
    print(augmented[terrain_cols].describe().to_string())


if __name__ == "__main__":
    main()
