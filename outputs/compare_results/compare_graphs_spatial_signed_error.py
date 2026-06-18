import os
import json
os.environ.setdefault("MPLCONFIGDIR", "/tmp")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Mapping of local model folder names to the display names
MODEL_MAP = {
    "pinn_baseline": "PINN",
    "dnn_baseline": "DNN",
    "chapman_richards": "CR",
    "avg_by_age": "AvgByAge",
}

EXPECTED_PLOT_TYPE = "signed_relative_error"
EXPECTED_TABLE = "Table4.1"
COMPARABLE_KEYS = [
    "plot_type",
    "table",
    "data_path_label",
    "n_test",
    "vmin",
    "vmax",
    "cmap",
    "reduce_C_function",
    "gridsize",
]


def metadata_path_for(plot_path):
    stem, _ = os.path.splitext(plot_path)
    return f"{stem}.json"


def load_plot_metadata(plot_path, display_name):
    metadata_path = metadata_path_for(plot_path)
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"Missing metadata for {display_name}: {metadata_path}\n"
            "Regenerate the model spatial plots before combining them."
        )
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    if metadata.get("plot_type") != EXPECTED_PLOT_TYPE:
        raise ValueError(f"{display_name} plot_type={metadata.get('plot_type')} but expected {EXPECTED_PLOT_TYPE}")
    if metadata.get("table") != EXPECTED_TABLE:
        raise ValueError(f"{display_name} table={metadata.get('table')} but expected {EXPECTED_TABLE}")
    return metadata


def assert_comparable(plot_data):
    if not plot_data:
        return
    reference = {key: plot_data[0]["metadata"].get(key) for key in COMPARABLE_KEYS}
    for data in plot_data[1:]:
        current = {key: data["metadata"].get(key) for key in COMPARABLE_KEYS}
        if current != reference:
            raise ValueError(
                "Spatial signed-error plots are not comparable.\n"
                f"Reference ({plot_data[0]['model_name']}): {reference}\n"
                f"Current ({data['model_name']}): {current}"
            )


def main():
    outputs_dir = os.path.join(os.path.dirname(__file__), '..')
    
    # Look for this specific plot in each model's directory
    target_filename = "spatial_signed_error_map.png"
    
    plot_data = []
    print(f"Looking for {target_filename} in model outputs...")
    
    for folder_name, display_name in MODEL_MAP.items():
        model_dir = os.path.join(outputs_dir, folder_name)
        plot_path = os.path.join(model_dir, target_filename)
        
        if os.path.exists(plot_path):
            print(f"  Found plot for {display_name}")
            metadata = load_plot_metadata(plot_path, display_name)
            
            caption = (
                f"{metadata['table']} | n={metadata['n_test']:,} | "
                f"mean signed rel={metadata['c_mean']:+.3f} | "
                f"vmin/vmax={metadata['vmin']}/{metadata['vmax']}"
            )
            
            plot_data.append({
                "model_name": display_name,
                "plot_path": plot_path,
                "caption": caption,
                "metadata": metadata,
            })
        else:
            raise FileNotFoundError(
                f"Missing {target_filename} for {display_name}: {plot_path}\n"
                "Regenerate all model plots before building the combined comparison figure."
            )

    if not plot_data:
        print(f"\nNo {target_filename} plots found to combine. Make sure the models have generated them!")
        return

    assert_comparable(plot_data)

    # Create a subplot grid (2 columns wide)
    n_plots = len(plot_data)
    cols = 2
    rows = (n_plots + 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows))
    fig.suptitle('Spatial Signed Relative Error Comparison (Temporal Experiment)', fontsize=16, fontweight='bold', y=1.07)
    fig.text(
        0.5,
        1.02,
        "Each panel is a spatial hexbin map: x-axis = Easting, y-axis = Northing (OS National Grid); colour shows mean signed relative error for plots in that area.",
        ha="center",
        va="top",
        fontsize=10,
        color="dimgray",
    )
    fig.text(
        0.5,
        0.99,
        "Signed relative error = (actual - predicted) / actual; negative = actual < predicted (overpredicted/reduced growth), positive = actual > predicted (underpredicted)",
        ha="center",
        va="top",
        fontsize=10,
        color="dimgray",
    )
    axes = axes.flatten()

    for i, data in enumerate(plot_data):
        ax = axes[i]
        ax.imshow(mpimg.imread(data["plot_path"]))
        ax.axis('off')
        ax.set_title(data["model_name"], fontsize=14, pad=10)
        # Add config caption below the plot
        ax.text(0.5, -0.05, data["caption"], size=10, ha="center", va="top", transform=ax.transAxes, color="dimgray", wrap=True)

    # Hide any unused subplots
    for j in range(len(plot_data), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    output_path = os.path.join(outputs_dir, "compare_results", "combined_spatial_signed_error.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"\nSaved combined comparison plot to: {output_path}")

if __name__ == "__main__":
    main()
