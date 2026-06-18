import os
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Mapping of local model folder names to the display names
MODEL_MAP = {
    "pinn_baseline": "PINN",
    "dnn_baseline": "DNN",
    "chapman_richards": "CR",
    "avg_by_age": "AvgByAge",
}

def main():
    outputs_dir = os.path.join(os.path.dirname(__file__), '..')
    
    # Look for this specific plot in each model's directory
    target_filename = "spatial_signed_error_map.png"
    
    plot_data = []
    print(f"Looking for {target_filename} in model outputs...")
    
    for folder_name, display_name in MODEL_MAP.items():
        model_dir = os.path.join(outputs_dir, folder_name)
        plot_path = os.path.join(model_dir, target_filename)
        config_path = os.path.join(model_dir, "config_used.json")
        
        if os.path.exists(plot_path):
            print(f"  Found plot for {display_name}")
            
            # Default caption acknowledging it's the temporal experiment
            caption = "Temporal Experiment"
            
            # If a config exists (like for the PINN), grab details for the caption
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        cfg = json.load(f)
                    
                    cfg_details = []
                    if 'learning_rate' in cfg: cfg_details.append(f"LR: {cfg['learning_rate']}")
                    if 'epochs_run' in cfg or 'epochs_max' in cfg: 
                        cfg_details.append(f"Epochs: {cfg.get('epochs_run', cfg.get('epochs_max'))}")
                    if 'lambda_ph' in cfg: cfg_details.append(f"λ_ph: {cfg['lambda_ph']}")
                        
                    if cfg_details:
                        caption = f"Temporal Experiment | {', '.join(cfg_details)}"
                except Exception as e:
                    print(f"    Could not read config for {display_name}: {e}")
            
            plot_data.append({"model_name": display_name, "plot_path": plot_path, "caption": caption})
        else:
            print(f"  No {target_filename} found for {display_name}")

    if not plot_data:
        print(f"\nNo {target_filename} plots found to combine. Make sure the models have generated them!")
        return

    # Create a subplot grid (2 columns wide)
    n_plots = len(plot_data)
    cols = 2
    rows = (n_plots + 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows))
    fig.suptitle('Spatial Signed Error Comparison (Temporal Experiment)', fontsize=16, fontweight='bold', y=1.02)
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