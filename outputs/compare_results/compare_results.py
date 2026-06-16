import os
import csv
import pandas as pd
from pathlib import Path

# Reuben's original hardcoded baseline metrics
REUBEN_METRICS = {
    "PINN": {"MAE": 4.2188, "MSE": 26.9186, "R2": 0.5289, "MRE": 0.1842, "Acc": 81.58},
    "DNN": {"MAE": 4.4597, "MSE": 30.9261, "R2": 0.4587, "MRE": 0.1986, "Acc": 80.14},
    "CR": {"MAE": 4.4814, "MSE": 30.9969, "R2": 0.4575, "MRE": 0.2021, "Acc": 79.80},
    "AvgByAge": {"MAE": 5.7346, "MSE": 47.6652, "R2": -0.09139, "MRE": 0.21845, "Acc": 78.15},
}

# Mapping of local model names to the names used in Reuben's table
MODEL_MAP = {
    "pinn_baseline": "PINN",
    "dnn_baseline": "DNN",
    "chapman_richards": "CR",
    "avg_by_age": "AvgByAge",
}

def main():
    project_root = str(Path(__file__).resolve().parents[2])
    outputs_dir = project_root / "outputs"
    comparison_dir = outputs_dir / "compare_results"
    os.makedirs(comparison_dir, exist_ok=True)
    
    results_list = []
    
    # Iterate through all available model folders in outputs/
    for model_folder in outputs_dir.iterdir():
        if not model_folder.is_dir() or model_folder.name == "compare_results":
            continue
            
        results_csv = model_folder / "results.csv"
        if results_csv.exists():
            local_metrics = {}
            with open(results_csv, "r") as f:
                reader = csv.reader(f)
                next(reader) # skip header
                for row in reader:
                    local_metrics[row[0].lower()] = float(row[1])
                    
            model_display_name = MODEL_MAP.get(model_folder.name, model_folder.name)
            reuben = REUBEN_METRICS.get(model_display_name, {})
            
            results_list.append({
                "Model": model_display_name,
                "MAE_Mine": local_metrics.get("mae", None),
                "MAE_Reuben": reuben.get("MAE", None),
                "MSE_Mine": local_metrics.get("mse", None),
                "MSE_Reuben": reuben.get("MSE", None),
                "R2_Mine": local_metrics.get("r2", None),
                "R2_Reuben": reuben.get("R2", None),
                "Acc_Mine": local_metrics.get("acc", None),
                "Acc_Reuben": reuben.get("Acc", None),
            })
            
    df = pd.DataFrame(results_list)
    if not df.empty:
        df = df.sort_values(by="MAE_Mine")
    
    csv_path = comparison_dir / "metrics_comparison.csv"
    df.to_csv(csv_path, index=False)
    
    print("\n--- Model Comparison against Reuben's Original Metrics ---")
    print(df.to_string(index=False))
    print(f"\nSaved comparison to {csv_path}")

if __name__ == "__main__":
    main()
