import os
import pandas as pd

# Reuben's original baseline metrics across all tables
REUBEN_METRICS = {
    "4.1": {
        "AvgByAge": {"MAE": 5.8121, "MSE": 48.9792, "R²": -0.1398, "MRE": 0.2170, "Acc%": 78.29},
        "LinReg":   {"MAE": 4.9311, "MSE": 35.4522, "R²": 0.1750, "MRE": 0.1911, "Acc%": 80.89},
        "CR":       {"MAE": 4.5592, "MSE": 31.0253, "R²": 0.2780, "MRE": 0.1844, "Acc%": 81.56},
        "RF":       {"MAE": 5.2237, "MSE": 41.0254, "R²": 0.0453, "MRE": 0.1959, "Acc%": 80.41},
        "DNN":      {"MAE": 4.4835, "MSE": 30.5735, "R²": 0.2885, "MRE": 0.1764, "Acc%": 82.36},
        "PINN":     {"MAE": 4.0424, "MSE": 24.1570, "R²": 0.4378, "MRE": 0.1539, "Acc%": 84.61},
    },
    "4.2": {
        "AvgByAge": {"MAE": 5.7346, "MSE": 47.6652, "R²": -0.0914, "MRE": 0.2185, "Acc%": 78.15},
        "LinReg":   {"MAE": 4.8489, "MSE": 36.0161, "R²": 0.0370, "MRE": 0.2130, "Acc%": 78.70},
        "CR":       {"MAE": 4.4814, "MSE": 30.9969, "R²": 0.4575, "MRE": 0.2021, "Acc%": 79.80},
        "RF":       {"MAE": 5.4347, "MSE": 45.3118, "R²": 0.2069, "MRE": 0.2318, "Acc%": 76.82},
        "DNN":      {"MAE": 4.4597, "MSE": 30.9261, "R²": 0.4587, "MRE": 0.1986, "Acc%": 80.14},
        "PINN":     {"MAE": 4.2188, "MSE": 26.9186, "R²": 0.5289, "MRE": 0.1842, "Acc%": 81.58},
    },
    "4.3": {
        "AvgByAge": {"MAE": 3.3536, "MSE": 17.9774, "R²": 0.6054, "MRE": 0.1854, "Acc%": 81.46},
        "LinReg":   {"MAE": 3.5407, "MSE": 20.0701, "R²": 0.5595, "MRE": 0.1926, "Acc%": 80.74},
        "RF":       {"MAE": 1.7297, "MSE": 4.9822, "R²": 0.8686, "MRE": 0.0904, "Acc%": 90.96},
        "DNN":      {"MAE": 2.6987, "MSE": 13.7242, "R²": 0.6987, "MRE": 0.1515, "Acc%": 84.85},
        "PINN":     {"MAE": 2.8134, "MSE": 13.3478, "R²": 0.7070, "MRE": 0.1484, "Acc%": 85.16},
    },
    "4.4": {
        "AvgByAge": {"MAE": 3.9888, "MSE": 17.9773, "R²": 0.4067, "MRE": 0.1671, "Acc%": 83.30},
        "LinReg":   {"MAE": 4.2716, "MSE": 28.8656, "R²": 0.3283, "MRE": 0.1789, "Acc%": 82.11},
        "RF":       {"MAE": 1.9305, "MSE": 7.0653, "R²": 0.8356, "MRE": 0.0762, "Acc%": 92.38},
        "DNN":      {"MAE": 3.2601, "MSE": 17.7137, "R²": 0.5877, "MRE": 0.1313, "Acc%": 86.87},
        "PINN":     {"MAE": 3.1629, "MSE": 17.0186, "R²": 0.6039, "MRE": 0.1285, "Acc%": 87.15},
    }
}

# Mapping of local model folder names to the names used in the tables
MODEL_MAP = {
    "pinn_baseline": "PINN",
    "dnn_baseline": "DNN",
    "chapman_richards": "CR",
    "avg_by_age": "AvgByAge",
}

# Mapping of CSV metric keys to display names
METRIC_MAP = {'mae': 'MAE', 'mse': 'MSE', 'r2': 'R²', 'mre': 'MRE', 'acc': 'Acc%'}

def main():
    # This script is in the 'outputs/compare_results' folder.
    # We need to go up one level to find the main 'outputs' folder where all the model results are.
    outputs_dir = os.path.join(os.path.dirname(__file__), '..')
    
    # We'll put all the results we find into this dictionary.
    # It will look like this: { '4.1': {'PINN': {'MAE': 4.04, ...}}, ... }
    all_my_results = {}
    
    # --- Step 1: Read all the results.csv files ---
    print("Looking for results...")
    # Get a list of all the folders inside the 'outputs' directory.
    for model_folder_name in os.listdir(outputs_dir):
        # We only want to look at folders that are for our models.
        if model_folder_name not in MODEL_MAP:
            continue

        print(f"  Found model folder: {model_folder_name}")
        # Build the full path to the results.csv file for this model.
        results_csv_path = os.path.join(outputs_dir, model_folder_name, "results.csv")

        # Check if the results file actually exists.
        if os.path.exists(results_csv_path):
            print(f"    Reading {results_csv_path}")
            # Use pandas to read the CSV file. It's like a spreadsheet in Python.
            df = pd.read_csv(results_csv_path)
            # Get the nice display name for our model, like "PINN".
            model_display_name = MODEL_MAP[model_folder_name]

            # Go through each row in the CSV file.
            for index, row in df.iterrows():
                metric_name = row['Metric'] # e.g., "Table4.1_mae"
                value = row['Value']        # e.g., 4.5183

                # The metric name has the table number in it, so we need to split it.
                if '_' in metric_name:
                    parts = metric_name.split('_')
                    table_id_str = parts[0]  # "Table4.1"
                    metric_key = parts[1]    # "mae"

                    # Make sure it's a table we care about.
                    if table_id_str.startswith("Table") and metric_key in METRIC_MAP:
                        # Get just the number part, like "4.1"
                        table_id = table_id_str.replace("Table", "")
                        # Get the nice display name for the metric, like "MAE"
                        display_metric = METRIC_MAP[metric_key]

                        # Now we add the result to our big dictionary.
                        # We have to make sure the nested dictionaries exist first.
                        if table_id not in all_my_results:
                            all_my_results[table_id] = {}
                        if model_display_name not in all_my_results[table_id]:
                            all_my_results[table_id][model_display_name] = {}
                        
                        all_my_results[table_id][model_display_name][display_metric] = value

    # --- Step 2: Build the output text ---
    # We will build up a big string with all our tables.
    final_output_string = ""
    final_output_string += "\n--- Model Comparison against Reuben's Original Metrics ---\n"

    # We want the columns in our tables to be in a specific order.
    cols_order = ["MAE", "MSE", "R²", "MRE", "Acc%"]
    # We also want to print the tables in order (4.1, 4.2, ...).
    table_ids_sorted = sorted(all_my_results.keys())

    # Loop through each table ID we found results for.
    for table_id in table_ids_sorted:
        final_output_string += f"\n\n========================= Table {table_id} =========================\n"

        # --- My Results Table ---
        # Create a pandas DataFrame from our results for this table.
        my_results_for_table = all_my_results[table_id]
        my_table_df = pd.DataFrame.from_dict(my_results_for_table, orient='index')
        my_table_df.index.name = "Model"

        # Make sure the columns are in the right order.
        cols_that_exist = []
        for col in cols_order:
            if col in my_table_df.columns:
                cols_that_exist.append(col)
        my_table_df = my_table_df[cols_that_exist]

        final_output_string += "\n>>> My Replication Results:\n"
        # Convert the DataFrame to a nicely formatted string.
        final_output_string += my_table_df.sort_index().to_string(float_format="%.4f")
        final_output_string += "\n"

        # --- Reuben's Results Table ---
        # Check if we have Reuben's original results for this table.
        if table_id in REUBEN_METRICS:
            final_output_string += f"\n>>> Reuben's Original Results (Table {table_id}):\n"
            
            # Create a DataFrame from Reuben's results.
            reuben_results_for_table = REUBEN_METRICS[table_id]
            reuben_table_df = pd.DataFrame.from_dict(reuben_results_for_table, orient='index')
            reuben_table_df.index.name = "Model"

            # Make sure the columns are in the right order.
            cols_that_exist_reuben = []
            for col in cols_order:
                if col in reuben_table_df.columns:
                    cols_that_exist_reuben.append(col)
            reuben_table_df = reuben_table_df[cols_that_exist_reuben]
            
            # Convert the DataFrame to a string.
            final_output_string += reuben_table_df.to_string(float_format="%.4f")
            final_output_string += "\n"

    # --- Step 3: Print to screen and save to a file ---
    print(final_output_string)

    # Define where we want to save the output file.
    summary_file_path = os.path.join(outputs_dir, "compare_results", "comparison_summary.txt")
    # Make sure the 'compare_results' folder exists.
    os.makedirs(os.path.dirname(summary_file_path), exist_ok=True)

    # Write the big string to the file.
    with open(summary_file_path, "w") as f:
        f.write(final_output_string)
    
    print(f"\n\nSaved comparison summary to: {summary_file_path}")

if __name__ == "__main__":
    main()
