#!/bin/bash

#SBATCH --job-name=pinn_table
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:1
#SBATCH --mem=8G

# --- How to run ---------------------------------------------------
# sbatch --job-name=pinn jobs/submit_job.sh 

# === SET THE TABLE YOU WANT TO REPLICATE HERE ===
# Options: "4.1", "4.2", "4.3", "4.4", or "all"
TABLE="4.1"
# ================================================

echo "--- SLURM JOB START ---"
echo "Running PINN Baseline for Table: $TABLE"
echo "Current directory is: $(pwd)"

# --- environment setup ---------------------------------------------
. /home/htang2/toolchain-20251006/toolchain.rc
source ./venv/bin/activate
export PYTHONPATH="$(pwd)"

mkdir -p logs outputs

echo "Navigating to models/pinn_baseline..."
cd "models/pinn_baseline" || exit 1

echo "Running train.py with --table $TABLE..."
python -u train.py --table "$TABLE"

echo "--- SLURM JOB END ---"