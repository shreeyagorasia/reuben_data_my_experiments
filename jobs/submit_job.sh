#!/bin/bash

#SBATCH --job-name=growth_model
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:1
#SBATCH --mem=8G


# --- How to run ---------------------------------------------------
# sbatch --job-name=chapman_richards jobs/submit_job.sh chapman_richards

echo "--- SLURM JOB START ---"
echo "Model requested: $1"
echo "Current directory is: $(pwd)"

# --- environment setup (edit for your cluster) ---------------------
# 1. Load the Edinburgh GPU toolchain (Crucial for the cluster)
. /home/htang2/toolchain-20251006/toolchain.rc

# 2. Activate your specific venv
source ./venv/bin/activate

# SLURM starts in the folder you submitted from. 
# Tell Python that this current folder is the root so it can find 'common/'
export PYTHONPATH="$(pwd)"

MODEL_NAME=$1

if [ -z "$MODEL_NAME" ]; then
    echo "Usage: sbatch jobs/submit_job.sh <model_name>"
    echo "e.g.:  sbatch jobs/submit_job.sh chapman_richards"
    exit 1
fi

# --- make sure output/log folders exist ------------------------------
mkdir -p logs outputs

# --- run the chosen model's training script --------------------------
echo "Navigating to models/${MODEL_NAME}..."

if [ ! -d "models/${MODEL_NAME}" ]; then
    echo "Error: Directory models/${MODEL_NAME} does not exist."
    exit 1
fi
cd "models/${MODEL_NAME}" || exit 1

echo "Running train.py..."
python -u train.py
echo "--- SLURM JOB END ---"