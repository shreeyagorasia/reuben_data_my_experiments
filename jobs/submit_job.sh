#!/bin/bash

#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:1
#SBATCH --mem=12G

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
echo "Starting training for ${MODEL_NAME} from directory: $(pwd)"
cd "models/${MODEL_NAME}"
python train.py