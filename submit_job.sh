#!/bin/bash

#SBATCH --job-name=growth_model
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:1
#SBATCH --mem=12G

# ------------------------------------------------------------------
# submit_job.sh
# ------------------------------------------------------------------
# Generic SLURM submission script that trains ONE model.
#
# Usage:
#   sbatch submit_job.sh chapman_richards
#   sbatch submit_job.sh pinn_baseline
#   sbatch submit_job.sh <your_new_model_name>
#
# Note: chapman_richards must be run BEFORE pinn_baseline (and any
# other model that uses the CR curve as a physics prior), since those
# models load outputs/chapman_richards/cr_params.json.
#
# Adjust the #SBATCH options and module/environment lines below for
# your specific cluster.
# ------------------------------------------------------------------
# --- environment setup (edit for your cluster) ---------------------
# 1. Load the Edinburgh GPU toolchain (Crucial for the cluster)
. /home/htang2/toolchain-20251006/toolchain.rc

# 2. Activate your specific venv
source ./venv/bin/activate

# Ensure we run from the project root, no matter where the script is located
cd "${SLURM_SUBMIT_DIR}"

# Tell Python where the root folder is so it can find the 'common' module
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH}"

MODEL_NAME=$1

if [ -z "$MODEL_NAME" ]; then
    echo "Usage: sbatch submit_job.sh <model_name>"
    echo "e.g.:  sbatch submit_job.sh chapman_richards"
    exit 1
fi

# --- make sure output/log folders exist ------------------------------
mkdir -p logs outputs

# --- run the chosen model's training script --------------------------
cd "models/${MODEL_NAME}"
python train.py
