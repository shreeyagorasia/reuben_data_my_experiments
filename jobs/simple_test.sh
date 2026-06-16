#!/bin/bash

#SBATCH --output=logs/simple_%j.out
#SBATCH --error=logs/simple_%j.err
#SBATCH --time=00:05:00
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:1
#SBATCH --mem=4G

echo "--- SLURM TEST JOB START ---"
echo "Current directory is: $(pwd)"

echo "Loading environment..."
. /home/htang2/toolchain-20251006/toolchain.rc
source ./venv/bin/activate

echo "Python version is:"
python --version
echo "--- SLURM TEST JOB END ---"