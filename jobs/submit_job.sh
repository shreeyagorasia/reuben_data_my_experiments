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

# Ensure we run from the project root, no matter where the script is located
cd "${SLURM_SUBMIT_DIR}"

# Tell Python where the root folder is so it can find the 'common' module
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH}"

MODEL_NAME=$1
