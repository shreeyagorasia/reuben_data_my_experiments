# Reuben data — height growth models

A plain Python conversion of the original Colab notebook
(`Rueben_data_my_extension.ipynb`), restructured so new models can be
added easily and run as SLURM jobs on a cluster.

## What it does

Uses 2012 plot data (features + tree height) to predict 2023 tree
height (`TOP_HEIGHT`), for plots that exist in both years. Currently
includes two models:

1. **chapman_richards** — a classic growth-curve baseline,
   `H(t) = y_max * (1 - exp(-k*t))^p`, fitted with `scipy.curve_fit`.
2. **pinn_baseline** — a small feed-forward neural network whose loss
   combines a normal prediction error with a "physics" term that
   pulls its predicted growth rate (`d(height)/d(age)`) towards the
   Chapman-Richards curve's growth rate.

## Directory structure

```
Reuben_data_my_experiments/
├── common/                  # code shared by every model
│   ├── config.py             # DATA_PATH, FEATURES, TARGET_COL, AGE_COL, RANDOM_SEED
│   ├── data_utils.py          # loading CSV, building arrays, scaling, splitting
│   └── metrics.py              # reuben_metrics()
│
├── data/
│   └── (put your CSV here)
│
├── models/
│   ├── chapman_richards/
│   │   ├── config.py          # model-specific settings + output path
│   │   ├── model.py            # chapman_richards()
│   │   └── train.py             # fit + evaluate + save
│   │
│   └── pinn_baseline/
│       ├── config.py            # hyperparameters + output path
│       ├── model.py              # PINN class, physics loss, cr_derivative
│       ├── train.py               # training script
│       └── evaluate.py             # reload checkpoint, re-evaluate
│
├── outputs/                  # created automatically, one folder per model
│   ├── chapman_richards/
│   │   ├── cr_params.json     # fitted y_max, k, p (used as PINN's physics prior)
│   │   ├── results.json        # metrics
│   │   └── cr_fit.png
│   │
│   └── pinn_baseline/
│       ├── checkpoint.pt        # model weights + scalers + history
│       ├── config_used.json      # snapshot of hyperparameters for this run
│       ├── results.json           # metrics
│       └── training_curve.png
│
├── logs/                     # SLURM stdout/stderr
├── submit_job.sh             # sbatch submit_job.sh <model_name>
├── requirements.txt
└── README.md
```

## Running it

```bash
pip install -r requirements.txt

# 1. Edit common/config.py and set DATA_PATH (or just drop your CSV
#    into data/ with the expected filename).

# 2. Run the Chapman-Richards baseline FIRST -- the PINN depends on its output.
cd models/chapman_richards
python train.py
cd ../..

# 3. Train the PINN baseline.
cd models/pinn_baseline
python train.py
python evaluate.py   # optional: re-check the saved checkpoint
cd ../..
```

On a cluster:
```bash
sbatch submit_job.sh chapman_richards
sbatch submit_job.sh pinn_baseline     # after chapman_richards has finished
```

## Adding a new model

1. Copy `models/pinn_baseline/` to `models/<your_model_name>/`.
2. In the new `config.py`, change `MODEL_NAME = "<your_model_name>"`
   -- its outputs will then automatically go to
   `outputs/<your_model_name>/`. Add/remove hyperparameters as needed.
3. Edit `model.py` for the new architecture/equations.
4. Edit `train.py` (and `evaluate.py`) as needed -- it's fine for this
   to look similar to the PINN's; keeping each model's training script
   self-contained makes it easier to read on its own, even if there's
   some duplication between models.
5. Run with `sbatch submit_job.sh <your_model_name>`.

Anything genuinely shared across ALL models (data loading, metrics)
belongs in `common/`. Anything that varies between models
(architecture, hyperparameters, loss function) belongs in that
model's own folder.

## Notes on the data path

The original notebook mounted Google Drive and read the CSV from:

```
/content/drive/My Drive/Colab Notebooks/Msc_diss/Reubens data/Aber_1223_conjoin_plot_purged_expanded_encoded.csv
```

On the cluster, copy/upload this CSV into `data/` (or wherever you
like, updating `DATA_PATH` in `common/config.py` accordingly).
