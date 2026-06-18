# Codebase Audit

Audit date: 2026-06-18

Current status: **AMBER until outputs are regenerated**. The framework is usable as a comparison of intentionally different model families, not as a same-feature architecture-only ablation. The current data populations are consistent across models within each table, Table 4.2's full unseen-CSV protocol is intentional, and spatial plotting has been centralized with metadata checks. The remaining practical requirement is to rerun/regenerate outputs so `results.csv`, map PNGs, map JSON sidecars, and configs on disk match the corrected code.

## Executive Summary

The strict claim "performance differences are due solely to model architecture" is not the right framing for this codebase. A more accurate claim is:

> Within each table, the scripts evaluate each model on the same target rows and metric definitions, but the comparison is between intentionally different modelling pipelines: age-only empirical/parametric baselines versus full-feature neural models.

Key current conclusions:

- Table 4.1 is the primary temporal common-plot experiment: 27,660 2012 training rows and 27,660 paired 2023 test rows.
- Table 4.2 is intentionally the full unseen-CSV temporal experiment: 31,147 post-purge 2012 training rows and 41,492 post-purge 2023 test rows, including 13,832 2023-only plots in the test set.
- AvgByAge and Chapman-Richards intentionally use `AGE` only; DNN and PINN intentionally use the full 18-feature set.
- PINN's Chapman-Richards prior is an intentional fixed-prior design choice, not a current blocker.
- Spatial plotting now delegates to `common/spatial_plots.py`; unsigned maps use capped relative error and signed maps use signed relative error.
- Combined spatial comparison scripts now require sidecar JSON metadata and refuse mixed table/data/scale artifacts.
- Existing plot PNGs may be stale until regenerated with the current code.

Minimum remaining actions before final reporting:

1. Regenerate Table 4.1 spatial maps and metadata for all four models.
2. Run the combined spatial comparison scripts only after metadata checks pass.
3. Document Table 4.2's intended full unseen-CSV protocol in the dissertation methods.
4. Document feature/training differences as intentional differences between model families.
5. Update stale README text before final submission.

## Detailed Forensic Audit

I did two passes over the repository. Pass 1 read the shared configuration/utilities, every model folder, the job scripts, the comparison script, README, requirements, and saved text/CSV outputs. Pass 2 re-read the implementation with targeted checks for cross-file inconsistencies: data paths, Table 4.2 handling, common-plot logic, feature lists, scalers, metric formulas, CR parameters, checkpoint behavior, and silent filtering. Binary artifacts (`.pt`, `.png`) were not semantically inspected beyond their presence and surrounding code paths.

Important environment note: the local `python3` in this shell does not have `pandas`, `scipy`, `sklearn`, or `torch` installed, and there is no local `venv/` in the repo. I therefore computed dataset counts directly from the CSVs with Python's standard `csv` module and used existing saved output CSV/JSON files where dependency-based model reruns were not possible.

## 1. Data Handling

### Shared data paths

`common/config.py` defines two CSV paths:

- `DATA_PATH_PURGED`: `data/Aber_1223_conjoin_plot_purged_expanded_encoded.csv` (`common/config.py:21-23`)
- `DATA_PATH_UNSEEN`: `data/Aber_1223_conjoin_expanded_encoded_duplicate_plots_removed.csv` (`common/config.py:24-26`)

There is no `DATA_PATH` variable, but `common/data_utils.py:31` still defaults to `config.DATA_PATH` when `load_data()` is called without an explicit path. Every current training/evaluation script passes either `DATA_PATH_PURGED` or `DATA_PATH_UNSEEN`, so the default is not hit in normal runs, but the default path is broken.

### `load_data()` behavior

`load_data(path)`:

1. Reads the CSV with `pd.read_csv(path)` (`common/data_utils.py:32`).
2. Splits rows by `YEAR == 2012` and `YEAR == 2023`, then indexes each dataframe by `PLOT_ID` (`common/data_utils.py:34-35`).
3. Computes `common_ids_raw = df12.index.intersection(df23.index)` (`common/data_utils.py:43`).
4. Removes common plots where `TOP_HEIGHT_2023 < TOP_HEIGHT_2012` (`common/data_utils.py:44-52`).
5. Computes sorted `common_ids` after that purge (`common/data_utils.py:55`).
6. Returns `df12, df23, common_ids` (`common/data_utils.py:61`).

The critical detail: it does **not** subset `df12` or `df23` to `common_ids`. It only drops negative-growth IDs from both years. On the purged CSV this makes no practical difference because both years contain exactly the same 27,660 plots. On the unseen CSV it makes a major difference because many 2012-only and 2023-only plots remain in the returned frames.

### Table 4.1: Temporal, common plots

CSV loaded: `DATA_PATH_PURGED`.

Code paths:

- AvgByAge: `models/avg_by_age/train.py:49-54`
- Chapman-Richards: `models/chapman_richards/train.py:62-67`
- DNN: `models/dnn_baseline/train.py:133-143`
- PINN: `models/pinn_baseline/train.py:187-195`

Actual CSV/counts after `load_data()` logic:

- Raw total rows: 55,320
- Raw 2012 rows / unique `PLOT_ID`: 27,660 / 27,660
- Raw 2023 rows / unique `PLOT_ID`: 27,660 / 27,660
- Raw common plot IDs: 27,660
- Negative-growth common IDs purged: 0
- Final 2012 train rows / unique plots: 27,660 / 27,660
- Final 2023 test rows / unique plots: 27,660 / 27,660
- Final common IDs: 27,660
- 2012 IDs not common after purge: 0
- 2023 IDs not common after purge: 0

Table 4.1 is the clean reference implementation: train and test rows are paired common plots, and the scripts' use of the full returned `df12`/`df23` frames happens to equal the common-plot subset.

### Table 4.2: Temporal, "unseen" plots

CSV loaded: `DATA_PATH_UNSEEN`.

Code paths:

- AvgByAge: `models/avg_by_age/train.py:76-79`
- Chapman-Richards: `models/chapman_richards/train.py:90-94`
- DNN: `models/dnn_baseline/train.py:180-188`
- PINN: `models/pinn_baseline/train.py:245-253`

Actual CSV/counts after `load_data()` logic:

- Raw total rows: 75,047
- Raw 2012 rows / unique `PLOT_ID`: 32,351 / 32,351
- Raw 2023 rows / unique `PLOT_ID`: 42,696 / 42,696
- Raw common plot IDs: 28,864
- Negative-growth common IDs purged: 1,204
- Final 2012 train rows / unique plots: 31,147 / 31,147
- Final 2023 test rows / unique plots: 41,492 / 41,492
- Final common IDs: 27,660
- 2012 IDs not common after purge: 3,487
- 2023 IDs not common after purge: 13,832

This is the intended Table 4.2 protocol. `load_data()` computes `common_ids`, but Table 4.2 trains on all post-purge 2012 rows in the unseen CSV and evaluates on all post-purge 2023 rows in the unseen CSV. The test set therefore intentionally includes 13,832 2023-only plots.

The important documentation point is that Table 4.2 is not "Table 4.1 on a held-out common-plot subset." It is the full unseen-CSV temporal experiment. The loader comments have been updated to make this explicit so this should not be treated as a future bug.

### Table 4.3: 3-fold CV within 2012

CSV loaded: `DATA_PATH_PURGED`.

Code paths:

- AvgByAge: `models/avg_by_age/train.py:58-65`
- Chapman-Richards: `models/chapman_richards/train.py:72-79`
- DNN: `models/dnn_baseline/train.py:148-162`
- PINN: `models/pinn_baseline/train.py:211-224`

Actual data:

- Source rows: final 2012 rows from the purged CSV.
- Total rows / unique plots: 27,660 / 27,660.
- K-fold splitter: `KFold(n_splits=3, shuffle=True, random_state=42)` (`common/data_utils.py:160-164`).
- Each outer fold: 18,440 train rows/plots and 9,220 test rows/plots.
- For DNN/PINN only, each outer training split is further randomly split into 12,354 training rows and 6,086 validation rows by `train_test_split(..., test_size=0.33, random_state=42)`.

### Table 4.4: 3-fold CV within 2023

CSV loaded: `DATA_PATH_PURGED`.

Code paths:

- AvgByAge: `models/avg_by_age/train.py:67-74`
- Chapman-Richards: `models/chapman_richards/train.py:81-88`
- DNN: `models/dnn_baseline/train.py:164-178`
- PINN: `models/pinn_baseline/train.py:228-241`

Actual data:

- Source rows: final 2023 rows from the purged CSV.
- Total rows / unique plots: 27,660 / 27,660.
- K-fold splitter: `KFold(n_splits=3, shuffle=True, random_state=42)`.
- Each outer fold: 18,440 train rows/plots and 9,220 test rows/plots.
- For DNN/PINN only, each outer training split is further randomly split into 12,354 training rows and 6,086 validation rows.

### Common plot intersection logic

The common-plot intersection is centralized in `common/data_utils.py:43` and `common/data_utils.py:55`. It is not reimplemented per table. However, the common IDs are not used downstream by any current table script. This only stays harmless for Tables 4.1, 4.3, and 4.4 because the purged CSV is already perfectly paired across years.

### Static feature assumptions

The requested "static" columns are not model features in `FEATURES`, but they exist in the CSVs. Across the 27,660 common post-purge plot IDs:

Purged CSV:

- `YIELD_CLASS` changed between 2012 and 2023 for 1,105 / 27,660 common plots.
- `PRI_PCT_AREA` changed for 1,216 / 27,660 common plots.
- `SHAPE_AREA` changed for 1,639 / 27,660 common plots.

Unseen CSV:

- `YIELD_CLASS` changed between 2012 and 2023 for 1,105 / 27,660 common plots.
- `PRI_PCT_AREA` changed for 1,216 / 27,660 common plots.
- `SHAPE_AREA` changed for 1,639 / 27,660 common plots.

So these fields should not be described as perfectly constant across years. The current model feature list excludes them, so this does not directly affect DNN/PINN inputs, but it matters for interpretation and any future feature additions.

### Silent filtering, `dropna()`, and `fillna()`

I found no `dropna()` or `fillna()` calls in the codebase. The only explicit row removal is `df12 = df12.drop(index=shrunk_ids)` and `df23 = df23.drop(index=shrunk_ids)` in `common/data_utils.py:51-52`.

The final feature/target matrices have zero missing values in `FEATURES + [TOP_HEIGHT]` for both CSVs and both years after the purge. There is no explicit validation/assertion for this in code.

## 2. Feature Engineering

### Exact feature list

`common/config.py:34-46` defines exactly 18 features:

1. `X`
2. `Y`
3. `AGE`
4. `CULTIVATN_MOUNDING`
5. `CULTIVATN_NO CULTIVATION`
6. `PRILANDUSE_Dead High Forest`
7. `PRILANDUSE_Failed`
8. `PRILANDUSE_Felled`
9. `PRILANDUSE_High Forest`
10. `PRILANDUSE_Open`
11. `PRILANDUSE_Open Water`
12. `PRILANDUSE_Other Built Facility`
13. `PRILANDUSE_Partially Intruded Broadleaf`
14. `PRILANDUSE_Quarries`
15. `PRILANDUSE_Residential`
16. `PRILANDUSE_Unplantable or bare`
17. `PRILANDUSE_Unplanted streamsides`
18. `PRILANDUSE_Windblow`

`AGE` is included in the shared `FEATURES` list at index 2 (`common/config.py:35`). DNN uses all 18 features directly. PINN calls `split_age_column()` (`common/data_utils.py:103-126`), producing 17 "other" features and a separate one-column AGE tensor, then concatenates them back inside `PINN.forward()` (`models/pinn_baseline/model.py:44-49`).

### One-hot encoding

The code does not perform one-hot encoding from raw categories. It assumes the CSVs already contain dummy columns. `build_feature_arrays()` only converts boolean dtype dummy columns to integers (`common/data_utils.py:75-84`).

Dummy columns used:

- Cultivation: 2 columns: `CULTIVATN_MOUNDING`, `CULTIVATN_NO CULTIVATION`
- Primary land use: 13 columns, all `PRILANDUSE_*` columns listed above
- Total dummy columns in `FEATURES`: 15
- Numeric non-dummy columns in `FEATURES`: `X`, `Y`, `AGE`

There is no reference category dropped in the current feature matrix: every row in both CSVs has exactly one active `CULTIVATN_*` dummy and exactly one active `PRILANDUSE_*` dummy. The code does not use `drop_first=True` or equivalent.

Actual activity:

- Purged 2012: both cultivation columns active; all 13 land-use columns active somewhere.
- Purged 2023: both cultivation columns active; 12/13 land-use columns active, with `PRILANDUSE_Open Water` zero in 2023.
- Unseen 2012: both cultivation columns active; all 13 land-use columns active.
- Unseen 2023: both cultivation columns active; all 13 land-use columns active.

### Feature consistency across tables

All model configs import the same `FEATURES`, `TARGET_COL`, `AGE_COL`, and random seed from `common.config`:

- AvgByAge config: `models/avg_by_age/config.py:15`
- Chapman-Richards config: `models/chapman_richards/config.py:20`
- DNN config: `models/dnn_baseline/config.py:15`
- PINN config: `models/pinn_baseline/config.py:21`

AvgByAge and Chapman-Richards do not use the full `FEATURES` list for modeling; they use `AGE` and `TOP_HEIGHT`. DNN and PINN use the full feature list. I found no per-table feature-list override.

## 3. Model Architecture

### DNN

Definition: `models/dnn_baseline/model.py:28-35`.

Architecture:

- Input: `n_features`
- Hidden 1: `Linear(n_features, 128)` + `LeakyReLU`
- Hidden 2: `Linear(128, 128)` + `LeakyReLU`
- Hidden 3: `Linear(128, 128)` + `LeakyReLU`
- Output: `Linear(128, 1)`
- Output is squeezed from shape `(batch, 1)` to `(batch,)` (`models/dnn_baseline/model.py:38-41`).

Actual input size in this codebase: 18.

Parameter count with the current 18 inputs:

- Layer 1: `18*128 + 128 = 2,432`
- Layer 2: `128*128 + 128 = 16,512`
- Layer 3: `128*128 + 128 = 16,512`
- Output: `128*1 + 1 = 129`
- Total: **35,585 trainable parameters**

Comparison to dissertation statement:

- With 20 inputs and PyTorch's default bias on every `Linear`, this architecture has `2,688 + 16,512 + 16,512 + 129 = 35,841` parameters.
- The dissertation's stated 35,840 parameters is exactly one less, which would match 20 inputs if the output bias were excluded. The current code uses PyTorch `nn.Linear(..., bias=True)` by default, so the code's 20-input equivalent is 35,841, not 35,840.
- The current code is also not using 20 inputs; it uses 18.

### PINN

Definition: `models/pinn_baseline/model.py:31-49`.

Architecture:

- Input: `n_other + 1`, where `n_other = 17` and `+1` is the separated AGE tensor.
- Hidden 1: `Linear(18, 128)` + `LeakyReLU`
- Hidden 2: `Linear(128, 128)` + `LeakyReLU`
- Hidden 3: `Linear(128, 128)` + `LeakyReLU`
- Output: `Linear(128, 1)`

Actual trainable parameter count is the same as the DNN: **35,585**.

AGE handling:

- `split_age_column()` finds `AGE` at index 2 and splits it into `X_train_age` / `X_test_age`, with all other columns in `X_train_other` / `X_test_other` (`common/data_utils.py:115-126`).
- Training calls `split_age_column()` in `models/pinn_baseline/train.py:58`.
- `PINN.forward()` concatenates `x_other` and `t_age` along feature dimension (`models/pinn_baseline/model.py:44-49`).
- During training, the batch AGE tensor is cloned and marked `requires_grad_(True)` before the forward pass (`models/pinn_baseline/train.py:110-114`).

PINN CR derivative:

`models/pinn_baseline/model.py:52-63` implements:

`H(t) = y_max * (1 - exp(-k*t))^p`

`dH/dt = y_max * p * (1 - exp(-k*t))^(p - 1) * k * exp(-k*t)`

The implementation:

```python
e = torch.exp(-k * t_unscaled)
return y_max * p * (1 - e) ** (p - 1) * k * e
```

This derivative is algebraically correct.

### Chapman-Richards

Formula definition: `models/chapman_richards/model.py`.

Training call: `models/chapman_richards/train.py:39-43`:

```python
curve_fit(
    chapman_richards,
    ages_train,
    y_train,
    p0=config.CR_P0,
    bounds=config.CR_BOUNDS,
    maxfev=50_000,
)
```

Config:

- `CR_P0 = [46.1, 0.0187, 1.017]` (`models/chapman_richards/config.py:35`)
- `CR_BOUNDS = ([30, 0.001, 0.5], [100, 0.1, 3.0])` (`models/chapman_richards/config.py:36`)

Data fit on per table:

- Table 4.1: fitted on full final 2012 purged data, evaluated on full final 2023 purged data (`models/chapman_richards/train.py:62-67`).
- Table 4.2: fitted on full final 2012 unseen data, evaluated on full final 2023 unseen data (`models/chapman_richards/train.py:90-94`).
- Table 4.3: for each 3-fold split, fitted on the 2012-purged fold's training rows and evaluated on that fold's 2012-purged test rows (`models/chapman_richards/train.py:72-79`).
- Table 4.4: for each 3-fold split, fitted on the 2023-purged fold's training rows and evaluated on that fold's 2023-purged test rows (`models/chapman_richards/train.py:81-88`).

Saved CR parameters in `outputs/chapman_richards/cr_params.json`:

- Table 4.1: `y_max = 46.60095402612564`, `k = 0.01687239286915855`, `p = 0.9917917126952261`
- Table 4.2: `y_max = 51.71757403066702`, `k = 0.012552071796756511`, `p = 0.8864792476902297`

PINN loads the Table 4.1 CR parameters as the purged prior and Table 4.2 CR parameters as the unseen prior (`models/pinn_baseline/train.py:170-184`). Therefore PINN Table 4.2 is using a different CR prior from Tables 4.1/4.3/4.4.

## 4. Training Configuration

### DNN

Config: `models/dnn_baseline/config.py`.

- Hidden size: 128
- L1 lambda: `1e-5`
- Epochs: 1000
- Batch size: 512
- Learning rate: `1e-4`
- Optimizer: Adam (`models/dnn_baseline/train.py:45`)
- Scheduler: `ReduceLROnPlateau(mode='min', factor=0.8, patience=10)` (`models/dnn_baseline/train.py:46`)
- Loss: scaled MSE plus L1 penalty during training (`models/dnn_baseline/train.py:67-71`)
- Early stopping patience: 50 epochs, improvement threshold `1e-5` (`models/dnn_baseline/train.py:91-99`)
- Validation split: `VAL_SPLIT = 0.33`
- Validation split implementation: `train_test_split(np.arange(n_samples), test_size=0.33, random_state=42)` (`common/data_utils.py:148-158`)

DNN per-table scaling:

- `StandardScaler().fit(X_tr_full)` and `StandardScaler().fit(y_tr_full.reshape(-1, 1))` are called inside `run_training()` (`models/dnn_baseline/train.py:20-25`).
- Because `run_training()` is called separately for each table and each CV fold, scalers are fit fresh per table/per fold.
- No scaler is reused across test data. Test data is transformed by the scaler fit on that run's training data (`models/dnn_baseline/train.py:33-38`).

DNN internal split counts:

- Table 4.1: 27,660 train rows split into 18,532 internal train and 9,128 validation.
- Table 4.2: 31,147 train rows split into 20,868 internal train and 10,279 validation.
- Table 4.3 folds: each 18,440 outer-train rows split into 12,354 internal train and 6,086 validation.
- Table 4.4 folds: each 18,440 outer-train rows split into 12,354 internal train and 6,086 validation.

### PINN

Config: `models/pinn_baseline/config.py`.

- Hidden size: 128
- Physics lambda: `1.0`
- L1 lambda: `1e-5`
- Epochs: 1000
- Batch size: 128
- Learning rate: `1e-4`
- Optimizer: Adam (`models/pinn_baseline/train.py:89`)
- Scheduler: `ReduceLROnPlateau(mode='min', factor=0.8, patience=10)` (`models/pinn_baseline/train.py:90`)
- Data loss: MSE on scaled target (`models/pinn_baseline/model.py:96`)
- Physics loss: MSE between unscaled predicted derivative and CR derivative (`models/pinn_baseline/model.py:107-127`)
- Total train loss: `data_loss + LAMBDA_PH * phys_loss + LAMBDA_L1 * l1` (`models/pinn_baseline/train.py:123`)
- Validation loss: plain scaled MSE only; no physics term on validation (`models/pinn_baseline/train.py:131-135`)
- Early stopping patience: 50 epochs, improvement threshold `1e-5` (`models/pinn_baseline/train.py:139-148`)
- Validation split: same `train_val_split()` helper as DNN.

PINN per-table scaling:

- `split_age_column()` separates AGE for each run (`models/pinn_baseline/train.py:58`).
- `make_scalers()` fits `scaler_Xo`, `scaler_age`, and `scaler_y` on that run's training rows only (`common/data_utils.py:129-145`).
- Because `run_training()` is called separately for each table and each CV fold, scalers are fit fresh per table/per fold.

### AvgByAge and Chapman-Richards

AvgByAge has no neural training, optimizer, scaler, validation split, scheduler, or early stopping. It groups the training data by AGE and predicts each test row from the nearest training age's mean height (`models/avg_by_age/train.py:17-30`).

Chapman-Richards has no neural optimizer, scaler, validation split, scheduler, or early stopping. It fits `curve_fit()` directly to the run's training AGE and TOP_HEIGHT arrays.

## 5. Scaling and the Physics Loss

PINN scaling is implemented in two places:

- `make_scalers()` fits `scaler_age` and `scaler_y` on the run's training rows (`common/data_utils.py:136-138`).
- `run_training()` stores `sigma_y`, `sigma_age`, and `age_mean` from those scalers (`models/pinn_baseline/train.py:63-66`).

Definitions:

- `sigma_y = scaler_y.scale_[0]`: standard deviation of training target height in metres.
- `sigma_age = scaler_age.scale_[0]`: standard deviation of training AGE in years.
- `age_mean = scaler_age.mean_[0]`: mean training AGE in years.

The model predicts scaled height, and AGE is passed to the network as scaled AGE. Let:

- `y_s = (y - mu_y) / sigma_y`
- `t_s = (t - mu_t) / sigma_t`

Then:

- `y = mu_y + sigma_y * y_s`
- `t = mu_t + sigma_t * t_s`

The real-world derivative is:

`dy/dt = d(mu_y + sigma_y*y_s) / d(mu_t + sigma_t*t_s)`

Constants vanish, so:

`dy/dt = sigma_y * dy_s / (sigma_t * dt_s)`

Because autograd gives `dy_s / dt_s`, the conversion is:

`dy/dt = (dy_s/dt_s) * (sigma_y / sigma_age)`

The exact code line is `dy_dt_unscaled = dy_dt_scaled * (sigma_y / sigma_age)` in `models/pinn_baseline/model.py:118`. This is correct.

The code then unscales AGE for the CR derivative target:

`t_years = t_scaled.squeeze() * sigma_age + age_mean` (`models/pinn_baseline/model.py:122`)

Then it computes:

`physics_loss = MSE(dy_dt_unscaled, cr_derivative(t_years, cr_params))` (`models/pinn_baseline/model.py:127`)

This scaling logic is mathematically correct, assuming the scaler standard deviations came from the same training population used to scale the batch.

## 6. Metrics

All model training scripts call the shared `reuben_metrics()` function:

- AvgByAge: `models/avg_by_age/train.py:30`
- Chapman-Richards: `models/chapman_richards/train.py:52`
- DNN: `models/dnn_baseline/train.py:118`
- PINN: `models/pinn_baseline/train.py:165`

Metric definitions in `common/metrics.py:19-25`:

- MAE: `sklearn.metrics.mean_absolute_error(y_true, y_pred)`
- MSE: `sklearn.metrics.mean_squared_error(y_true, y_pred)`
- R²: `sklearn.metrics.r2_score(y_true, y_pred)`
- MRE: `mean(abs(y_true - y_pred) / (abs(y_true) + 1e-8))`
- Acc: `(1 - MRE) * 100`

This matches the epsilon-stabilized denominator and `Acc = (1-MRE)*100` definition. It is shared by all training/evaluation code, so the formulas are consistent across the four tables.

Resolved update: spatial plotting now delegates to `common/spatial_plots.py`, and plot titles use the shared `reuben_metrics()` function. Unsigned spatial maps use capped relative error, while signed spatial maps use signed relative error.

## 7. Per-Table Summary

### Table 4.1

Table 4.1 is the temporal common-plot experiment. It trains on the purged CSV's 2012 rows and tests on the purged CSV's 2023 rows. The effective train/test sets are 27,660 rows each, with 27,660 unique `PLOT_ID`s and perfect common-plot overlap. Relative to other tables, this is the reference path: the CSV has already been purged/paired, and the code's failure to subset to `common_ids` does not change the data. DNN/PINN additionally carve 33% of the 2012 training rows into validation before training.

### Table 4.2

Table 4.2 is the temporal unseen-CSV experiment. Every model switches from `DATA_PATH_PURGED` to `DATA_PATH_UNSEEN`: AvgByAge, Chapman-Richards, DNN, and PINN all train on the unseen CSV's 31,147 post-purge 2012 rows and evaluate on the unseen CSV's 41,492 post-purge 2023 rows. Only 27,660 of those test rows are common with the 2012 side; 13,832 are 2023-only plots. This is intentional and should be stated clearly in the methods.

### Table 4.3

Table 4.3 is a 3-fold cross-validation experiment within 2012 purged data. It uses the same purged CSV as Table 4.1 but ignores 2023 as a prediction target. Each fold trains on 18,440 2012 rows and tests on 9,220 2012 rows. For DNN/PINN, the 18,440 fold-training rows are split again into 12,354 train and 6,086 validation rows. Structurally, it differs from Table 4.1 because the target year is not future 2023 height; it is same-year 2012 height held out by random CV.

### Table 4.4

Table 4.4 is a 3-fold cross-validation experiment within 2023 purged data. It uses the same purged CSV as Table 4.1 but uses only 2023 rows. Each fold trains on 18,440 2023 rows and tests on 9,220 2023 rows. For DNN/PINN, the 18,440 fold-training rows are split into 12,354 train and 6,086 validation rows. Structurally, it differs from Table 4.1 because both train and test are 2023 same-year rows, not 2012-to-2023 temporal prediction.

## 8. Current Open Items / Resolved Audit Notes

### Current Open Items

1. `common/data_utils.py:31` still references `config.DATA_PATH`, but `common/config.py` defines only `DATA_PATH_PURGED` and `DATA_PATH_UNSEEN`. Any call to `load_data()` without an explicit path will fail. Normal training/evaluation scripts pass an explicit path, so this is a low-risk cleanup item.

2. Existing spatial PNGs may be stale until regenerated with the current plotting code. The current standard maps are:
   - `spatial_error_map.png`: capped relative error, `min(abs(y_true - y_pred) / (abs(y_true) + 1e-8), 0.5)`.
   - `spatial_signed_error_map.png`: signed relative error, `(y_true - y_pred) / (abs(y_true) + 1e-8)`.
   Both maps now write JSON sidecars and combined comparison scripts refuse mixed metadata.

3. DNN and PINN still do not save table-level prediction CSVs for every table. This is not required for the current Table 4.1 plot workflow because their checkpoints can regenerate predictions, but prediction CSVs would improve reproducibility.

4. PINN partial reruns can preserve stale rows in `outputs/pinn_baseline/results.csv` because `load_existing_results()` merges old and new rows. This is convenient for running one table at a time, but should be handled carefully in final reporting.

5. README text is stale. It still refers to a single `DATA_PATH` and under-describes the current four-model setup.

6. `YIELD_CLASS`, `PRI_PCT_AREA`, and `SHAPE_AREA` are not constant across years for all common plots. This is not a current model bug because they are excluded from `FEATURES`, but any prose or future model treating them as static should be corrected.

### Resolved / Intended Design

1. Table 4.2 is intentionally the full unseen-CSV temporal experiment, not a common-only subset. It trains on 31,147 post-purge 2012 rows and tests on 41,492 post-purge 2023 rows.

2. Feature differences are intentional. AvgByAge and Chapman-Richards are age-only baselines; DNN and PINN are full-feature neural models.

3. PINN's Chapman-Richards prior is an intentional fixed-prior design choice. It should be documented as part of the PINN method rather than treated as a split bug.

4. Plotting functions now centralize metric/title logic through `common/spatial_plots.py` and `common.metrics.reuben_metrics()`.

5. `outputs/compare_results/compare_results.py` now defaults to Table 4.1 only and supports `--table 4.2`, `--table 4.3`, `--table 4.4`, and `--table all`.
