# Scientific Fairness Audit

Audit date: 2026-06-18

Scope: repository code, current output artifacts, and the verified CSV population counts for the four dissertation tables. This audit evaluates whether the statement below is scientifically defensible:

> Performance differences are due solely to model differences, not pipeline differences.

## Verdict

**AMBER**

The framework is defensible as a comparison of deliberately different modelling approaches, provided the methods section states that the baselines intentionally differ in feature access and training procedure. The numeric metrics are mostly computed through a shared function and the major train/test row populations are consistent across models within each named table. The main remaining concern is artifact hygiene: plots and saved outputs must be regenerated after the current metadata/plotting fixes.

- Models intentionally do not all receive the same predictors.
- Neural and non-neural models intentionally do not follow equivalent training procedures.
- Spatial plots now share one common plotting function and metadata contract, but existing PNGs must be regenerated before use.
- Saved artifacts are improved but still depend on rerunning the model scripts to refresh configs, maps, and metadata.

The fairer claim is:

> Within each table, the scripts evaluate each model on the same target rows and metric definitions, but the comparison is between intentionally different modelling pipelines, not only different neural architectures.

## Data Consistency

The row populations used by the current scripts are consistent across the four models for each table.

| Table | Data source | Training rows | Test rows | Notes |
|---|---:|---:|---:|---|
| 4.1 | purged CSV | 27,660 2012 rows | 27,660 2023 rows | paired common plots |
| 4.2 | unseen CSV | 31,147 2012 rows | 41,492 2023 rows | includes 13,832 2023-only plots in test |
| 4.3 | purged CSV | 18,440 per fold | 9,220 per fold | CV within 2012 |
| 4.4 | purged CSV | 18,440 per fold | 9,220 per fold | CV within 2023 |

Table 4.2 is intentionally defined as: train on all available post-purge 2012 rows in the unseen CSV and test on all available post-purge 2023 rows in the unseen CSV. That means 31,147 training rows and 41,492 test rows, including 13,832 2023-only plots in the test set.

Affected code:

- `common/data_utils.py:10-11`
- `common/data_utils.py:23-61`
- `models/avg_by_age/train.py:159-160`
- `models/chapman_richards/train.py:154-156`
- `models/dnn_baseline/train.py:182-190`
- `models/pinn_baseline/train.py:245-253`

The loader comments have been updated to document this explicitly, so this should not be treated as a future audit finding.

## Feature Design

The feature sets are intentionally not equivalent.

| Model | Predictors used | Scaling | Consequence |
|---|---|---|---|
| AvgByAge | `AGE` only | none | age lookup baseline |
| Chapman-Richards | `AGE` only | none | parametric age curve |
| DNN | all 18 `FEATURES` | `StandardScaler` for X and y | uses spatial coordinates and one-hot site variables |
| PINN | all 18 `FEATURES`, with `AGE` split for autograd | separate scalers for non-age, age, and y | DNN-like feature set plus CR physics prior |

The 18-feature neural input is defined in `common/config.py:34-46`. AvgByAge and Chapman-Richards intentionally use only `AGE` and `TOP_HEIGHT`.

This is a deliberate modelling choice. The dissertation should describe the comparison as age-only empirical/parametric baselines versus full-feature neural models, rather than as a same-feature architecture-only comparison.

Recommended wording:

- Add a clear methods statement: "The baselines intentionally differ in predictor access: age-only baselines versus full-feature neural baselines."
- Do not frame these feature differences as an implementation bug.

## Training Consistency

The training protocols differ materially by design.

| Model | Training procedure |
|---|---|
| AvgByAge | deterministic grouped mean by nearest age |
| Chapman-Richards | `scipy.optimize.curve_fit` with fixed initial parameters and bounds |
| DNN | Adam, 1000 max epochs, batch size 512, 33% validation split, L1, scheduler, early stopping |
| PINN | Adam, 1000 max epochs, batch size 128, 33% validation split, L1, scheduler, early stopping, physics loss |

The neural models also use stochastic minibatch training. For the dissertation's main purpose, this is acceptable if the run configuration and seed are saved and the methods section does not claim bit-exact reproducibility across hardware.

Affected code:

- `models/dnn_baseline/train.py:123-125`
- `models/pinn_baseline/train.py:294-295`
- DNN `DataLoader` setup in `models/dnn_baseline/train.py`
- PINN `DataLoader` setup in `models/pinn_baseline/train.py`

Recommended documentation:

- Save the exact device, package versions, seed, table, row counts, feature list, and final epoch for every run.
- State that neural results are seed-controlled but may not be bit-exact across hardware.

## Evaluation Consistency

Metric formulas are mostly consistent. All training scripts call `common.metrics.reuben_metrics()`, which computes:

- MAE
- MSE
- R2
- MRE using `abs(y_true) + 1e-8`
- Accuracy as `(1 - MRE) * 100`

Affected code:

- `common/metrics.py:13-25`
- `models/avg_by_age/train.py:111`
- `models/chapman_richards/train.py:114`
- `models/dnn_baseline/train.py:119`
- `models/pinn_baseline/train.py:165`

Resolved: plotting functions now call the shared metric function for displayed MAE/Accuracy, so plot titles and result tables use the same metric definitions.

Implemented fix:

- Spatial plotting now delegates to `common/spatial_plots.py`, which uses `reuben_metrics()`.

## Plotting Consistency

The current plot code has been mostly standardized for spatial hexbins:

- Absolute error maps use `C = np.abs(y_test - y_pred)`.
- Absolute maps use `vmin=0`, `vmax=10`.
- Absolute maps use `cmap=plt.cm.viridis`.
- Signed maps use `C = y_test - y_pred`.
- Signed maps use `vmin=-10`, `vmax=10`.
- Signed maps use `PINK_WHITE_BLUE`.
- All checked maps use `reduce_C_function=np.mean`.

Affected plot files:

- `models/avg_by_age/plots.py`
- `models/chapman_richards/plots.py`
- `models/dnn_baseline/plots.py`
- `models/pinn_baseline/plots.py`

The larger issue was that maps were generated from different entry points:

- AvgByAge and Chapman-Richards previously generated Table 4.2 spatial maps.
- DNN and PINN previously relied on `evaluate.py` for Table 4.1 spatial maps.
- The comparison graph scripts previously pasted existing PNG files without validating table, data path, predictions, or timestamp.

Affected code:

- `models/avg_by_age/train.py:164-189`
- `models/chapman_richards/train.py:161-195`
- `models/dnn_baseline/evaluate.py:30-32` and `models/dnn_baseline/evaluate.py:101-108`
- `models/pinn_baseline/evaluate.py:30-33` and `models/pinn_baseline/evaluate.py:102-109`
- `outputs/compare_results/compare_graphs_spatial_error.py:23-83`
- `outputs/compare_results/compare_graphs_spatial_signed_error.py`

Implemented fix: all model plotting wrappers now call `common/spatial_plots.py`; standard `spatial_error_map.png` and `spatial_signed_error_map.png` are Table 4.1 temporal common-plot maps; every map writes a JSON sidecar; comparison scripts refuse to combine plots unless table, data label, sample size, color scale, colormap, reducer, and grid size match.

Remaining requirement:

- Rerun each model's training script so the old PNGs are replaced with new Table 4.1 maps and sidecar JSON files.

## Output Saving Consistency

Current artifact saving is uneven.

| Model | Results CSV | Config | Predictions | Maps from train | Checkpoint |
|---|---|---|---|---|---|
| AvgByAge | yes | yes | Table 4.2 diagnostics | Table 4.1 maps | not applicable |
| Chapman-Richards | yes | yes | Table 4.2 diagnostics | Table 4.1 maps | not applicable |
| DNN | yes | yes after full train | no prediction CSV | Table 4.1 maps | Table 4.1 only |
| PINN | yes | yes if Table 4.1 ran | no prediction CSV | Table 4.1 maps | Table 4.1 only |

Two additional risks:

1. `models/pinn_baseline/train.py` merges existing `results.csv` with newly run table results. If only one table is rerun, old rows can remain and look fresh.
2. Existing map PNGs may be stale until the model scripts are rerun.

Affected code:

- `models/pinn_baseline/train.py:269-322`
- `models/pinn_baseline/train.py:324-369`
- `models/dnn_baseline/train.py:194-278`
- `models/dnn_baseline/evaluate.py:30-108`
- `models/pinn_baseline/evaluate.py:30-122`

Recommended fix:

- For every model and every table, save a prediction CSV with `PLOT_ID`, `X`, `Y`, `AGE`, `y_true`, `y_pred`, `abs_error`, `signed_error`, and `table`.
- Generate `results.csv`, maps, and comparison figures from those prediction CSVs, not from separate ad hoc paths.
- Avoid merging fresh results into old CSV rows unless stale rows are explicitly marked.

## PINN Physics Prior

This is an intentional methodological choice in this project, not a blocker for the current dissertation comparison.

PINN loads Chapman-Richards priors from `outputs/chapman_richards/cr_params.json`:

- Table 4.1 prior for purged runs.
- Table 4.2 prior for unseen runs.

Then PINN Table 4.3 and Table 4.4 use the full purged Table 4.1 CR prior for every CV fold.

Affected code:

- `models/pinn_baseline/train.py:170-184`
- `models/pinn_baseline/train.py:211-225`
- `models/pinn_baseline/train.py:228-242`

The PINN uses the saved Chapman-Richards parameters as a fixed physics prior. This should be described as part of the PINN design rather than treated as a data split bug.

Recommended documentation:

- State that PINN uses the Chapman-Richards curve as a fixed prior generated by the CR baseline.
- Save the CR parameters actually used in `config_used.json` / checkpoint metadata.

## Hidden Bugs And Documentation Risks

1. `load_data(path=None)` references `config.DATA_PATH`, but `common/config.py` no longer defines `DATA_PATH`. Any default call will fail.

2. `common/data_utils.py` documentation says only common plots are kept, but the function returns full post-purge frames.

3. `models/pinn_baseline/train.py` header and CLI comments have been corrected to state that checkpoint/config/plots are Table 4.1-oriented.

4. `README.md` still describes a single `DATA_PATH` and says the repo includes only Chapman-Richards and PINN, while it now includes AvgByAge and DNN too.

5. Compare graph scripts do not validate that all plots share the same table, colormap scale, source predictions, or generation date.

## Severity Register

| Severity | Finding | Affected experiments | Fix priority |
|---|---|---|---|
| High | Existing spatial PNGs can be stale until regenerated with sidecar metadata | all visual comparisons | rerun before using figures |
| Medium | Models intentionally use different feature sets | all model comparisons | disclose clearly |
| Medium | DNN/PINN do not save table-level prediction CSVs | DNN/PINN all tables | fix for reproducibility |
| Medium | PINN results merge can preserve stale table rows | PINN partial reruns | fix stale-result handling |
| Resolved | Loader documentation contradicted intended Table 4.2 behavior | Table 4.2 interpretation | comments updated |
| Medium | Neural reproducibility is seed-controlled but not fully deterministic | DNN/PINN | document or harden |
| Low | Plot title metrics duplicate metric formulas | all maps | centralize metrics |
| Low | README is stale | repository documentation | update before submission |

## Minimum Fixes Before Reporting Final Results

1. Rerun all four model training scripts so `spatial_error_map.png`, `spatial_signed_error_map.png`, and their JSON sidecars are fresh Table 4.1 artifacts.
2. Run the comparison graph scripts only after the metadata checks pass.
3. Document Table 4.2 as the intended full unseen-CSV protocol: 31,147 post-purge 2012 training rows and 41,492 post-purge 2023 test rows.
4. Document feature/training differences as intentional differences between model families.
5. Save complete `config_used.json` files for all models, including row counts, feature list, table IDs, data paths, metrics, and output files.
6. Update stale README text before final submission.

## Final Reviewer Statement

The framework is usable as a comparison of intentionally different model families, not as an architecture-only ablation. The current data populations are consistent across models within each table, Table 4.2's full unseen-CSV protocol is intentional, and spatial plotting has been centralized with metadata checks. The remaining practical requirement is to rerun the model scripts so outputs on disk match the corrected code.

Final status: **AMBER until all outputs are regenerated; GREEN is reasonable after fresh maps/configs/results are produced and the methods text documents the intentional design differences.**
