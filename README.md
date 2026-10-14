# ExcursionForecastability

Code and experiment scripts for time-series forecasting experiments on TelecomTS, MIT Supercloud GPU telemetry, and ESNet.

The repository provides dataset preparation, whole-signal Pearson correlation analysis, event-specific precursor analysis, univariate and multivariate forecasting, and evaluation with MSE, Enhanced Peak (EP), and Modified Enhanced Peak (MEP) training objectives. It also includes a sensitivity analysis for the event prominence threshold and temporal matching tolerance.

## Environment setup

Python 3.11 is recommended. Create a local virtual environment from the repository root and install the package in editable mode:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

After activation, verify that the environment is being used:

```bash
which python
python --version
```

On macOS/Linux, `which python` should point to `.venv/bin/python` inside this repository. Reactivate the environment in a new shell with:

```bash
source .venv/bin/activate
```

## Experiment configuration

- Input length: `336`
- Forecast horizons: `96, 192, 336, 720`
- Univariate models: DLinear, PatchTST, iTransformer, TimeMixer
- Multivariate models: TSMixer, iTransformer, TimeMixer
- Training objectives: MSE, EP, MEP
- Event evaluation: prominence-qualified peaks and troughs
- Primary event prominence fraction: `0.10` of the training `Q99 - Q01` range
- Primary event matching tolerance: `±3` time steps
- Train/validation/test split: chronological
- Correlation calculations: training portion only
- Event-specific precursor analysis: training portion only

The third positional argument passed to `run_horizon_sweep.sh` is the maximum number of optimizer updates used for the final model fit. The examples below use `1000`. This value is a training-step budget, not a number of epochs.

## Loss implementations

### Enhanced Peak (EP)

The `ep` objective applies different penalties to underprediction and overprediction when the ground-truth value is above a configurable upper threshold.

The threshold and penalty factors are chosen with the validation split. The default search is:

```text
T quantile in {0.90, 0.95}
Fu         in {1.0, 2.0}
Fo         in {1.0, 2.0}
```

This produces 8 EP configurations per model and forecast horizon.

### Modified Enhanced Peak (MEP)

The `mep` objective extends the EP implementation so that the additional underprediction and overprediction penalties are applied to values below a configurable lower threshold as well as above an upper threshold.

The default MEP search is:

```text
TL quantile in {0.05, 0.10}
TH quantile in {0.90, 0.95}
Fu          in {1.0, 2.0}
Fo          in {1.0, 2.0}
```

This produces 16 MEP configurations per model and forecast horizon.

EP and MEP grid-search candidates use `--tuning-max-steps` (`100` by default). After selection on the validation split, the winning configuration is initialized again and trained with the full `--max-steps` budget before test evaluation. Test data is not used to choose the loss hyperparameters.

## Prepare datasets

The raw datasets are not stored in this repository. The commands below assume that raw data is placed under `../data/raw/` and processed files are written under `../data/processed/`.

### TelecomTS

TelecomTS is publicly available from the Hugging Face dataset repository `AliMaatouk/TelecomTS`:

```text
https://huggingface.co/datasets/AliMaatouk/TelecomTS
```

One way to obtain the raw directory structure used by the preparation command is:

```bash
mkdir -p ../data/raw
git clone https://huggingface.co/datasets/AliMaatouk/TelecomTS \
  ../data/raw/telecomts
```

The dataset is approximately 1.3 GB. The preparation command reads the scenario-level telemetry files from this directory and produces the aligned CSV used by the experiments:

```bash
ptaf prepare telecomts \
  --root ../data/raw/telecomts \
  --output ../data/processed/telecomts.csv
```

Selected target:

```text
telecomts::normal/stationary/Zone_B/no_congestion/File::TX_Bytes
```

Signals used by the multivariate experiments:

```text
TX_Bytes
RX_Bytes
Estimated_UL_Buffer
PRB_Utilization_DL
PRB_Utilization_UL
```

### MIT Supercloud

The MIT Supercloud Dataset is publicly available through the Amazon Open Data Registry. The January 2022 release is stored at:

```text
s3://mit-supercloud-dataset/datacenter-challenge/202201/
```

Install the AWS CLI if it is not already available. The bucket supports anonymous reads, so an AWS account is not required. To inspect the available GPU data:

```bash
aws s3 ls \
  s3://mit-supercloud-dataset/datacenter-challenge/202201/gpu/ \
  --recursive --no-sign-request
```

The complete MIT Supercloud release is very large, so downloading the entire dataset is unnecessary for these experiments. Copy the GPU CSV files needed for the selected workload(s) into:

```text
../data/raw/mit_supercloud/gpu_labeled_sample/
```

For example, individual files can be downloaded with:

```bash
mkdir -p ../data/raw/mit_supercloud/gpu_labeled_sample

aws s3 cp \
  s3://mit-supercloud-dataset/datacenter-challenge/202201/gpu/<path-to-file.csv> \
  ../data/raw/mit_supercloud/gpu_labeled_sample/ \
  --no-sign-request
```

Then prepare the GPU traces:

```bash
ptaf prepare mit-gpu \
  --root ../data/raw/mit_supercloud/gpu_labeled_sample \
  --output ../data/processed/mit_supercloud_gpu_utilization_gpu_pct.csv
```

Selected workload:

```text
mit_gpu::67691045781943-r7343737-n685852::utilization_gpu_pct
```

The MIT workload is used only for univariate forecasting.

### ESNet

The ESNet telemetry used by these experiments is not publicly distributed with this repository. Users with access to the source IF-MIB telemetry should place it in a local directory and pass that directory to `generate_esnet_delta_csv.py`.

Convert the cumulative IF-MIB counters to five-minute deltas:

```bash
python scripts/generate_esnet_delta_csv.py \
  --root /path/to/ESNet_Data \
  --device r-nova-fd-01 \
  --output ../data/processed/r-nova-fd-01_metrics_delta.csv
```

Convert the resulting wide file to the repository's aligned format:

```bash
ptaf prepare esnet \
  --raw ../data/processed/r-nova-fd-01_metrics_delta.csv \
  --output ../data/processed/esnet_r-nova-fd-01_delta.csv
```

Selected target:

```text
esnet::r-nova-fd-01::ifHCOutOctets
```

Signals used by the multivariate experiments:

```text
ifHCOutOctets
ifHCOutUcastPkts
ifHCOutMulticastPkts
ifHCOutBroadcastPkts
ifHCInOctets
ifHCInUcastPkts
ifHCInMulticastPkts
ifHCInBroadcastPkts
```

## Pearson correlation calculations

For each selected target, the correlation script evaluates

```text
corr(y_t, y_{t+h})
```

for the target history and

```text
corr(x_j,t, y_{t+h})
```

for each auxiliary input signal. The offsets are `96`, `192`, `336`, and `720`. Calculations use only the training portion of the series.

### TelecomTS

```bash
./scripts/run_correlation_analysis.sh \
  ../data/processed/telecomts.csv \
  'telecomts::normal/stationary/Zone_B/no_congestion/File::TX_Bytes' \
  results/correlation/telecomts.csv \
  'TX_Bytes,RX_Bytes,Estimated_UL_Buffer,PRB_Utilization_DL,PRB_Utilization_UL'
```

### MIT Supercloud

```bash
./scripts/run_correlation_analysis.sh \
  ../data/processed/mit_supercloud_gpu_utilization_gpu_pct.csv \
  'mit_gpu::67691045781943-r7343737-n685852::utilization_gpu_pct' \
  results/correlation/mit_supercloud.csv
```

### ESNet

```bash
./scripts/run_correlation_analysis.sh \
  ../data/processed/esnet_r-nova-fd-01_delta.csv \
  'esnet::r-nova-fd-01::ifHCOutOctets' \
  results/correlation/esnet.csv \
  'ifHCOutOctets,ifHCOutUcastPkts,ifHCOutMulticastPkts,ifHCOutBroadcastPkts,ifHCInOctets,ifHCInUcastPkts,ifHCInMulticastPkts,ifHCInBroadcastPkts'
```


## Event-specific precursor analysis

The Pearson correlation analysis above measures dependence over the complete signal. The event-specific precursor analysis instead asks whether the history immediately available to a forecasting model contains structure that distinguishes future excursion windows from ordinary windows.

The analysis uses only the same 70% training prefix used by the forecasting experiments; it does not use the forecasting validation or test splits. For each horizon, a 336-step history ending at time `t` is labeled positive when a prominence-qualified event occurs within `±3` steps of `t + h`.

For each history window, the analysis extracts deterministic summary features (`last`, `mean`, `std`, `min`, `max`, `delta`, and `slope`) for the available signals. It reports:

- the strongest absolute point-biserial correlation between a precursor feature and the future-event label;
- the signal and summary statistic producing that correlation;
- ROC AUC and balanced accuracy from a lightweight nearest-centroid classifier.

The classifier is used only as a diagnostic of whether event-specific precursor structure generalizes within the training prefix. It is not treated as a competing forecasting model.

### TelecomTS univariate

```bash
./scripts/run_event_precursor_analysis.sh \
  ../data/processed/telecomts.csv \
  'telecomts::normal/stationary/Zone_B/no_congestion/File::TX_Bytes' \
  results/telecomts_event_precursors_univariate.csv
```

### TelecomTS multivariate

```bash
./scripts/run_event_precursor_analysis.sh \
  ../data/processed/telecomts.csv \
  'telecomts::normal/stationary/Zone_B/no_congestion/File::TX_Bytes' \
  results/telecomts_event_precursors_multivariate.csv \
  'TX_Bytes,RX_Bytes,Estimated_UL_Buffer,PRB_Utilization_DL,PRB_Utilization_UL'
```

### MIT Supercloud

```bash
./scripts/run_event_precursor_analysis.sh \
  ../data/processed/mit_supercloud_gpu_utilization_gpu_pct.csv \
  'mit_gpu::67691045781943-r7343737-n685852::utilization_gpu_pct' \
  results/mit_event_precursors_univariate.csv
```

### ESNet univariate

```bash
./scripts/run_event_precursor_analysis.sh \
  ../data/processed/esnet_r-nova-fd-01_delta.csv \
  'esnet::r-nova-fd-01::ifHCOutOctets' \
  results/esnet_event_precursors_univariate.csv
```

### ESNet multivariate

```bash
./scripts/run_event_precursor_analysis.sh \
  ../data/processed/esnet_r-nova-fd-01_delta.csv \
  'esnet::r-nova-fd-01::ifHCOutOctets' \
  results/esnet_event_precursors_multivariate.csv \
  'ifHCOutOctets,ifHCOutUcastPkts,ifHCOutMulticastPkts,ifHCOutBroadcastPkts,ifHCInOctets,ifHCInUcastPkts,ifHCInMulticastPkts,ifHCInBroadcastPkts'
```

The output contains separate rows for `any`, `peak`, and `trough` events at horizons `96`, `192`, `336`, and `720`. The main diagnostic columns are:

```text
horizon
event_type
n_positive
positive_fraction
max_abs_point_biserial_r
best_point_biserial_r
best_precursor_feature
classifier_auc
classifier_balanced_accuracy
n_fit_positive
n_eval_positive
```

## Forecasting sweeps

`run_horizon_sweep.sh` runs the four configured horizons and all three objectives (`mse`, `ep`, and `mep`).

### TelecomTS univariate

```bash
./scripts/run_horizon_sweep.sh \
  ../data/processed/telecomts.csv \
  'telecomts::normal/stationary/Zone_B/no_congestion/File::TX_Bytes' \
  1000 univariate
```

### TelecomTS multivariate

```bash
./scripts/run_horizon_sweep.sh \
  ../data/processed/telecomts.csv \
  'telecomts::normal/stationary/Zone_B/no_congestion/File::TX_Bytes' \
  1000 multivariate \
  'TX_Bytes,RX_Bytes,Estimated_UL_Buffer,PRB_Utilization_DL,PRB_Utilization_UL'
```

### MIT Supercloud

```bash
./scripts/run_horizon_sweep.sh \
  ../data/processed/mit_supercloud_gpu_utilization_gpu_pct.csv \
  'mit_gpu::67691045781943-r7343737-n685852::utilization_gpu_pct' \
  1000 univariate
```

### ESNet univariate

```bash
./scripts/run_horizon_sweep.sh \
  ../data/processed/esnet_r-nova-fd-01_delta.csv \
  'esnet::r-nova-fd-01::ifHCOutOctets' \
  1000 univariate
```

### ESNet multivariate

```bash
./scripts/run_horizon_sweep.sh \
  ../data/processed/esnet_r-nova-fd-01_delta.csv \
  'esnet::r-nova-fd-01::ifHCOutOctets' \
  1000 multivariate \
  'ifHCOutOctets,ifHCOutUcastPkts,ifHCOutMulticastPkts,ifHCOutBroadcastPkts,ifHCInOctets,ifHCInUcastPkts,ifHCInMulticastPkts,ifHCInBroadcastPkts'
```

To change the tuning budget:

```bash
TUNING_MAX_STEPS=50 ./scripts/run_horizon_sweep.sh ...
```

To run only selected horizons:

```bash
HORIZONS='96 192' ./scripts/run_horizon_sweep.sh ...
```


## Event-definition sensitivity analysis

The primary paper results use a prominence fraction of `0.10` and an event-matching tolerance of `±3` steps. The sensitivity analysis recomputes the event metrics from the saved forecasts without retraining any forecasting models.

It evaluates the same paper configurations over:

```text
prominence fraction: 0.05, 0.10, 0.20
matching tolerance:  ±1, ±3, ±5 steps
```

Run the analysis after the forecasting sweeps have completed:

```bash
./scripts/run_event_sensitivity.sh \
  results \
  results/event_sensitivity.csv
```

The script evaluates the 216 forecasting configurations used by the paper. With nine event-definition combinations per configuration, the output contains 1,944 rows.

For each configuration and event definition, the output reports both the mean/median window-level Event F1 and event counts aggregated across all test windows:

```text
window_event_macro_f1_mean
window_event_macro_f1_median
global_peak_f1
global_trough_f1
global_event_macro_f1
peak_tp
peak_fp
peak_fn
trough_tp
trough_fp
trough_fn
```

This analysis is intended to test whether the qualitative excursion-forecasting conclusions depend on the exact prominence threshold or temporal matching tolerance.

## Output files

Each run writes:

```text
forecast.csv
window_metrics.csv
metrics.csv
config.json
```

EP and MEP runs also write:

```text
loss_grid_search.csv
```

Multivariate runs additionally write:

```text
metrics_by_signal.csv
metrics_focus_signal.csv
```

`window_metrics.csv` contains metrics for individual non-overlapping test windows. `metrics.csv` aggregates those window-level values and stores the mean, median, population standard deviation, and number of valid observations for each metric.

The event-specific precursor commands write the requested `*_event_precursors_*.csv` files directly under `results/`. The event-definition sensitivity command writes `results/event_sensitivity.csv`.
