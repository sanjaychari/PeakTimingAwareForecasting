# ExcursionForecastability

Code and experiment scripts for time-series forecasting experiments on TelecomTS, MIT Supercloud GPU telemetry, and ESNet.

The repository provides dataset preparation, Pearson correlation analysis, univariate and multivariate forecasting, and evaluation with MSE, Enhanced Peak (EP), and Modified Enhanced Peak (MEP) training objectives.

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
- Train/validation/test split: chronological
- Correlation calculations: training portion only

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
