# ExcursionForecastability

Code and experiment scripts for time-series forecasting experiments on TelecomTS, MIT Supercloud GPU telemetry, and ESNet.

The repository provides dataset preparation, Pearson correlation analysis, univariate and multivariate forecasting, and evaluation with MSE, Enhanced Peak (EP), and Modified Enhanced Peak (MEP) training objectives.

## Install

```bash
pip install -e '.[dev]'
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

### TelecomTS

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
