#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 PREPARED_CSV SERIES_ID [MAX_STEPS] [univariate|multivariate] [SIGNALS_CSV]" >&2
  exit 2
fi

DATA=$1
SERIES_ID=$2
MAX_STEPS=${3:-1000}
TUNING_MAX_STEPS=${TUNING_MAX_STEPS:-100}
MODE=${4:-univariate}
SIGNALS_CSV=${5:-}

HORIZONS=${HORIZONS:-"96 192 336 720"}

if [[ "$MODE" == "univariate" ]]; then
  MODELS=(dlinear patchtst itransformer timemixer)

elif [[ "$MODE" == "multivariate" ]]; then
  if [[ -z "$SIGNALS_CSV" ]]; then
    echo "multivariate mode requires SIGNALS_CSV" >&2
    exit 2
  fi

  IFS=',' read -r -a SIGNALS <<< "$SIGNALS_CSV"
  MODELS=(tsmixer itransformer timemixer)

else
  echo "MODE must be univariate or multivariate" >&2
  exit 2
fi

for h in $HORIZONS; do
  for model in "${MODELS[@]}"; do
    for loss in mse ep mep; do

      echo "=== horizon=$h model=$model loss=$loss mode=$MODE ==="

      CMD=(
        ptaf run
        --data "$DATA"
        --series-id "$SERIES_ID"
        --model "$model"
        --loss "$loss"
        --horizon "$h"
        --input-size 336
        --max-steps "$MAX_STEPS"
        --tuning-max-steps "$TUNING_MAX_STEPS"
        --prominence-fraction 0.10
        --ep-quantile-grid 0.90 0.95
        --mep-low-quantile-grid 0.05 0.10
        --mep-high-quantile-grid 0.90 0.95
        --ep-under-weight-grid 1.0 2.0
        --ep-over-weight-grid 1.0 2.0
        --tuning-metric event_macro_f1
        --seed 1
      )

      if [[ "$MODE" == "multivariate" ]]; then
        CMD+=(--multivariate-signals "${SIGNALS[@]}")
      fi

      "${CMD[@]}"

    done
  done
done
