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
SEEDS=${SEEDS:-"1"}
LOSSES=${LOSSES:-"mse ep mep"}
OUTPUT_ROOT=${OUTPUT_ROOT:-}
SKIP_EXISTING=${SKIP_EXISTING:-1}

if [[ "$MODE" == "univariate" ]]; then
  if [[ -n "${MODELS:-}" ]]; then
    read -r -a MODELS_ARRAY <<< "$MODELS"
  else
    MODELS_ARRAY=(dlinear patchtst itransformer timemixer)
  fi

elif [[ "$MODE" == "multivariate" ]]; then
  if [[ -z "$SIGNALS_CSV" ]]; then
    echo "multivariate mode requires SIGNALS_CSV" >&2
    exit 2
  fi

  IFS=',' read -r -a SIGNALS <<< "$SIGNALS_CSV"
  if [[ -n "${MODELS:-}" ]]; then
    read -r -a MODELS_ARRAY <<< "$MODELS"
  else
    MODELS_ARRAY=(tsmixer itransformer timemixer)
  fi

else
  echo "MODE must be univariate or multivariate" >&2
  exit 2
fi

for h in $HORIZONS; do
  for model in "${MODELS_ARRAY[@]}"; do
    for loss in $LOSSES; do
      for seed in $SEEDS; do

      echo "=== horizon=$h model=$model loss=$loss mode=$MODE seed=$seed ==="

      SAFE_SERIES=${SERIES_ID//\//_}
      SAFE_SERIES=${SAFE_SERIES//:/_}
      RUN_SUFFIX=""
      if [[ "$MODE" == "multivariate" ]]; then
        RUN_SUFFIX="__mv"
      fi
      RUN_DIR=""
      if [[ -n "$OUTPUT_ROOT" ]]; then
        RUN_DIR="$OUTPUT_ROOT/${SAFE_SERIES}__${model}__${loss}__h${h}__seed${seed}${RUN_SUFFIX}"
        if [[ "$SKIP_EXISTING" == "1" && -f "$RUN_DIR/config.json" && -f "$RUN_DIR/forecast.csv" ]]; then
          echo "=== skip existing $RUN_DIR ==="
          continue
        fi
      fi

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
        --seed "$seed"
      )

      if [[ "$MODE" == "multivariate" ]]; then
        CMD+=(--multivariate-signals "${SIGNALS[@]}")
      fi
      if [[ -n "$RUN_DIR" ]]; then
        CMD+=(--output-dir "$RUN_DIR")
      fi

      "${CMD[@]}"

      done
    done
  done
done
