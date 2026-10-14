#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 PREPARED_CSV SERIES_ID OUTPUT_CSV [SIGNALS_CSV]" >&2
  exit 2
fi

DATA=$1
SERIES_ID=$2
OUTPUT=$3
SIGNALS_CSV=${4:-}

ARGS=(
  --data "$DATA"
  --series-id "$SERIES_ID"
  --horizons 96 192 336 720
  --input-size 336
  --train-fraction 0.70
  --analysis-val-fraction 0.30
  --prominence-fraction 0.10
  --prominence-window 31
  --event-distance 3
  --event-tolerance 3
  --output "$OUTPUT"
)

if [[ -n "$SIGNALS_CSV" ]]; then
  IFS=',' read -r -a SIGNALS <<< "$SIGNALS_CSV"
  ARGS+=(--signals "${SIGNALS[@]}")
fi

ptaf event-precursors "${ARGS[@]}"
