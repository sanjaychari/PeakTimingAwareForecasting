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
  --output "$OUTPUT"
)

if [[ -n "$SIGNALS_CSV" ]]; then
  IFS=',' read -r -a SIGNALS <<< "$SIGNALS_CSV"
  ARGS+=(--signals "${SIGNALS[@]}")
fi

ptaf correlate "${ARGS[@]}"
