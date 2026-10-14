#!/usr/bin/env bash
set -euo pipefail

RESULTS_ROOT=${1:-results}
OUTPUT=${2:-"$RESULTS_ROOT/event_sensitivity.csv"}

ptaf event-sensitivity \
  --results-root "$RESULTS_ROOT" \
  --prominence-fractions 0.05 0.10 0.20 \
  --event-tolerances 1 3 5 \
  --output "$OUTPUT"
