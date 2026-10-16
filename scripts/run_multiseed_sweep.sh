#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 PREPARED_CSV SERIES_ID [MAX_STEPS] [univariate|multivariate] [SIGNALS_CSV]" >&2
  exit 2
fi

# Five seeds by default. Override, e.g. SEEDS="1 2 3".
export SEEDS=${SEEDS:-"1 2 3 4 5"}
export OUTPUT_ROOT=${OUTPUT_ROOT:-"results/multiseed"}
export SKIP_EXISTING=${SKIP_EXISTING:-1}

exec "$(dirname "$0")/run_horizon_sweep.sh" "$@"
