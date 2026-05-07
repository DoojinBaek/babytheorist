#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${1:-/tmp/babytheorist_generation_equivalence_fast/run-$(date +%Y%m%d-%H%M%S)-$$}"
RESUME=()
if [[ -e "$RUN_DIR" ]]; then
  RESUME=(--resume)
fi

cd "$ROOT"
python scripts/compare_generation_equivalence.py \
  --train-n 50 \
  --eval-n 50 \
  --levels 1,2,3,4 \
  --modes multi_dev,single_a,single_b,multi_a,multi_b \
  --splits train,eval \
  --workers 32 \
  --run-dir "$RUN_DIR" \
  --stop-on-diff \
  --skip-info \
  "${RESUME[@]}"
