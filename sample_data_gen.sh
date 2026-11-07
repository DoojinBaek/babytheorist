#!/usr/bin/env bash
# Generate the compact BabyTheorist sample release for reviewer inspection.
#
# Default: train=128, eval=32 per suite/level. Levels 1..8.
# Custom:  bash sample_data_gen.sh <train_n> <eval_n>
#
# Output default:
#   data_sample/single_a-v<version>/
#   data_sample/multi_a-v<version>/
#   data_sample/single_b-v<version>/
#   data_sample/multi_b-v<version>/

set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python}"
N_TRAIN="${1:-128}"
N_EVAL="${2:-32}"
OUT_ROOT="${OUT_ROOT:-data_sample}"
MAX_SUPPORT="${MAX_SUPPORT:-5}"
SHAPE_PROFILE="${SHAPE_PROFILE:-standard}"
WORKERS="${WORKERS:-32}"
LEVELS="${LEVELS:-1 2 3 4 5 6 7 8}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

echo "=== BabyTheorist sample generation ==="
echo "OUT_ROOT=$OUT_ROOT"
echo "N_TRAIN=$N_TRAIN N_EVAL=$N_EVAL"
echo "WORKERS=$WORKERS LEVELS=$LEVELS"

for script in \
    sample_single_a.sh \
    sample_multi_a.sh \
    sample_single_b.sh \
    sample_multi_b.sh
do
    echo
    echo "=== running $script ==="
    PYTHON="$PY" \
    OUT_ROOT="$OUT_ROOT" \
    MAX_SUPPORT="$MAX_SUPPORT" \
    SHAPE_PROFILE="$SHAPE_PROFILE" \
    WORKERS="$WORKERS" \
    LEVELS="$LEVELS" \
    SKIP_EXISTING="$SKIP_EXISTING" \
        bash "$script" "$N_TRAIN" "$N_EVAL"
done
