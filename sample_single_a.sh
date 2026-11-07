#!/usr/bin/env bash
# Sample single_a release: same benchmark configuration as the full single_a
# suite, with reduced task counts for reviewer inspection.
#
# Default: train=128, eval=32 per level. Levels 1..8.
# Custom:  bash sample_single_a.sh <train_n> <eval_n>
#
# Output default: data_sample/single_a-v<babytheorist_version>/

set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python}"
N_TRAIN="${1:-128}"
N_EVAL="${2:-32}"
BABYTHEORIST_VERSION="$("$PY" -c 'from pathlib import Path; ns = {}; exec((Path("babytheorist") / "version.py").read_text(), ns); print(ns["__version__"])')"
OUT_ROOT="${OUT_ROOT:-data_sample}"
MAX_SUPPORT="${MAX_SUPPORT:-5}"
SHAPE_PROFILE="${SHAPE_PROFILE:-standard}"
WORKERS="${WORKERS:-4}"
LEVELS="${LEVELS:-1 2 3 4 5 6 7 8}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
OUT_DIR="${OUT_DIR:-${OUT_ROOT}/single_a-v${BABYTHEORIST_VERSION}}"
DOMINANT_POOL_DIR="${DOMINANT_POOL_DIR:-${OUT_ROOT}/dominant_pools-v${BABYTHEORIST_VERSION}}"
DATASET_VERSION="${DATASET_VERSION:-BabyTheorist-sample-single_a-v${BABYTHEORIST_VERSION}}"
EXTRA_ARGS=()
mkdir -p "$OUT_DIR"

gen() {  # gen <level> <split> <n> <split_mode> <seed> <pool_seed>
    local level=$1 split=$2 n=$3 split_mode=$4 seed=$5 pool_seed=$6
    local out_file="$OUT_DIR/lv${level}_${split}.h5"
    local info_file="$OUT_DIR/lv${level}_${split}.info.json"
    if [[ "$SKIP_EXISTING" == "1" && -s "$out_file" && -s "$info_file" ]]; then
        echo "=== L$level $split already exists; skipping $out_file ==="
        return
    fi
    echo "=== L$level $split (n=$n, split=$split_mode, single_a sample) ==="
    "$PY" generate.py \
        --level "$level" --n "$n" --mode "$split_mode" \
        --world-size 15 --shape-profile "$SHAPE_PROFILE" \
        --benchmark-mode single_a \
        --max-support "$MAX_SUPPORT" \
        --backend filter \
        --workers "$WORKERS" \
        --seed "$seed" --pool-seed "$pool_seed" \
        --dominant-pool-dir "$DOMINANT_POOL_DIR" \
        "${EXTRA_ARGS[@]}" \
        --dataset-version "$DATASET_VERSION" \
        --exclude "$OUT_DIR" \
        --out "$out_file"
}

SEED=42
POOL_SEED=42
for level in $LEVELS; do
    gen "$level" train "$N_TRAIN" train "$SEED"       "$POOL_SEED"
    gen "$level" eval  "$N_EVAL"  eval  "$((SEED+1))" "$POOL_SEED"
done
