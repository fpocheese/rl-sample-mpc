#!/bin/bash
# Train all 3 RL variants sequentially.
# Writes logs to checkpoints/<variant>_train.log
# Usage: bash run_all_training.sh

set -e
cd "$(dirname "$0")/.."

LOGDIR="tactical_acados/checkpoints"
mkdir -p "$LOGDIR"

echo "=========================================="
echo "Starting 3-variant RL training"
echo "=========================================="

# 1) oursrl
echo "[$(date)] Starting oursrl..."
mkdir -p "$LOGDIR/oursrl"
conda run --no-capture-output -n a2rldet python3 -u tactical_acados/rl/train_variants.py \
    --variant oursrl \
    --total-episodes 150 \
    --max-steps 300 \
    --save-dir "$LOGDIR/oursrl" \
    --seed 42 \
    > "$LOGDIR/oursrl_train.log" 2>&1
echo "[$(date)] oursrl DONE. Log: $LOGDIR/oursrl_train.log"

# 2) A-oursrl
echo "[$(date)] Starting A-oursrl..."
mkdir -p "$LOGDIR/A_oursrl"
conda run --no-capture-output -n a2rldet python3 -u tactical_acados/rl/train_variants.py \
    --variant A-oursrl \
    --total-episodes 150 \
    --max-steps 300 \
    --save-dir "$LOGDIR/A_oursrl" \
    --seed 42 \
    > "$LOGDIR/A_oursrl_train.log" 2>&1
echo "[$(date)] A-oursrl DONE. Log: $LOGDIR/A_oursrl_train.log"

# 3) pure-rl
echo "[$(date)] Starting pure-rl..."
mkdir -p "$LOGDIR/pure_rl"
conda run --no-capture-output -n a2rldet python3 -u tactical_acados/rl/train_variants.py \
    --variant pure-rl \
    --total-episodes 150 \
    --max-steps 300 \
    --save-dir "$LOGDIR/pure_rl" \
    --seed 42 \
    > "$LOGDIR/pure_rl_train.log" 2>&1
echo "[$(date)] pure-rl DONE. Log: $LOGDIR/pure_rl_train.log"

echo "=========================================="
echo "ALL 3 VARIANTS COMPLETE"
echo "=========================================="
