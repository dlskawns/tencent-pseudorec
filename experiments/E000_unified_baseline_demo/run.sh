#!/usr/bin/env bash
# E000 — unified-block baseline anchor (H001).
#
# Wrapper that:
#   1. Sets path env vars to ${ROOT}/experiments/E000_unified_baseline_demo/.
#   2. Computes config_sha256 from card.yaml + run args.
#   3. Calls competition/run.sh with explicit args.
#   4. After training, dumps a metrics.json with seed, git_sha, host,
#      best_val_AUC, best_oof_AUC, label_time_cutoff, oof_user count.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export EXP_DIR="${SCRIPT_DIR}"
export TRAIN_DATA_PATH="${ROOT}/data"
export TRAIN_CKPT_PATH="${SCRIPT_DIR}/ckpt"
export TRAIN_LOG_PATH="${SCRIPT_DIR}/logs"
export TRAIN_TF_EVENTS_PATH="${SCRIPT_DIR}/tf_events"
export TRAIN_WORK_PATH="${SCRIPT_DIR}/work"

bash "${ROOT}/competition/run.sh" \
    --num_epochs 8 \
    --patience 5 \
    --seed 42 \
    --batch_size 256 \
    --loss_type bce \
    "$@"

echo "[E000] training done. ckpt under ${TRAIN_CKPT_PATH}"
echo "[E000] sidecar metrics.json should be inspected next."
