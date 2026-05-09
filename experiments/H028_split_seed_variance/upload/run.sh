#!/bin/bash
# Platform run.sh for H028_split_seed_variance.
# Cohort saturation diagnostic: 3 launches with split_seed 42 / 43 / 44.
# 9 H all share split_seed=42 → same val/OOF cohort → val_auc 0.832~0.836
# saturation across 9 mechanism mutations. H028 varies cohort to test if
# val_auc 변동이 cohort 의존 (#1 hypothesis: cohort saturation).
# H010 mechanism + envelope byte-identical EXCEPT split_seed via "$@".
# §17.2 single mutation = split_seed (cohort assignment).
# §17.4 measurement re-entry justified (H022/H023/H025 sibling — methodology framework).
# §18.8 emit_train_summary inherit from H022.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export EXP_ID="H028_split_seed_variance"

: "${TRAIN_DATA_PATH:?TRAIN_DATA_PATH not set by platform}"
: "${TRAIN_CKPT_PATH:?TRAIN_CKPT_PATH not set by platform}"
: "${TRAIN_LOG_PATH:=${TRAIN_CKPT_PATH}/logs}"
: "${TRAIN_TF_EVENTS_PATH:=${TRAIN_CKPT_PATH}/tf_events}"
: "${TRAIN_WORK_PATH:=${TRAIN_CKPT_PATH}/work}"
mkdir -p "${TRAIN_CKPT_PATH}" "${TRAIN_LOG_PATH}" "${TRAIN_TF_EVENTS_PATH}" "${TRAIN_WORK_PATH}"
export TRAIN_LOG_PATH TRAIN_TF_EVENTS_PATH

echo "[run.sh] DATA=${TRAIN_DATA_PATH}"
echo "[run.sh] CKPT=${TRAIN_CKPT_PATH}"
echo "[run.sh] EXP_ID=${EXP_ID}"
echo "[run.sh] mode=H028 split_seed variance (override --split_seed via \"\$@\")"

# Recommended invocations (3 launches, separate ckpt dirs):
#   bash run.sh --split_seed 42      # control = H010/H022/H023 baseline
#   bash run.sh --split_seed 43      # cohort variation
#   bash run.sh --split_seed 44      # cohort variation
# Note: --seed (training) stays 42 across all 3 to isolate split-only effect.
# Run parallel (3 GPU/slot) for ~3.5h wall, or serial for ~10.5h.
# Diff vs H022/upload/run.sh:
#   --batch_size 2048 → 1024     H028 NEW (OOM safety, H023/H025/H026 fallback regime)
python3 -u "${SCRIPT_DIR}/train.py" \
    --num_epochs 10 \
    --patience 3 \
    --seed 42 \
    --batch_size 1024 \
    --lr 1e-4 \
    --loss_type bce \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --ns_groups_json "" \
    --emb_skip_threshold 1000000 \
    --num_workers 2 \
    --buffer_batches 4 \
    --train_ratio 0.3 \
    --seq_max_lens "seq_a:64,seq_b:64,seq_c:128,seq_d:128" \
    --use_label_time_split \
    --oof_user_ratio 0.1 \
    --split_seed 42 \
    --fusion_type dcn_v2 \
    --dcn_v2_num_layers 2 \
    --dcn_v2_rank 8 \
    --use_ns_to_s_xattn \
    --ns_xattn_num_heads 4 \
    --log_attn_entropy \
    "$@"

echo "[run.sh] training complete (H028 split_seed=$(echo "$@" | grep -oE 'split_seed [0-9]+' || echo 'split_seed 42 default'))"
