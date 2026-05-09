#!/bin/bash
# Platform run.sh for H027_train_ratio_expand.
# Data envelope axis sub-H: train_ratio 0.3 → 0.5 (+67% data exposure).
# H010 mechanism + envelope byte-identical except train_ratio.
# §17.2 single mutation: data envelope axis (different from L4 truncate dense).
# §17.4 data_envelope first-touch (rotation auto-justified).
# 9 H 모두 train_ratio=0.3 (sample 30%). 더 많은 data 가 ceiling 깨는지 검증.
# Risk: more data = same regime = same ceiling 가능성 → marginal expected.
# §18.8 emit_train_summary inherit from H022.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export EXP_ID="H027_train_ratio_expand"

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
echo "[run.sh] mode=H027 train_ratio expand (0.3 → 0.5)"

# Diff vs H022/upload/run.sh:
#   --train_ratio 0.3 → 0.5    H027 NEW (data envelope expansion)
# Wall: ~5-6h expected (1.67× of 3.5h).
python3 -u "${SCRIPT_DIR}/train.py" \
    --num_epochs 10 \
    --patience 3 \
    --seed 42 \
    --batch_size 2048 \
    --lr 1e-4 \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --ns_groups_json "" \
    --emb_skip_threshold 1000000 \
    --num_workers 2 \
    --buffer_batches 4 \
    --train_ratio 0.5 \
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

echo "[run.sh] training complete (H027); metrics at ${TRAIN_CKPT_PATH}/metrics.json"
