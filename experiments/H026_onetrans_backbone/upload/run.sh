#!/bin/bash
# Platform run.sh for H026_onetrans_backbone.
# Backbone class first-touch: --backbone onetrans (default hyformer).
# H010 mechanism (NS xattn + DCN-V2) + envelope byte-identical, backbone swap.
# H004 was OneTrans anchor 측정 (invalid, heuristic fallback 0.5 era).
# 현재 §18 인프라 + corrected eval data 로 OneTrans backbone 첫 valid 측정.
# §17.2 single mutation: backbone class swap.
# §17.4 backbone_replacement first-touch (rotation auto-justified).
# §10.9 attn entropy threshold check active (--log_attn_entropy).
# §18.8 emit_train_summary inherit from H022.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export EXP_ID="H026_onetrans_backbone"

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
echo "[run.sh] mode=H026 OneTrans backbone (--backbone onetrans, all other H010 args identical)"

# Diff vs H022/upload/run.sh:
#   + --backbone onetrans     H026 NEW (default 'hyformer')
python3 -u "${SCRIPT_DIR}/train.py" \
    --num_epochs 10 \
    --patience 3 \
    --seed 42 \
    --batch_size 2048 \
    --lr 1e-4 \
    --backbone onetrans \
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

echo "[run.sh] training complete (H026); metrics at ${TRAIN_CKPT_PATH}/metrics.json"
