#!/bin/bash
# Platform run.sh for H025_dcnv2_stronger.
# DCN-V2 capacity sweep (sub-H of H008 winner): rank 8→16, layers 2→4.
# H010 mechanism + envelope byte-identical except 2 DCN-V2 args.
# §17.2 single mutation: capacity within winning interaction-axis category.
# §17.4 interaction_axis_subh re-entry — H008 was best (PASS, platform 0.8387).
# Sub-H 정당화: H008 capacity ceiling 검증 (rank 16 paper-grade DCN-V2 default).
# §18.8 emit_train_summary inherit from H022.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export EXP_ID="H025_dcnv2_stronger"

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
echo "[run.sh] mode=H025 DCN-V2 stronger (rank 16, layers 4)"

# Diff vs H022/upload/run.sh:
#   --dcn_v2_num_layers 2 → 4    H025 NEW
#   --dcn_v2_rank 8 → 16         H025 NEW
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
    --train_ratio 0.3 \
    --seq_max_lens "seq_a:64,seq_b:64,seq_c:128,seq_d:128" \
    --use_label_time_split \
    --oof_user_ratio 0.1 \
    --split_seed 42 \
    --fusion_type dcn_v2 \
    --dcn_v2_num_layers 4 \
    --dcn_v2_rank 16 \
    --use_ns_to_s_xattn \
    --ns_xattn_num_heads 4 \
    --log_attn_entropy \
    "$@"

echo "[run.sh] training complete (H025); metrics at ${TRAIN_CKPT_PATH}/metrics.json"
