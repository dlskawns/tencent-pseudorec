#!/bin/bash
# Platform run.sh for H030_loss_type_focal.
# Loss type isolation diagnostic: --loss_type focal --focal_alpha 0.25
# --focal_gamma 2.0 explicit (paper-grade focal loss for class imbalance).
# H023 = bce explicit (best_val 0.8334) baseline. H030 = focal explicit
# paired Δ — train_loss scale 차이 (H011/H012/H013/H018 0.12 = focal-like)
# 가 진짜 focal 인지 explicit 검증 (#4 hypothesis).
# H010 mechanism + envelope byte-identical EXCEPT loss_type + focal hyperparams.
# §17.2 single mutation = loss type swap (bce → focal).
# §17.4 measurement re-entry (methodology framework).
# Note: H005 was focal but pre-correction era → invalid. H030 = first valid
# focal measurement on corrected eval data.
# §18.8 emit_train_summary inherit.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export EXP_ID="H030_loss_type_focal"

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
echo "[run.sh] mode=H030 loss_type focal (alpha=0.25, gamma=2.0)"

# Diff vs H022/upload/run.sh:
#   --batch_size 2048 → 1024       H030 NEW (OOM safety, match H023 regime)
#   --loss_type bce → focal        H030 NEW (loss class swap)
#   + --focal_alpha 0.25            H030 NEW (paper default)
#   + --focal_gamma 2.0             H030 NEW (paper default)
python3 -u "${SCRIPT_DIR}/train.py" \
    --num_epochs 10 \
    --patience 3 \
    --seed 42 \
    --batch_size 1024 \
    --lr 1e-4 \
    --loss_type focal \
    --focal_alpha 0.25 \
    --focal_gamma 2.0 \
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

echo "[run.sh] training complete (H030 focal); metrics at ${TRAIN_CKPT_PATH}/metrics.json"
