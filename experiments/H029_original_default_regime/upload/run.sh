#!/bin/bash
# Platform run.sh for H029_original_default_regime.
# Keskar trap escape diagnostic: --batch_size 256 + --lr 1e-4 (H010 train.py
# default). 9 H 모두 사용자 override batch=2048 또는 fallback 1024 사용 →
# original default regime 한 번도 측정 안 됨. F-2 carry-forward: batch=2048
# + lr=1e-4 → effective lr 1/8 underpowered. batch=256 default 가 진짜
# baseline 일 가능성 → val_auc 0.832 ceiling 깨질 수 있음 (#2 hypothesis).
# H010 mechanism + envelope byte-identical EXCEPT batch_size + (recompute) lr.
# §17.2 single mutation = optimization regime (batch+lr).
# §17.4 measurement re-entry (methodology framework).
# §18.8 emit_train_summary inherit.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export EXP_ID="H029_original_default_regime"

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
echo "[run.sh] mode=H029 original default regime (batch=256, lr=1e-4)"

# Diff vs H022/upload/run.sh:
#   --batch_size 2048 → 256       H029 NEW (H010 train.py default)
#   --lr 1e-4 (unchanged, but explicit at small batch)
# Wall: ~5-7h expected (8x more steps per epoch vs batch=2048).
# Risk: 너무 작은 batch → wall 길어짐. patience=3 + early stop 으로 wall
# 단축 가능.
python3 -u "${SCRIPT_DIR}/train.py" \
    --num_epochs 10 \
    --patience 3 \
    --seed 42 \
    --batch_size 256 \
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

echo "[run.sh] training complete (H029 original default regime); metrics at ${TRAIN_CKPT_PATH}/metrics.json"
