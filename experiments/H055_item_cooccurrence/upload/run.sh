#!/bin/bash
# Platform run.sh for H055_item_cooccurrence.
# PARADIGM SHIFT: 14 H 모두 single supervision. H055 = Cross-domain pool
# InfoNCE — 같은 user 의 2 도메인 pool 은 같아야 (positive), 다른 user 의
# pool 은 달라야 (negative). 4 도메인 jaccard 0.7~10% (§3.5) 직접 attack:
# 동일 user 가 4 도메인에서 같은 user 정체성 보유 → cross-domain consistency
# 학습. NEW data-side SSL form, 14 H 동안 0회.
#
# §17.2: model.py forward tuple return + trainer.py InfoNCE. dataset.py /
# utils.py / make_schema.py / local_validate.py byte-identical to H019.
# infer.py 만 use_cross_domain_contrast flag (parity, H043 방지).
#
# trainable params: 0 (loss term only).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export EXP_ID="H055_item_cooccurrence"

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
echo "[run.sh] mode=H055 cross-domain pool InfoNCE (lambda=0.1, tau=0.1)"

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
    --seq_max_lens "seq_a:256,seq_b:256,seq_c:256,seq_d:256" \
    --use_label_time_split \
    --oof_user_ratio 0.1 \
    --split_seed 42 \
    --fusion_type dcn_v2 \
    --dcn_v2_num_layers 2 \
    --dcn_v2_rank 8 \
    --use_ns_to_s_xattn \
    --ns_xattn_num_heads 4 \
    --use_twin_retrieval \
    --twin_top_k 64 \
    --twin_num_heads 4 \
    --twin_gate_init -2.0 \
    --use_cross_domain_contrast \
    --cross_domain_contrast_lambda 0.1 \
    --cross_domain_contrast_temperature 0.1 \
    --log_attn_entropy \
    "$@"

echo "[run.sh] training complete (H053); metrics at ${TRAIN_CKPT_PATH}/metrics.json"
echo "[run.sh] check stdout above for §18.8 SUMMARY block + H055 cross-domain log"
