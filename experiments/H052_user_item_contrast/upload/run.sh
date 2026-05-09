#!/bin/bash
# Platform run.sh for H052_user_item_contrast.
# Data-signal-driven (CLAUDE.md §0.5): 14 H 모두 pointwise BCE 단독 supervision.
# H040 BPR (logit pairwise) REFUTED. H052 = REPRESENTATION-level contrastive
# (InfoNCE) — backbone output user_repr 와 item_ns mean item_repr 사이 in-batch
# negatives. positive = (user_i, item_i) same row, negative = (user_i, item_j≠i).
# user_repr 가 자기 item 을 다른 user 의 item 보다 더 가깝게 학습 강제.
# 14 H 동안 representation-level contrastive 0회 — SSL-style new supervision.
#
# §17.2 single mutation: model.py 가 (logits, user_repr, item_repr) tuple 반환,
# trainer.py 가 InfoNCE auxiliary loss 추가. dataset.py / utils.py /
# make_schema.py / local_validate.py byte-identical to H019. infer.py 만
# use_user_item_contrast flag 추가 (parity, H043 사고 방지).
#
# trainable params 추가: 0 (loss term only, no new module).
# contrast_lambda=0.1 (보수적), contrast_temperature=0.1 (sharpening 강함).
#
# Hypothesis docs: hypotheses/H052_user_item_contrast/.
# §18.7 + §18.8 H019 carry.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export EXP_ID="H052_user_item_contrast"

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
echo "[run.sh] mode=H052 user-item InfoNCE contrastive aux (lambda=0.1, tau=0.1)"

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
    --use_user_item_contrast \
    --contrast_lambda 0.1 \
    --contrast_temperature 0.1 \
    --log_attn_entropy \
    "$@"

echo "[run.sh] training complete (H052); metrics at ${TRAIN_CKPT_PATH}/metrics.json"
echo "[run.sh] check stdout above for §18.8 SUMMARY block + 'H052 user-item contrastive' log"
