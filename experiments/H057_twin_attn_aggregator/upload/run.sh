#!/bin/bash
# Platform run.sh for H057_twin_attn_aggregator.
# H019 TWIN aggregator: 4 per-domain (B, D) → uniform mean. §3.5 4 도메인 imbalance
# (domain d 8% empty, c outlier max 3894) → uniform mean 이 noise 큰 도메인도 동등.
# H057 = per-user per-domain attention weight (softmax) — user 의 active 한 도메인
# 중심 weighted aggregation. TWIN-internal mutation, **single residual 유지**
# (H048 fail mode 회피).
#
# §17.2 single mutation: TwinRetrievalAggregator class 에 use_attn 분기 추가.
# trainer.py / dataset.py / utils.py byte-identical to H019. infer.py 만 flag 추가.
#
# trainable params 추가: ~64 (Linear(D, 1)).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export EXP_ID="H057_twin_attn_aggregator"

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
echo "[run.sh] mode=H057 TWIN attention aggregator (4-domain mean → softmax weighted)"

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
    --use_twin_attn_aggregator \
    --log_attn_entropy \
    "$@"

echo "[run.sh] training complete (H057); metrics at ${TRAIN_CKPT_PATH}/metrics.json"
echo "[run.sh] check stdout above for §18.8 SUMMARY block + 'use_attn_aggregator=True' log"
