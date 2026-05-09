#!/bin/bash
# Platform run.sh for H048_user_item_bilinear.
# Data-signal-driven (CLAUDE.md §0.5): 14 H 분석 — PASS mechanism (H010 NS→S
# xattn, H019 TWIN candidate Q) 모두 *새 정보 흐름* 추가. 14 H 동안 user
# representation 과 item representation 사이 explicit bilinear interaction 0회.
# H019 TWIN 은 item attends history (item → user_seq), DCN-V2 는 within-side
# cross. user × item DIRECT interaction NEW pathway.
#
# §17.2 single mutation: model.py 에 UserItemBilinearCross class + flag 추가,
# forward 의 backbone output 에 residual ADD (TWIN 직전).
# trainer.py / dataset.py / infer.py 마지막 빼고 byte-identical to H019.
# infer.py 만 use_user_item_cross flag 추가 (H043 사고 재발 방지).
#
# Hypothesis docs: hypotheses/H048_user_item_bilinear/.
# §18.7 + §18.8 H019 carry.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export EXP_ID="H048_user_item_bilinear"

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
echo "[run.sh] mode=H048 user × item bilinear cross (gate sigmoid(-2)=0.12)"

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
    --use_user_item_cross \
    --user_item_cross_gate_init -2.0 \
    --log_attn_entropy \
    "$@"

echo "[run.sh] training complete (H048); metrics at ${TRAIN_CKPT_PATH}/metrics.json"
echo "[run.sh] check stdout above for §18.8 SUMMARY block + 'H048 user-item bilinear cross enabled' log"
