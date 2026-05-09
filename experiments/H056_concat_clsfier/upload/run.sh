#!/bin/bash
# Platform run.sh for H056_concat_clsfier.
# H048 LESSON: 2 residual ADD on output (cross + twin) → cohort transfer 깨짐
# (Val/OOF +0.0003 → Platform −0.0042). H056 = same goal (user × item interaction)
# *different mechanism layer*: NOT residual ADD, BUT classifier input concat.
#   - output 자체 unchanged (TWIN residual 그대로, no stacking)
#   - clsfier 의 input dim D → 2D (concat with item_repr=item_ns mean)
#   - item info 가 clsfier 에 직접 도달
#
# §17.2 single mutation: clsfier input dim 변경 + forward concat.
# trainer.py / dataset.py / utils.py byte-identical to H019.
# infer.py 만 use_concat_clsfier flag (H043 사고 방지).
#
# trainable params 추가: ~4K (Linear(2D, D) vs Linear(D, D) = 4096 extra).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export EXP_ID="H056_concat_clsfier"

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
echo "[run.sh] mode=H056 concat-input clsfier (clsfier_input_dim=2D, NO residual stacking)"

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
    --use_concat_clsfier \
    --log_attn_entropy \
    "$@"

echo "[run.sh] training complete (H056); metrics at ${TRAIN_CKPT_PATH}/metrics.json"
echo "[run.sh] check stdout above for §18.8 SUMMARY block + 'H056 concat-input clsfier' log"
