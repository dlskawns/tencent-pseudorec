#!/bin/bash
# Platform run.sh for H053_simcse_user_contrast.
# PARADIGM SHIFT: 14 H 모두 single supervision (label-only). H053 = SimCSE
# (Gao 2021) style SSL self-contrastive on backbone output user_repr — 2 dropout
# views → InfoNCE in-batch positive=(z1_i,z2_i), negative=(z1_i,z2_j≠i).
# Forces user representation invariance to dropout noise = SSL paradigm.
# 14 H 동안 SSL self-supervision 0회.
#
# §17.2: model.py forward tuple return + trainer.py SimCSE loss term. dataset.py /
# infer.py 외 byte-identical to H019. infer.py 만 use_user_simcse flag (parity).
# trainable params 추가: 0.
#
# simcse_lambda=0.1, temperature=0.1, dropout=0.1.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export EXP_ID="H053_simcse_user_contrast"

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
echo "[run.sh] mode=H053 SimCSE user self-contrastive (lambda=0.1, tau=0.1, dropout=0.1)"

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
    --use_user_simcse \
    --simcse_lambda 0.1 \
    --simcse_temperature 0.1 \
    --simcse_dropout 0.1 \
    --log_attn_entropy \
    "$@"

echo "[run.sh] training complete (H050); metrics at ${TRAIN_CKPT_PATH}/metrics.json"
echo "[run.sh] check stdout above for §18.8 SUMMARY block + 'H053 SimCSE' log"
