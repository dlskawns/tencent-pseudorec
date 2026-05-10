#!/bin/bash
# Platform run.sh for H058_onetrans_twin.
# PARADIGM SHIFT (backbone class): 14 H 모두 HyFormer 위 mutation. H058 =
# OneTrans backbone (single-stream attention) + TWIN retrieval. H026 (OneTrans
# 단독, no TWIN) val 0.8330 = HyFormer underperformed → H058 = OneTrans + TWIN
# 으로 H026 의 question (backbone class swap underperform 진짜 원인이 backbone
# 인지 missing TWIN 인지) 분리 측정.
#
# §17.2 single mutation: --backbone hyformer → onetrans. CLI only, 모든 .py
# byte-identical to H019.
#
# §17.4 backbone_replacement re-entry (H026 first-touch + TWIN add).
# Decision tree post-result:
#   - Δ vs H019 ≥ +0.001pt → OneTrans+TWIN 가 HyFormer+TWIN 보다 강함, paradigm shift confirm
#   - Δ vs H019 < +0.001pt && > H026 (0.8330 val) → TWIN add 가 큰 lift, backbone 부분 무관
#   - Δ vs H019 << 0 → H026 underperform 의 진짜 원인 = backbone class 자체
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export EXP_ID="H058_onetrans_twin"

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
echo "[run.sh] mode=H058 OneTrans backbone + TWIN retrieval"

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
    --backbone onetrans \
    --fusion_type dcn_v2 \
    --dcn_v2_num_layers 2 \
    --dcn_v2_rank 8 \
    --use_ns_to_s_xattn \
    --ns_xattn_num_heads 4 \
    --use_twin_retrieval \
    --twin_top_k 64 \
    --twin_num_heads 4 \
    --twin_gate_init -2.0 \
    --log_attn_entropy \
    "$@"

echo "[run.sh] training complete (H058); metrics at ${TRAIN_CKPT_PATH}/metrics.json"
echo "[run.sh] check stdout above for §18.8 SUMMARY block + 'Backbone: onetrans' indicator"
