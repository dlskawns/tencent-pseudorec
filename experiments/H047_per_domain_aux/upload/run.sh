#!/bin/bash
# Platform run.sh for H047_per_domain_aux.
# Data-signal-driven (CLAUDE.md §0.5): 14 H 동안 main classifier 1개 만 사용,
# per-domain prediction multi-task 0회. §3.5 4 도메인 disjoint vocab signal —
# 각 도메인이 독립적 prediction 신호 가질 가능성. backbone 이 4 도메인 정보
# 를 *대표 1개 logit* 로 압축 → per-domain signal mixing 손실 가능.
# H047 = 4 per-domain auxiliary classifier heads — 각 도메인 pooled seq 로
# 같은 main label 예측 → backbone 이 balanced per-domain prediction 신호 학습.
#
# §17.2 single mutation: model.py 에 4 per-domain heads 추가, forward 가
# (logits, per_domain_aux_list) tuple 반환. trainer.py 가 tuple 처리 + aux BCE.
# dataset.py / infer.py / utils.py / make_schema.py / local_validate.py
# byte-identical to H019.
#
# Hypothesis docs: hypotheses/H047_per_domain_aux/.
# §18.7 + §18.8 H019 carry.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export EXP_ID="H047_per_domain_aux"

# ---------------------------------------------------------------------------
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
echo "[run.sh] mode=H047 per-domain aux heads (4 heads, multi-task BCE, aux_weight=0.25)"

# ---------------------------------------------------------------------------
# Train (H019 base + per-domain aux heads). Diff vs H019/upload/run.sh:
#   + --use_per_domain_aux               H047 NEW: enable 4 per-domain heads
# 다른 flag 모두 H019 byte-identical.
# ---------------------------------------------------------------------------
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
    --use_per_domain_aux \
    --log_attn_entropy \
    "$@"

echo "[run.sh] training complete (H047); metrics at ${TRAIN_CKPT_PATH}/metrics.json"
echo "[run.sh] check stdout above for §18.8 SUMMARY block + 'H047 per-domain aux heads enabled' log"
