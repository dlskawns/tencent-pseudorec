#!/bin/bash
# Platform run.sh for H046_proper_dann.
# Data-signal-driven (CLAUDE.md §0.5): F-A 패턴 (OOF over Platform 9 H 누적)
# + H019/H038/H039 platform 비교 cohort drift = transfer 실패 root cause
# 정량 confirm. H044 (GRL on aux raw_logits col) **design 결함** — clsfier-aux
# head + backbone 둘 다 reverse → no actual discriminator → loss 발산
# (epoch 1 = 4.93, epoch 5 = 545). H046 = **proper DANN**: separate cohort_head
# (positive grad, learns to predict timestamp) + GRL between backbone and
# cohort_head (backbone reverses, forgets cohort).
#
# §17.2 single mutation: model.py 에 _GradReverse + cohort_head 추가, forward
# 가 (logits, cohort_pred) tuple 반환. trainer.py 가 tuple 처리 + cohort MSE.
# dataset.py / infer.py / utils.py / make_schema.py / local_validate.py
# byte-identical to H019.
#
# Hypothesis docs: hypotheses/H046_proper_dann/.
# §18.7 + §18.8 H019 carry.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export EXP_ID="H046_proper_dann"

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
echo "[run.sh] mode=H046 proper DANN (separate cohort_head + GRL, dann_cohort_lambda=0.1)"

# ---------------------------------------------------------------------------
# Train (H019 base + proper DANN). Diff vs H019/upload/run.sh:
#   + --use_dann_cohort                  H046 NEW: enable cohort head + GRL
#   + --dann_cohort_lambda 0.1           GRL strength (보수적 — H044 0.5 의 1/5)
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
    --use_dann_cohort \
    --dann_cohort_lambda 0.1 \
    --log_attn_entropy \
    "$@"

echo "[run.sh] training complete (H046); metrics at ${TRAIN_CKPT_PATH}/metrics.json"
echo "[run.sh] check stdout above for §18.8 SUMMARY block + 'H046 proper DANN cohort head enabled' log"
