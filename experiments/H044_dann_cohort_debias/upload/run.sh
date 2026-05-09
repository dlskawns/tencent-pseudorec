#!/bin/bash
# Platform run.sh for H044_dann_cohort_debias.
# Data-signal-driven (CLAUDE.md §0.5): F-A pattern (OOF over Platform) 9 H
# 누적, H019/H038/H039 비교 후 cohort drift 가 platform transfer 실패의
# 근본 원인 confirm. H038 (BCE+aux MSE on log1p(timestamp)) Platform 0.839071
# vs H019 0.839674 = OOF +0.0012 → Platform −0.0006 (transfer fail). H044 =
# H038 architecture (action_num=2) 위 gradient reversal layer (Ganin &
# Lempitsky 2015 DANN) 추가 — backbone 이 timestamp signal *forget* 하도록
# 강제 → cohort-invariant feature 학습.
#
# §17.2 single mutation: trainer.py 에 GradReverse + dann_lambda branch 추가.
# model.py / dataset.py / infer.py / utils.py / make_schema.py / local_validate.py
# byte-identical to H038.
#
# Hypothesis docs: hypotheses/H044_dann_cohort_debias/.
# §18.7 + §18.8 H019/H038 carry.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export EXP_ID="H044_dann_cohort_debias"

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
echo "[run.sh] mode=H044 DANN cohort debias (GRL on aux timestamp, dann_lambda=0.5)"

# ---------------------------------------------------------------------------
# Train. Diff vs H038/upload/run.sh:
#   + --dann_lambda 0.5      H044 NEW: gradient reversal strength on aux path
# 다른 flag 모두 H038 byte-identical. aux_lambda 0.1 retained — H038 base.
# dann_lambda 0.5: 보수적 (Ganin & Lempitsky 권장 1.0 의 절반, sample-scale
# safety). 효과: backbone 이 timestamp 예측 못 하도록 강제, scale -0.05
# (-aux_lambda * dann_lambda).
# ---------------------------------------------------------------------------
python3 -u "${SCRIPT_DIR}/train.py" \
    --num_epochs 10 \
    --patience 3 \
    --seed 42 \
    --batch_size 1024 \
    --lr 1e-4 \
    --loss_type aux_timestamp \
    --action_num 2 \
    --aux_lambda 0.1 \
    --dann_lambda 0.5 \
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
    --log_attn_entropy \
    "$@"

echo "[run.sh] training complete (H044); metrics at ${TRAIN_CKPT_PATH}/metrics.json"
echo "[run.sh] check stdout above for §18.8 SUMMARY block + 'dann_lambda=0.5' GRL active log"
