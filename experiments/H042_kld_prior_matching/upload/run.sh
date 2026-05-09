#!/bin/bash
# Platform run.sh for H042_kld_prior_matching.
# Data-signal-driven (CLAUDE.md §0.5): H038 (BCE+aux MSE) result OOF Δ +0.0012,
# Val Δ +0.0001 — output-distribution-level supervision axis 작동 first signal.
# H042 = H038 base + KLD prior matching: per-sample Bernoulli KL(σ(logit)‖prior).
# §3.4 class prior 0.124 (12.4% positive). Pereyra et al. 2017 confidence
# penalty form — 사용자 §0.5 의 shift-scheduling KLD pattern 의 binary 적용.
# Init kld_lambda=0.0 시 H038 byte-identical (safe carrier).
# H019/H038 의 다른 모든 부분 byte-identical (TWIN, NS xattn, DCN-V2, MSE aux).
#
# §17.2 mutation: trainer.py KLD term 추가 (kld_lambda > 0 시 활성).
# trainer.py + train.py + run.sh 만 변경. model.py / dataset.py / infer.py /
# utils.py byte-identical to H038 (md5 verify).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export EXP_ID="H042_kld_prior_matching"

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
echo "[run.sh] mode=H042 KLD prior matching (H038 base + λ_kld=0.01, prior=0.124)"

# ---------------------------------------------------------------------------
# Train. Diff vs H038/upload/run.sh:
#   + --kld_lambda 0.01           H042 NEW: KLD prior matching weight
#   + --kld_target_prior 0.124    §3.4 class prior (12.4% positive)
# 다른 flag 모두 H038 byte-identical.
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
    --kld_lambda 0.01 \
    --kld_target_prior 0.124 \
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

echo "[run.sh] training complete (H042); metrics at ${TRAIN_CKPT_PATH}/metrics.json"
echo "[run.sh] check stdout above for §18.8 SUMMARY block + KLD activation log"
