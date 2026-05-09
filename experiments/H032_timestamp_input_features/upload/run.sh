#!/bin/bash
# Platform run.sh for H032_timestamp_input_features.
# Single mutation vs H023 baseline: --use_timestamp_features flag enables
# TimeFeaturesBlock — 3 derived categorical features (hour_of_day=24,
# day_of_week=7, recency_bucket=8 from log2(timestamp - max_seq_ts)) →
# 16-dim emb sum → LayerNorm → Linear → (B, d_model) time_state. Residual
# ADD post-backbone: output += sigmoid(gate) * time_state. Gate init = -2.0
# (sigmoid ≈ 0.12) per CLAUDE.md §10.10 InterFormer mandate.
#
# Differentiation from H015~H018 (loss-axis): H032 = input feature axis
# (model.forward 가 per-sample temporal context 직접 학습). H015~H018 모두
# loss reweight axis (per-batch / per-user gradient magnitude). H032 가
# input-axis temporal_input 카테고리 첫 진입.
#
# §17.2: 1 mutation = "timestamp-derived 3 categorical features → 1 NS-token
# state → residual ADD post-backbone". H010 anchor (NS xattn + DCN-V2) +
# H023 envelope (bce explicit, batch=2048) byte-identical EXCEPT
# --use_timestamp_features flag.
#
# EDA reference: eda/out/timestamp_signal_audit.json (this session 2026-05-04).
# Hypothesis docs: hypotheses/H032_timestamp_input_features/.
# §18.7 timestamp.fill_null(0) defensive applied (dataset.py).
# §18.8 emit_train_summary inherit from H023.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export EXP_ID="H032_timestamp_input_features"

# ---------------------------------------------------------------------------
# Platform must set TRAIN_DATA_PATH and TRAIN_CKPT_PATH. Other paths derive.
# Caller may pass --seed 42 (and any other override) via "$@".
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
echo "[run.sh] mode=H032 timestamp input features (hour/dow/recency residual ADD)"

# ---------------------------------------------------------------------------
# Train (H010 mechanism + H023 envelope, single mutation = --use_timestamp_features).
# Diff vs H023/upload/run.sh:
#   + --use_timestamp_features          H032 NEW: enable TimeFeaturesBlock
#   + --time_emb_dim 16                 per-feature time embedding dim
#   + --time_gate_init -2.0             gate init sigmoid(-2) ≈ 0.12 (§10.10)
# ---------------------------------------------------------------------------
python3 -u "${SCRIPT_DIR}/train.py" \
    --num_epochs 10 \
    --patience 3 \
    --seed 42 \
    --batch_size 2048 \
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
    --seq_max_lens "seq_a:64,seq_b:64,seq_c:128,seq_d:128" \
    --use_label_time_split \
    --oof_user_ratio 0.1 \
    --split_seed 42 \
    --fusion_type dcn_v2 \
    --dcn_v2_num_layers 2 \
    --dcn_v2_rank 8 \
    --use_ns_to_s_xattn \
    --ns_xattn_num_heads 4 \
    --use_timestamp_features \
    --time_emb_dim 16 \
    --time_gate_init -2.0 \
    --log_attn_entropy \
    "$@"

echo "[run.sh] training complete (H032); metrics at ${TRAIN_CKPT_PATH}/metrics.json"
echo "[run.sh] check stdout above for §18.8 SUMMARY block + TimeFeaturesBlock log line"
