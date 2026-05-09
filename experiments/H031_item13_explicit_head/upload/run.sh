#!/bin/bash
# Platform run.sh for H031_item13_explicit_head.
# Single mutation vs H023 baseline: --use_item13_user_cross flag enables
# Item13UserCrossBlock — item_int_feats_13 (vs=10, demo_1000 univariate
# GBDT 5-fold AUC 0.6561, i13=4 → 64% positive +51pt lift) dedicated
# 32-dim embedding × 5 small-vocab user_int (fids 1, 49, 58, 95, 97 — all
# vs ≤ 6) dedicated 16-dim embeddings, outer-product cross → mean →
# 32-dim → Linear → LayerNorm → (B, d_model) cross_state. Residual ADD
# post-backbone: output += sigmoid(gate) * cross_state. Gate init = -2.0
# (sigmoid ≈ 0.12) per CLAUDE.md §10.10 InterFormer mandate.
#
# §17.2: 1 mutation = "fid-level explicit cross block 추가". H010 anchor
# (NS xattn + DCN-V2) + H023 envelope (bce explicit, batch=2048) byte-
# identical EXCEPT --use_item13_user_cross flag.
#
# EDA reference: eda/out/item_int_signal_audit.json (this session 2026-05-04).
# Hypothesis docs: hypotheses/H031_item13_explicit_head/ (problem, transfer,
# predictions, verdict).
# §18.8 emit_train_summary inherit from H023.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export EXP_ID="H031_item13_explicit_head"

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
echo "[run.sh] mode=H031 item_13 explicit head + user_int outer-product cross"

# ---------------------------------------------------------------------------
# Train (H010 mechanism + H023 envelope, single mutation = --use_item13_user_cross).
# Diff vs H023/upload/run.sh:
#   + --use_item13_user_cross           H031 NEW: enable Item13UserCrossBlock
#   + --item13_emb_dim 32               item_13 dedicated embedding dim
#   + --item13_user_emb_dim 16          per-user_int dedicated embedding dim
#   + --item13_cross_dim 32             pair-wise cross output dim
#   + --item13_gate_init -2.0           gate init sigmoid(-2) ≈ 0.12 (§10.10)
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
    --use_item13_user_cross \
    --item13_emb_dim 32 \
    --item13_user_emb_dim 16 \
    --item13_cross_dim 32 \
    --item13_gate_init -2.0 \
    --log_attn_entropy \
    "$@"

echo "[run.sh] training complete (H031); metrics at ${TRAIN_CKPT_PATH}/metrics.json"
echo "[run.sh] check stdout above for §18.8 SUMMARY block + Item13UserCrossBlock log line"
