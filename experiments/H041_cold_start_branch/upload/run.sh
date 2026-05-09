#!/bin/bash
# Platform run.sh for H041_cold_start_branch.
# Data-signal-driven (CLAUDE.md §0.5, R2 reframe): target_in_history = 0.4%
# (§3.5) → 99.6% predictions are extrapolation. 18 H 모두 single classifier
# 로 두 regime (familiar 0.4% vs novel 99.6%) 동시 처리 — capacity 부족 가능.
# Mutation: dual classifier (main + cold) + per-sample learned gate.
# Init: gate bias=2.0 → sigmoid≈0.88 → main dominates (H019 등가 init).
# H019 의 다른 모든 부분 byte-identical (TWIN, NS xattn, DCN-V2, gate=-2.0).
#
# §3.5 EDA: seq p90=1393~2215 events vs current truncate 64-128 = 95%+ tail
# loss. H014 (dense 192 uniform) REFUTED → retrieval form 으로 다시 attack.
# 11 H ceiling 0.832~0.836 mechanism mutation 무관 → paradigm shift 정당.
#
# Mechanism (TWIN paper Tencent 2024 RecSys arXiv:2302.02352):
#   GSU: parameter-free inner-product score per history token vs candidate.
#   Top-K filter (K=64).
#   ESU: MultiHeadAttention with candidate Q, top-K K/V → (B, D).
#   4 per-domain (B, D) → mean → Linear → LayerNorm → residual ADD post-
#   backbone (output += sigmoid(gate) * twin_state). Gate init = sigmoid(-2.0)
#   ≈ 0.12 (CLAUDE.md §10.10 InterFormer mandate). num_ns 변경 없음 →
#   d_model%T 제약 안전.
#
# Differs from paper: (1) per-domain (not single long history), (2) top_k=64
# (paper 128 — sample-scale conservative), (3) GSU = simple inner product
# (parameter-free, §10.6 sample budget friendly).
#
# §17.2: 1 mutation = "TWIN GSU+ESU per-domain retrieval block 추가".
# H010 anchor (NS xattn + DCN-V2) + H023 envelope (bce explicit, batch=2048)
# byte-identical EXCEPT TWIN flags + seq_max_lens 64-128 → 256 (retrieval
# 의 enabling condition, top-K=64 가 의미 있는 더 긴 history 위에서).
#
# Hypothesis docs: hypotheses/H019_twin_long_seq_retrieval/.
# §18.7 label_time fill_null + §18.8 emit_train_summary inherit from H023.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export EXP_ID="H041_cold_start_branch"

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
echo "[run.sh] mode=H041 cold-start dual branch (R2 reframe — 99.6% extrapolation specialization)"

# ---------------------------------------------------------------------------
# Train (H010 mechanism + H023 envelope, single mutation = TWIN flags + seq cap).
# Diff vs H023/upload/run.sh:
#   + --use_twin_retrieval              H019 NEW: enable TWINBlock per domain
#   + --twin_top_k 64                   GSU top-K filter
#   + --twin_num_heads 4                ESU MHA num_heads
#   + --twin_gate_init -2.0             gate init sigmoid(-2)≈0.12 (§10.10)
#   * --seq_max_lens 256/256/256/256    cap expansion (was 64/64/128/128) for
#                                       retrieval to access more tail. May OOM
#                                       at batch=2048 → fallback batch=1024.
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
    --use_cold_start_branch \
    --cold_start_gate_init 2.0 \
    --log_attn_entropy \
    "$@"

echo "[run.sh] training complete (H041); metrics at ${TRAIN_CKPT_PATH}/metrics.json"
echo "[run.sh] check stdout above for §18.8 SUMMARY block + 'H041 cold-start dual branch enabled' log"
