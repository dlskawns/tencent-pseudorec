#!/bin/bash
# Platform run.sh for H045_cross_domain_bridge.
# Data-signal-driven (CLAUDE.md §0.5): §3.5 4 도메인 disjoint vocab (jaccard
# 0.7~10%) — disjoint 한 vocab 이지만 0.7~10% overlap 존재. TWIN aggregator
# 는 per-domain 독립 처리 후 mean → cross-domain feature transfer 0. H045 =
# per-domain encoder output 사이 explicit MultiHeadAttention bridge → cross-
# domain semantic 흐름 추가.
#
# §17.2 single mutation: CrossDomainBridge 추가, residual ADD post-backbone
# (TWIN 직전). gate init sigmoid(-2)≈0.12 (§10.10). 다른 모든 부분 byte-
# identical to H019 (TWIN GSU+ESU + NS xattn + DCN-V2 unchanged). model.py
# 만 변경, trainer.py / dataset.py / infer.py / utils.py / make_schema.py /
# local_validate.py md5 verify byte-identical.
#
# Hypothesis docs: hypotheses/H045_cross_domain_bridge/.
# §18.7 + §18.8 H019 carry.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export EXP_ID="H045_cross_domain_bridge"

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
echo "[run.sh] mode=H045 cross-domain bridge attention (4 domains attend each other, gate=-2.0)"

# ---------------------------------------------------------------------------
# Train (H019 base + cross-domain bridge). Diff vs H019/upload/run.sh:
#   + --use_cross_domain_bridge          H045 NEW: enable bridge
#   + --cross_domain_num_heads 4         MultiheadAttention num_heads
#   + --cross_domain_gate_init -2.0      gate init sigmoid(-2)≈0.12 (§10.10)
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
    --use_cross_domain_bridge \
    --cross_domain_num_heads 4 \
    --cross_domain_gate_init -2.0 \
    --log_attn_entropy \
    "$@"

echo "[run.sh] training complete (H045); metrics at ${TRAIN_CKPT_PATH}/metrics.json"
echo "[run.sh] check stdout above for §18.8 SUMMARY block + 'H045 cross-domain bridge enabled' log"
