#!/bin/bash
# Platform run.sh for H043_item_side_dcnv2.
# Data-signal-driven (CLAUDE.md §0.5): 14 H 동안 item-side modeling 전무 —
# item_int 13 scalar + 1 array 가 nn.Embedding lookup 후 NS-token 으로 직행,
# explicit feature interaction 0회. user-side DCN-V2 (H008) PASS +0.0035pt
# 의 mechanism 을 item-side 에 first 적용. §3.5 4 도메인 disjoint vocab +
# target_in_history=0.4% 가 item 측 cross-domain semantic 의 첫 motivation.
#
# §17.2 single mutation: item_ns 토큰에 DCN-V2 polynomial cross 적용.
# (item_side_cross: layers=2, rank=4, low-rank Pre-LN). 다른 모든 부분
# byte-identical to H019 (TWIN GSU+ESU + NS xattn + DCN-V2 user-side
# unchanged). model.py 만 변경, trainer.py / dataset.py / infer.py /
# utils.py / make_schema.py / local_validate.py md5 verify byte-identical.
#
# Hypothesis docs: hypotheses/H043_item_side_dcnv2/.
# §18.7 + §18.8 H019 carry.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export EXP_ID="H043_item_side_dcnv2"

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
echo "[run.sh] mode=H043 item-side DCN-V2 cross (layers=2, rank=4, on item_ns tokens)"

# ---------------------------------------------------------------------------
# Train (H019 base + item-side DCN-V2 cross). Diff vs H019/upload/run.sh:
#   + --use_item_side_cross              H043 NEW: enable item_side_cross
#   + --item_side_cross_layers 2         num cross layers
#   + --item_side_cross_rank 4           low-rank
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
    --use_item_side_cross \
    --item_side_cross_layers 2 \
    --item_side_cross_rank 4 \
    --log_attn_entropy \
    "$@"

echo "[run.sh] training complete (H043); metrics at ${TRAIN_CKPT_PATH}/metrics.json"
echo "[run.sh] check stdout above for §18.8 SUMMARY block + 'H043 item-side DCN-V2 cross enabled' log"
