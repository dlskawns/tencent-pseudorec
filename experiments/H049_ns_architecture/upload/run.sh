#!/bin/bash
# Platform run.sh for H049_ns_architecture.
# Data-signal-driven (CLAUDE.md §0.5): 14 H 동안 NS architecture 자체 0회 변경.
# - item_ns_tokens 모든 H 에서 default=2 (item 이 prediction target 인데
#   representation slot 부족)
# - NS slot type 차별 embedding 0개 (backbone 이 user_ns / item_ns / dense
#   role 학습으로 알아야 함)
# H049 = NS architecture refinement: (a) item_ns_tokens 2→6 (capacity), (b)
#   slot-type embedding (4 types: user_ns/user_dense/item_ns/item_dense).
#   둘 다 NS structure 변경 = single concept mutation.
#
# §17.2 single mutation: model.py 에 ns_type_emb + buffer ns_type_ids 추가,
# _build_token_streams 에서 broadcast ADD. trainer.py / dataset.py / utils.py
# / make_schema.py / local_validate.py byte-identical to H019. infer.py 만
# use_ns_slot_type_emb flag 추가 (H043 사고 방지).
#
# Hypothesis docs: hypotheses/H049_ns_architecture/.
# §18.7 + §18.8 H019 carry.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export EXP_ID="H049_ns_architecture"

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
echo "[run.sh] mode=H049 NS architecture refinement (item_ns 2→6, slot type emb)"

# Diff vs H019/upload/run.sh:
#   * --item_ns_tokens 2 → 6     H049: item NS expansion
#   + --use_ns_slot_type_emb     H049: 4-type embedding for NS slots

python3 -u "${SCRIPT_DIR}/train.py" \
    --num_epochs 10 \
    --patience 3 \
    --seed 42 \
    --batch_size 1024 \
    --lr 1e-4 \
    --loss_type bce \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 6 \
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
    --use_ns_slot_type_emb \
    --log_attn_entropy \
    "$@"

echo "[run.sh] training complete (H049); metrics at ${TRAIN_CKPT_PATH}/metrics.json"
echo "[run.sh] check stdout above for §18.8 SUMMARY block + 'H049 NS slot type emb enabled' log"
