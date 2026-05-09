#!/bin/bash
# Platform run.sh for H022_h010_multi_seed_variance.
# Measurement H — H010 byte-identical mechanism, seed only varies (42/43/44).
# Output: 3 platform AUC values → mean ± stdev → variance baseline for all
# subsequent paired Δ comparisons.
# §17.2 EXEMPT (measurement H, no mutation). §17.4 measurement category first-touch.
# §18.8 emit_train_summary added (3 SUMMARY blocks per launch).
# Diff vs H010/upload/run.sh:
#   + EXP_ID env (SUMMARY block identity)
#   + --batch_size 2048 + --lr 1e-4 (H010 corrected 0.837806 의 실측 regime —
#     INDEX F-2: 사용자가 batch=2048 + lr=1e-4 로 override 해서 H006~H012/H015 모두 같은 regime.
#     H010 run.sh defaults (batch=256) 는 한 번도 안 사용됐음. variance baseline 의 정합성 위해
#     H022 도 동일 regime explicit bake.)
#   + caller invokes 3 times with --seed 42 / 43 / 44 (parallel or serial)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export EXP_ID="H022_h010_multi_seed_variance"   # §18.8 SUMMARY identity

# ---------------------------------------------------------------------------
# Platform must set TRAIN_DATA_PATH and TRAIN_CKPT_PATH. Other paths derive.
# Caller passes --seed 42|43|44 (and any other override) via "$@".
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
echo "[run.sh] mode=H022 measurement (H010 byte-identical, seed only)"

# ---------------------------------------------------------------------------
# Train (H010 envelope + H010 mechanism, seed via "$@" override).
# Recommended invocation per seed:
#   bash run.sh --seed 42
#   bash run.sh --seed 43
#   bash run.sh --seed 44
# Each launches in its own ckpt dir. Launch parallel (3 GPU/slot) for ~3.5h
# total wall, or serial for ~10.5h.
# ---------------------------------------------------------------------------
python3 -u "${SCRIPT_DIR}/train.py" \
    --num_epochs 10 \
    --patience 3 \
    --seed 42 \
    --batch_size 2048 \
    --lr 1e-4 \
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
    --log_attn_entropy \
    "$@"

echo "[run.sh] training complete (H022 measurement); metrics at ${TRAIN_CKPT_PATH}/metrics.json"
echo "[run.sh] §18.8 SUMMARY block above (copy block bracketed by '==== TRAIN SUMMARY' / '==== END SUMMARY')"
