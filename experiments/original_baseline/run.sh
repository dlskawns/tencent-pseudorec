#!/bin/bash
# Smoke run.sh for original_baseline + H001 leakage-fix patches.
#
# Equivalent envelope to E_baseline_organizer (val_AUC=0.8251) BUT with
# label_time-aware split + 10% user OOF holdout (CLAUDE.md §4.3 + §4.4)
# instead of organizer's row-group split. Exposes whether the row-group
# split was leaky — if val drops sharply (e.g., 0.85 → 0.55) we know prior
# anchor measurements were inflated.
#
# Backbone defaults to hyformer (PCVRHyFormer original). H004 OneTrans path
# in model.py is dormant.
#
# Override at runtime, e.g.:
#     bash run.sh --train_ratio 0.3 --num_epochs 5
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

# ---------------------------------------------------------------------------
# Platform must set TRAIN_DATA_PATH and TRAIN_CKPT_PATH. Other paths derive.
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
echo "[run.sh] mode=organizer-pure-smoke + label_time split + OOF holdout"

# ---------------------------------------------------------------------------
# Train (organizer-pure baseline + smoke envelope + H001 leakage fix).
# Diff vs E_baseline_organizer/run.sh:
#   --use_label_time_split    NEW: train/val split by label_time cutoff
#   --oof_user_ratio 0.1      NEW: 10% user_id held out (seed=42)
#   --split_seed 42           NEW: OOF user sampling seed
#
# Expected smoke val AUC: lower than 0.8251 (probably 0.55–0.70) — that lower
# number is the *honest* generalization estimate. If it stays at 0.85 even
# with label_time split, leakage is elsewhere (DUP_USER_POLICY, schema, etc.).
# ---------------------------------------------------------------------------
python3 -u "${SCRIPT_DIR}/train.py" \
    --num_epochs 1 \
    --patience 5 \
    --seed 42 \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --ns_groups_json "" \
    --emb_skip_threshold 1000000 \
    --num_workers 2 \
    --buffer_batches 4 \
    --train_ratio 0.05 \
    --seq_max_lens "seq_a:64,seq_b:64,seq_c:128,seq_d:128" \
    --use_label_time_split \
    --oof_user_ratio 0.1 \
    --split_seed 42 \
    "$@"

echo "[run.sh] training complete (organizer-pure-smoke + leak-fix); metrics at ${TRAIN_CKPT_PATH}/metrics.json"
