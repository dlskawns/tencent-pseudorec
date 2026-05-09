#!/bin/bash
# Platform run.sh for H039_no_history_baseline.
# Diagnostic H — DOES NOT propose new mechanism, instead REMOVES all 14 H의
# mechanism additions (DCN-V2, NS xattn, TWIN) from H019 to measure the
# bare PCVRHyFormer + RankMixer NS tokenizer baseline.
#
# Purpose (CLAUDE.md §0.5 data-signal-driven):
# - Measure absolute floor of our backbone family without any mechanism mutation.
# - Compare with H019 (0.83967 platform AUC) to quantify total mechanism contribution.
# - Result interpretation:
#   * Baseline ≈ 0.84 → 14 H mechanism work is essentially NOOP, paradigm pivot mandatory.
#   * Baseline ~0.81 → mechanism work contributed +0.025pt, optimization continues on this axis.
#
# §17.2: 1 mutation = "remove all mechanism flags". 다른 모든 부분 H019 byte-identical
# (model.py / dataset.py / trainer.py / infer.py 코드 그대로). 변경 = run.sh flag 만.
# §17.4: diagnostic H, mechanism category 무관 (NEW first-touch, AUTO_JUSTIFIED).
# §18.7 label_time fill_null + §18.8 emit_train_summary inherit from H019.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export EXP_ID="H039_no_history_baseline"

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
echo "[run.sh] mode=H039 organizer-bare PCVRHyFormer (no DCN-V2 / no NS xattn / no TWIN — diagnostic)"

# ---------------------------------------------------------------------------
# Train (H019 envelope - mechanism flags). Diff vs H019/upload/run.sh:
#   - REMOVED --fusion_type dcn_v2          (default 'rankmixer')
#   - REMOVED --dcn_v2_num_layers / rank
#   - REMOVED --use_ns_to_s_xattn           (no H010 NS→S xattn)
#   - REMOVED --ns_xattn_num_heads
#   - REMOVED --use_twin_retrieval          (no H019 TWIN GSU+ESU)
#   - REMOVED --twin_top_k / num_heads / gate_init
#   - REMOVED --log_attn_entropy
# Kept: H019 와 같은 batch=1024, lr=1e-4, seq 256/256/256/256, label_time split, OOF holdout, RankMixer NS tokenizer.
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
    "$@"

echo "[run.sh] training complete (H039); metrics at ${TRAIN_CKPT_PATH}/metrics.json"
echo "[run.sh] check stdout above for §18.8 SUMMARY block + bare baseline indicators"
