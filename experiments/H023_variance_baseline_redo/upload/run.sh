#!/bin/bash
# Platform run.sh for H023_variance_baseline_redo.
# H022 redo with bug-fix + explicit loss_type bake.
# - H022 produced anomalous result: best_val 0.8322 (low), best_epoch 2/5 (very early),
#   OOF=N/A. Three issues:
#     (1) loss_type ambiguity — H022 default = bce. H011~H018 train_loss scale 다름
#         (focal vs bce vs weighted bce). H010 corrected anchor 0.837806 의 실측 loss 도
#         확실 안 함. H023 = bce explicit bake (default 와 동일하지만 명시적으로 reproducibility).
#     (2) OOF=N/A — 어느 path 가 fail 했는지 모름. H023 train.py 에 [H023 DEBUG] 진단 print
#         추가: oof_loader/best_model None 여부 + try/except 로 raise message 출력.
#     (3) early stop ep5 (5 of 10 epochs) — patience=3 + best_epoch=2 면 ep5 trigger 정상.
#         단 best_val 자체 anomalous → train regime 차이 신호.
# §17.2 EXEMPT (measurement H, no mutation). §17.4 measurement re-entry justified by
# H022 anomaly + variance baseline framework prerequisite for paradigm shift decisions.
# §18.8 emit_train_summary inherit from H022.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export EXP_ID="H023_variance_baseline_redo"   # §18.8 SUMMARY identity

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
echo "[run.sh] mode=H023 variance baseline redo (H022 + loss_type bce explicit + OOF debug)"

# ---------------------------------------------------------------------------
# Train (H010 envelope + H010 mechanism, seed via "$@" override).
# Diff vs H022/upload/run.sh:
#   + --loss_type bce              H023 NEW: explicit bake (avoid default ambiguity)
#   + train.py [H023 DEBUG] prints around OOF eval (oof_loader, best_model, raises)
# Recommended invocation per seed:
#   bash run.sh --seed 42
#   bash run.sh --seed 43
#   bash run.sh --seed 44
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
    --log_attn_entropy \
    "$@"

echo "[run.sh] training complete (H023); metrics at ${TRAIN_CKPT_PATH}/metrics.json"
echo "[run.sh] check stdout above for [H023 DEBUG] lines + §18.8 SUMMARY block"
