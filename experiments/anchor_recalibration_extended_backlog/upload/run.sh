#!/bin/bash
# Platform run.sh for H010_anchor_recalibration_extended.
# Anchor recalibration H — mechanism mutation 0, envelope mutation only.
# original_baseline byte-identical 코드 + envelope flags 3개 변경:
#   --num_epochs 1   → 10        (smoke 1 → extended 10)
#   --train_ratio    → 0.3       (smoke 0.05 → extended 0.3)
#   --patience 5     → 3         (smoke 5 → extended 3, H008 F-4 carry-forward)
#
# H007 verdict F-3 직접 충족 ("anchor 도 extended 에서 측정 필요").
# H009 verdict F-3 정량 동기 (anchor 정확값 의존성이 결론 분류 흔듦).
# §17.3 binary 임계 미적용 — measurement objective.
# §18 inference 인프라 룰 inherit (original_baseline 패키지에 이미 포함).
#
# 12 file upload package: original_baseline/upload/ 의 12 파일 byte-identical
# 카피 + 본 run.sh + README.md (H010 정체성). 다음 turn 빌드 예정.
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
echo "[run.sh] mode=anchor-recalibration-extended (mechanism mutation 0, envelope only)"

# ---------------------------------------------------------------------------
# Train (original_baseline pure baseline + extended envelope).
# Diff vs original_baseline/run.sh:
#   --num_epochs 1 → 10               extended (paired with H006/H007/H008/H009)
#   --train_ratio 0.05 → 0.3          extended
#   --patience 5 → 3                  H008 F-4 carry-forward (early stop aggressive)
#
# 그 외 모든 args byte-identical. mechanism flags (use_candidate_summary_token,
# fusion_type, dcn_v2_*) 어떤 것도 추가 안 함. PCVRHyFormer pure baseline.
# ---------------------------------------------------------------------------
python3 -u "${SCRIPT_DIR}/train.py" \
    --num_epochs 10 \
    --patience 3 \
    --seed 42 \
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
    "$@"

echo "[run.sh] training complete (anchor-recalibration-extended); metrics at ${TRAIN_CKPT_PATH}/metrics.json"
