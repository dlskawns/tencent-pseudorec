#!/usr/bin/env bash
# Submission packaging — TAAC 2026 UNI-REC.
# Runs local_validate.py, then zips submission/ into a clean artifact.
# Usage:
#     export EVAL_DATA_PATH=/abs/path/to/test_dir
#     export EVAL_RESULT_PATH=/abs/path/to/out_dir
#     bash submission/prepare.sh

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"

: "${EVAL_DATA_PATH:?must be set to test parquet dir or file}"
: "${EVAL_RESULT_PATH:?must be set to writable output dir}"

PY="${PY_BIN:-${ROOT}/.venv-arm64/bin/python}"
if [[ ! -x "${PY}" ]]; then
    PY="$(command -v python3)"
fi

echo "[prepare] using python: ${PY}"
echo "[prepare] EVAL_DATA_PATH=${EVAL_DATA_PATH}"
echo "[prepare] EVAL_RESULT_PATH=${EVAL_RESULT_PATH}"
echo "[prepare] MODEL_OUTPUT_PATH=${MODEL_OUTPUT_PATH:-<unset>}"

echo "[prepare] running local_validate.py ..."
"${PY}" "${HERE}/local_validate.py"

GIT_SHA="$(git -C "${ROOT}" rev-parse --short HEAD 2>/dev/null || echo "no-git")"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
ART="${ROOT}/submission_${TS}_${GIT_SHA}.zip"

echo "[prepare] packaging -> ${ART}"
( cd "${HERE}" && zip -qr "${ART}" \
    infer.py \
    local_validate.py \
    $( [[ -d ckpt ]]    && echo ckpt    ) \
    $( [[ -f README.md ]] && echo README.md ) \
)

echo "[prepare] DONE: ${ART}"
echo "[prepare] append submission record to ${HERE}/README.md before uploading."
