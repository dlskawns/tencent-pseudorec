# Submission Log — TAAC 2026 UNI-REC

> CLAUDE.md §14.2 — 매 제출마다 1블록 추가. (블록 = `### YYYY-MM-DDThh:mmZ — exp_id` 헤더 + 아래 4 필드.)

## Layout

```
submission/
├── infer.py            # §13 contract entry — env-only paths, def main(), no internet
├── local_validate.py   # G1–G6 gates (§14.1)
├── prepare.sh          # validates then zips the bundle
├── ckpt/               # (optional) state dict + metadata; loaded by infer.py via MODEL_OUTPUT_PATH
│   ├── model.pt
│   ├── prior.json      # {"prior": <float in [0,1]>} — T0 fallback class prior
│   └── (paper-tier specific configs)
└── README.md           # this file
```

## Smoke test (local)

```bash
export EVAL_DATA_PATH=/Users/david/Desktop/assignments/DataEngine/tencent-cc2/data/demo_1000.parquet
export EVAL_RESULT_PATH=$(mktemp -d)
.venv-arm64/bin/python submission/local_validate.py
```

기대: `Result: 5/5 gates passed`. 실패 시 §13.7 매핑 + `progress.txt`에 사유 기록.

## Packaging

```bash
bash submission/prepare.sh
# -> submission_YYYYMMDDThhmmssZ_<sha>.zip in repo root
```

zip을 플랫폼에 업로드. 업로드 전 본 README에 새 블록 1개 추가.

---

## Submissions

(no submissions yet)

### Template

```
### 2026-MM-DDThh:mmZ — H001/E001 (T0.1 class-prior)
- git_sha: abc1234
- compute_tier: T0
- local_validate: 5/5 PASS
- platform_status: pending|success|failed
- score: TBD
- notes: first round-trip verification of §13 contract
```
