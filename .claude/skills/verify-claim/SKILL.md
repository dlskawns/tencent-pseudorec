---
name: verify-claim
description: Use when cloud training results return — user pastes metrics.json or training_result.md content, or says "E### 결과 왔어", "학습 끝났어", "metrics.json 받았어", "verdict 써줘", "H### 판정", "다음 H 뭐?". Parses metrics.json, computes paired Δ vs control_exp_id from card.yaml, applies §17.3 binary verdict (PASS/REFUTED/INVALID), then atomically updates 6 artifacts: experiments/E###/{verdict.md, metrics.json, training_result.md}, hypotheses/HXXX/verdict.md, hypotheses/INDEX.md (Active → Archive + carry-forward F-N), experiments/INDEX.md row, progress.txt new iter block, and §10.7/§17.4 rotation counter. Suggests next H. Do NOT use for T0/T1 sample-scale runs.
---

# verify-claim

> Cloud 결과 paste → 6 artifact 원자적 갱신 + 다음 H 추천.
> CLAUDE.md §17.3 (binary verdict), §17.4 + §10.7 (rotation), §18 (infra check), §4 (reproducibility metadata).

## When to invoke

**Auto** (description 매칭):
- "E### 결과 왔어" / "학습 끝났어" / "metrics.json 받았어"
- 사용자가 `metrics.json` 본문 또는 `training_result.md` 를 paste
- "verdict 써줘" / "H### 판정해줘" / "다음 H 뭐?"

**Explicit**:
- `/verify-claim H###`

**NOT-when**:
- T0/T1 sample-scale (formal verdict 불필요)
- `predictions.json` local_validate 결과 (= submission-validate skill 영역)
- card.yaml 작성 (= new-experiment-card skill 영역)

## Inputs

**필수 (§18.8 format, H018+ default)**:
- **§18.8 TRAIN SUMMARY block** — `==== TRAIN SUMMARY (HXXX_slug) ====`
  부터 `==== END SUMMARY ====` 까지 단일 블록. 사용자가 stdout 마지막
  ~15줄 그대로 paste. 모든 핵심 metric (best/last/overfit/calib +
  per-epoch trajectory) 한 번에.
- **Platform AUC** — 사용자 별도 paste (Taiji 채점 결과 단일 숫자).

**필수 (legacy fallback, pre-H018)**:
- `metrics.json` — best_val_AUC, best_oof_AUC, best_step, config_sha256,
  git_sha, host, split_meta
- 학습 로그 마지막 50–200줄 (NaN check, 실제 wall, 이상 징후)
- `eval auc: 0.XXXXXX` 단일 라인 (legacy 포맷)

**선택**:
- `training_result.md` 본문 (사용자 채운 양식)
- 학습 ckpt 경로 (다음 H control 용)
- 플랫폼 메타 (실제 GPU, 비용, 큐 시간)

## Workflow

### 0. Parsing §18.8 TRAIN SUMMARY block (H018+ primary path)

**Anchors (변경 금지 — 정규식 파싱)**:
- 시작: `^==== TRAIN SUMMARY \((?P<exp_id>[^)]+)\) ====$`
- 종료: `^==== END SUMMARY ====$`

**Required fields (parse 후 metrics.json 으로 정규화)**:

| 필드 (SUMMARY) | metrics.json 키 | 정규식 hint |
|---|---|---|
| `git=<sha7>` | `git_sha` | `git=([a-f0-9]{7,})` |
| `cfg=<sha8>` | `config_sha256` (앞 8자 only) | `cfg=([a-f0-9]{6,})` |
| `seed=<int>` | `seed` | `seed=(\d+)` |
| `ckpt_exported=<best\|last>` | `ckpt_exported` | `ckpt_exported=(best\|last)` |
| epoch table rows | `epoch_history[]` | `^\s*(\d+)\s*\|\s*([\d.NA/]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)$` |
| `best=epoch<N>` ... | `best_epoch`, `best_val_auc`, `best_oof_auc` | `best=epoch(\d+)\s+val=([\d.]+)\s+oof=([\d.]+)` |
| `last=epoch<N>` ... | `last_epoch`, `last_val_auc`, `last_oof_auc` | `last=epoch(\d+)\s+val=([\d.]+)\s+oof=([\d.]+)` |
| `overfit=<+/-float>` | `overfit_gap` | `overfit=([+-]?[\d.]+)` |
| `calib pred=` ... | `calib_pred_mean`, `calib_label_mean`, `ece` | `calib pred=([\d.]+) label=([\d.]+) ece=([\d.]+)` |

**Computed fields (parser 가 platform AUC 받은 후 산출)**:

- `val_platform_gap` = `best_val_auc − platform_auc`
- `oof_platform_gap` = `best_oof_auc − platform_auc` (redefined OOF, H016 default)
- `last_minus_best_val` = `last_val_auc − best_val_auc` (음수면 best 가 마지막보다 좋음 = 정상 overfit)

**OOF default (H016 carry-forward)**: `best_oof_auc` 컬럼은
**redefined OOF (label_time future-only quantile 0.9 cutoff)** 만 기록.
legacy random-user OOF 는 saturated 0.858~0.860 → ranking decision 무용 →
metrics.json 에 별도 `legacy_oof_auc` 필드로 분리 (보조용).

### 0.5 Fallback — legacy format (pre-H018)

SUMMARY 블록 미감지 시:
1. **WARN to user**: "§18.8 SUMMARY block not found. Falling back to
   legacy `eval auc:` parsing. Upgrade `train.py` to emit_train_summary()
   per `notes/refs/inference_lessons.md` §18.8 for richer metrics."
2. Parse `eval auc: 0.XXXXXX` 단일 라인 → `best_val_auc` 로 매핑 (정확
   하지 않음, last epoch 일 가능성).
3. `overfit_gap`, `calib_*` 등 신규 필드 = `null`. P5/P6 row = WARN 표시.
4. verdict 진행 가능하지만 **"신호 빈약"** 표기.

### 1. Sanity gate (실패 시 차단)

- `metrics.json.config_sha256` ↔ `experiments/E###/card.yaml.config_sha256` 일치 확인.
  - 불일치 → 사용자에게 "다른 코드 돌렸나요?" 확인. 응답 없이 진행 금지.
- `metrics.json.git_sha` ↔ card.yaml expected git_sha 일치 확인.
  - 불일치 시 verdict.md Status 에 `(git_sha mismatch)` 표시.
- `split_meta` 필드 ↔ card.yaml `expected_split` 비교 (rows, cutoff, oof_user_count).
- §4.5 메타 (seed/git_sha/config_sha256) 셋 다 누락 → INVALID, verdict 미작성, 사용자에게 메타 보강 요청.

### 2. Paired Δ 계산

- `card.yaml.control_exp_id` 읽음.
- control 의 `metrics.json` 에서 동일 metric 추출.
- paired Δ = treatment − control on **platform AUC** (H006 F-3 carry-forward: OOF 는 supplementary).
- seed=1 → 점추정 + warning. seed≥3 → paired bootstrap CI lower bound.

### 3. Verdict 분류 (§17.3 binary)

predictions.md 의 P1–P5 falsification cut 그대로 적용:
- **PASS / supported**: 모든 P pass + primary lift Δ ≥ §17.3 cut + (seed≥3 시) CI > 0
- **REFUTED**: P1 fail (NaN/OOM) OR Δ < cut OR strongest-single paired interference
- **INVALID**: 메타 누락 OR sha mismatch unresolved

**Anti-pattern 차단**: "promising trends" / "with more tuning" 진입 금지. 임계 미달 = REFUTED.

### 4. Findings (F-N) 추출

H009 verdict 패턴 — 각 F 는:
- 한 줄 headline + carry-forward 명시
- 정량 수치 (AUC / gap / wall)
- 향후 H 에 어떻게 영향 주는지 ("H011+ 부터 ...")

### 5. 6-artifact 원자적 갱신

원자적으로 (한 turn 안에) 작성:

1. **`experiments/E###/verdict.md`** — 아래 template 형식
2. **`experiments/E###/metrics.json`** — paste 내용 verbatim 저장 (1차 출처 보존)
3. **`experiments/E###/training_result.md`** — submitted_at, platform, gpu, wall_time, cost_usd, metrics_json_blob, log_tail, falsification_check 표, notes, next_actions 자동 채움
4. **`hypotheses/HXXX/verdict.md`** — E 결과 → H 판정 매핑, status (supported/refuted/invalid/inconclusive)
5. **`hypotheses/INDEX.md`**:
   - Active Pipeline 행을 Archive 표로 이동 (Platform AUC, status, verdict 핵심 1줄)
   - "Recent Findings (carry-forward)" 새 항목 prepend (F-N + 핵심 수치)
   - "Active Phase" 섹션 갱신 (직전 verdict 반영, 다음 H 후보 1줄)
6. **`experiments/INDEX.md`** — 해당 행의 val_AUC / OOF_AUC / platform_AUC / wall (학습) / wall (infer) / status 컬럼 채움
7. **`progress.txt`** — 새 iter 블록 append (아래 형식)
8. **카테고리 rotation 카운터** (§10.7 + §17.4):
   - 직전 2 H 의 primary_category 추적 (transfer.md 에서 추출)
   - 다음 H 후보 추천 시 직전 2개와 다른 카테고리 우선
   - 같은 카테고리 3회 연속 시 challenger 재진입 정당화 요구 (§10.3)

### 6. 다음 H 추천

- §17.3 (binary): treatment 가 임계 미달이면 그 component 방향 retire 메시지
- §17.4 (rotation): 다음 H 후보 1개 (직전 2 H 와 다른 카테고리)
- 사용자 OK 시 → `new-hypothesis` skill 위임 (별도 turn)

## verdict.md template (H009 형식)

```
# H### — Verdict (STATUS — short reason)

> 1-paragraph 요약 (날짜, 핵심 수치, 결정).

## Status
`done` — STATUS. Platform AUC X.XXXX vs anchor/control X.XXXX, ΔX.XXpt.

## Source data
- 학습 envelope, ckpt, inference wall.

## P1 — Code-path success
- Measured: ...
- Verdict: PASS / FAIL.

## P2 — Primary lift (§17.3 binary)
- Measured: Platform AUC = X.XXXX, OOF AUC = X.XXXX.
- Δ vs anchor / control / strongest-single.
- Verdict: PASS / REFUTED.

## P3 — Mechanism 작동 검증
- Measured: ...
- Verdict: ...

## P4 — §18 인프라 통과
- Measured: inference wall, batch heartbeat, "[infer] OK" log.
- Verdict: PASS / FAIL.

## P5 — (보너스 quantitative)
- Measured: ...

## Findings (F-N carry-forward)
- **F-1 (...)**: 한 줄 + 정량 + carry-forward.
- **F-2 (...)**: ...

## Surprises
- 예상 못한 관찰.

## Update to CLAUDE.md?
- 본문 갱신 보류 / §X.Y 추가 제안 / 카운터만.

## Carry-forward to H### (다음 H)
- F-N → ...

## Decision applied (per predictions.md decision tree)
- 어느 분기가 적용됐는지.
```

## progress.txt iter block 형식

```
================================================================================
YYYY-MM-DD — iter — H### verdict (STATUS) [+ 다음 H scaffold/제안]
================================================================================

CONTEXT
- ...

EXECUTION
- 학습 결과 핵심 수치, wall, infer wall.

VERDICT (H###)
- status, F-1~F-N 1줄 요약.

FINDINGS
- F-N 자세히.

ARTIFACTS
- 갱신된 파일 리스트.

NEXT (iter+1)
- 다음 H 추천 (rotation respect).
```

## Anti-patterns (skill 차단 사유)

1. **수치 무비판 적용**: config_sha256 / git_sha mismatch 무시한 채 verdict 작성. 항상 sanity gate 먼저.
2. **카테고리 rotation 무시**: 직전 2 H 와 같은 카테고리 H 추천 (challengers 재진입 정당화 없이).
3. **"promising trends" 문구**: §17.3 binary 임계 미달이면 REFUTED. 회색지대 진입 금지.
4. **메타 누락 verdict**: seed / git_sha / config_sha256 셋 다 빠진 결과 → INVALID 분류, verdict 미작성.
5. **6-artifact 부분 갱신**: 시간 부족하다고 progress.txt 만 빼거나 INDEX 만 빼면 안 됨. 원자성.
6. **paired Δ 단위 혼동**: platform AUC primary (H006 F-3). OOF AUC 는 supplementary.
7. **Active Pipeline 행 미이동**: H 가 verdict 받았는데 INDEX 의 Active Pipeline 에 남아있으면 다음 세션 혼란.
8. **§18.8 SUMMARY 마커 변형 수용**: `==== TRAIN SUMMARY (` / `==== END SUMMARY ====` 외 형식 (예: `### TRAIN SUMMARY`, `--- summary ---`) 은 INVALID. 사용자에게 train.py 수정 요청, verdict 미작성.
9. **legacy OOF (saturated 0.858~0.860) 를 ranking signal 로 사용**: H016 redefine 이후 legacy OOF 는 noise. `best_oof_auc` = redefined OOF only. legacy 는 `legacy_oof_auc` 보조 필드.
