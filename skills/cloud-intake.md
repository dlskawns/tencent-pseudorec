# Skill: cloud-intake

> 클라우드 학습이 끝나고 사용자가 결과를 들고 돌아왔을 때 호출. 모든 추적 문서를 한 번에 갱신한다.

## When to invoke
- 사용자가 "E??? 결과 왔어" / "학습 끝났어" / "metrics.json 받았어" 라고 할 때.
- 사용자가 `training_result.md` 본문을 paste 했을 때.
- 사용자가 `metrics.json` 또는 train.log 일부를 paste 했을 때.

## Inputs
사용자 paste 또는 다운로드 파일:
- `metrics.json` (1차 출처 — `best_val_AUC`, `best_oof_AUC`, `config_sha256`, `git_sha`, `host`, `split_meta`).
- 학습 로그 마지막 50–200줄 (P1 NaN-check, 실제 wall time, 이상 징후).
- (옵션) 학습된 ckpt 경로 (다음 실험 control로 쓸 때 필요).
- (옵션) 플랫폼 메타 (실제 GPU, 비용, 큐 시간).

## Outputs
1. **`experiments/{exp_id}/training_result.md`** — 채워진 양식.
2. **`experiments/{exp_id}/metrics.json`** — paste된 내용 그대로 저장 (1차 출처 보존).
3. **`experiments/INDEX.md` 행 갱신** — `val_AUC`, `OOF_AUC`, `cost_usd`, `status` 컬럼.
4. **`hypotheses/{H}/verdict.md`** — 해당 실험이 가설 P-condition을 해소하면 status / Findings 갱신.
5. **`progress.txt` iter 블록 append**.
6. **CLAUDE.md §17.4 카테고리 rotation 카운터 업데이트** (직전 2 H의 primary_category 추적).

## Workflow
1. **Parse metrics.json**:
   - `best_val_AUC`, `best_oof_AUC` — 핵심 수치.
   - `config_sha256` 가 card.yaml의 값과 일치하는지 확인. 불일치 시 사용자에게 변경 사항 확인.
   - `split_meta` — train/valid/oof rows + cutoff. card.yaml `expected_split` 와 비교.
2. **Compute paired Δ vs control** (control_exp_id가 card.yaml에 있을 때):
   - control의 `metrics.json`에서 동일 metric을 읽음.
   - paired Δ = treatment − control.
   - seed 수가 1이면 신뢰구간 계산 불가, 단일 점추정 + warning.
   - seed ≥ 3이면 paired bootstrap CI lower bound 계산.
3. **Verdict 결정** (`falsification:` block 적용):
   - P1–P5 별로 PASS / FAIL / inconclusive 판정.
   - 전체 status = `supported` (모든 P pass) / `refuted` (P1 미달 또는 임계치 미달) / `inconclusive` (측정 불가).
4. **Write artifacts**:
   - `training_result.md` 채움.
   - `experiments/INDEX.md` 행 갱신.
   - 가설이 종결되면 `hypotheses/{H}/verdict.md` 작성.
   - `progress.txt` iter 블록 append.
5. **Trigger next**:
   - §17.3 (binary success): treatment가 임계 미달이면 그 component 방향 retire 메시지.
   - §17.4 (rotation): 다음 H 후보 중 직전 2개와 다른 카테고리 1개 추천.
   - 다음 실험의 `card.yaml` 스캐폴드 제안 (사용자 확인 후 실제 작성).

## Output schema (training_result.md 핵심 필드)
- `submitted_at` (UTC ISO).
- `platform` (Tencent Cloud / Colab Pro / Modal / Lambda / RunPod / Kaggle / 기타).
- `gpu` (실제 모델, 예: A100-40GB).
- `wall_time` (hh:mm:ss).
- `cost_usd` (실측 또는 추정).
- `metrics_json_blob` — paste된 verbatim.
- `log_tail` — 마지막 30–200줄.
- `falsification_check` — P1–P5 결과 표.
- `notes` — 플랫폼 특이사항, OOM/재시도, surprise.
- `next_actions` — intake skill이 자동 채움 (다음 H 제안 등).

## Anti-patterns
- **사용자 수치 무비판 적용 금지**: `metrics.json` 의 `git_sha` 가 우리 expected와 다르면 다른 코드를 돌린 것. 차이 사유 확인 후 적용.
- **카테고리 rotation 카운터 무시 금지**: 직전 2 H가 같은 카테고리고 본 H도 같으면 §17.4 위반. challengers.md 재진입 정당화 검증.
- **"promising trends" 차단**: §17.3 binary 임계 미달이면 verdict = refuted. "with more tuning" 문구 진입 금지.
- **메타 누락 verdict 작성 금지**: `seed`, `git_sha`, `config_sha256` 셋 다 빠진 결과는 reproducibility skill 규약 위반 — verdict 미작성, 사용자에게 메타 보강 요청.
