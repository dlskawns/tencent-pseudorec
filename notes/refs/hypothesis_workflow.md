# Hypothesis & Experiment Workflow (former §5 + §9 + §17)

> **트리거**: 새 H (hypothesis) 또는 새 E (experiment) 작성·실행 시 본 파일 먼저 읽기.

---

## §5 Workflow Cycle (7 steps, 각 step ↔ skill)

```
1. redefine-problem   → hypotheses/HXXX/problem.md
2. challenger         → hypotheses/HXXX/challengers.md  (≥2 반대 프레임)
3. literature-scout   → papers/{category}/*.md (있으면 먼저 읽기, 없으면 수집)
4. method-transfer    → hypotheses/HXXX/transfer.md     (P1+ PreToolUse hook 강제)
5. hypothesize        → hypotheses/HXXX/predictions.md  (반증 가능)
6. experiment-card    → hypotheses/HXXX/experiments/EYYY/card.yaml + run
7. verify-claim + journal-update → verdict.md, progress.txt append
```

step 1, 2를 skip하면 local-minima 박힘. EDA 의견은 좋은 "첫 가설"이지 "정답"이 아님.

---

## §9 Hypotheses — 조직 단위

```
hypotheses/
├── INDEX.md                     # 전 가설 레지스트리: (id, claim, status, verdict, UNI-REC axes, primary_category)
├── TEMPLATE/                    # 새 H 생성용 스켈레톤
└── HXXX_slug/
    ├── problem.md               # redefine-problem
    ├── challengers.md           # ≥2 반대 프레임
    ├── transfer.md              # P1+ 필수 — §⑤ UNI-REC alignment 포함
    ├── predictions.md           # 반증 조건, 기대 effect size
    ├── lit_refs.md              # papers/ 링크
    ├── verdict.md               # supported / refuted / inconclusive
    └── experiments/
        └── EYYY_.../
            ├── card.yaml        # 실행 전 (hypothesis_id, seed, metric, 조기종료, compute_tier)
            ├── config.yaml      # 모델/데이터/학습 하이퍼. sha256 대상
            ├── run.sh           # 재현 명령 1줄
            ├── metrics.json     # 결과 + repro 메타
            └── notes.md         # 관찰, surprise, 다음 가설 시드
```

**새 H 생성 전**: `hypotheses/INDEX.md`를 grep. 유사 claim이 있으면 새 H 대신 기존 H의 `experiments/`에 새 실험 추가.

**새 실험 생성 전**: `config.yaml`의 canonical sha256 → `experiments/INDEX.md` 중복 체크.

---

## §17 Anti-LM Discipline (2026-04-26)

> 사용자 회고: 이전 캠페인이 local-minima에 빠짐 (8/10 실험이 같은 카테고리). 본 §17은 그 trap을 **구조적으로** 차단하는 7원칙. §10의 anti-bias rule 위에 운영 규율을 추가.

### §17.1 — Baseline-first, end-to-end on cloud
첫 실험 (E000, H001) 은 ablation이 아니라 **organizer baseline + 결함 패치만 적용한 anchor**. 모든 후속 H는 E000의 OOF AUC 대비 paired Δ로만 평가. anchor 없이 측정한 lift는 거부.

### §17.2 — One-mutation-per-experiment, structural not parametric
한 실험은 **한 component 클래스**를 교체. 예: `MultiSeqHyFormerBlock` → InterFormer 2-arch bridge / `RankMixerNSTokenizer` → OneTrans NS-token mixed-causal. 하이퍼파라미터 (focal γ, lr, dropout, init scale) 튜닝은 P2까지 명시 금지. 다중 mutation 한 실험에 묶기 차단.

### §17.3 — Component-level success is binary
임계: Δ ≥ +0.5 pt OOF AUC (cloud full-data 기준), seed × 3 paired bootstrap, CI lower bound > 0. 미달 → REFUTED, 그 component 방향 retire. "promising trends with more tuning" 문구는 verdict.md에서 차단 사유.

### §17.4 — Forced category rotation
새 H의 `primary_category`가 직전 H와 같으면 `challengers.md`에 명시 재진입 정당화 + 직전 verdict.md F-N 직접 인용 필수. 어느 카테고리도 다른 모든 카테고리 1회 미경험인 상태에서 2회차 금지 (§10.7 강화).

### §17.5 — Sample-scale = code-path verification only
1000-row 결과로 winner 선정 금지. sample run의 4가지 용도만:
- (a) 코드 실행 가능?
- (b) loss가 100 step 안에 하강?
- (c) trainable params § §10.6 budget?
- (d) `submission/local_validate.py` 5/5 통과?
점수 비교는 cloud full-data에서만.

### §17.6 — Cloud cost = bottleneck, not compute
T0/T1: 무한대. T2 (Modal/Colab): per-job ≤ $5, per-day ≤ $20. T3 (Lambda/RunPod on-demand): per-run ≤ $12, per-campaign ≤ $100. 초과 시 사용자 confirm. `experiments/INDEX.md`에 cumulative cost 기록 의무.

### §17.7 — Falsification-first hypothesis
실험 시작 전 `predictions.md` 에 "negative result로부터 무엇을 배우나" 명확히 기술. negative가 uninterpretable 이면 **malformed experiment** — 실행 차단.

### §17.8 — Cloud handoff discipline (사용자가 학습 직접 실행)
**전제**: 학습은 사용자의 클라우드 플랫폼에서 사용자가 직접 실행. 어시스턴트는 학습 자동 호출 금지.

**플랫폼 업로드 모델 (TAAC Training Code 페이지)**:
- 업로드 단위 = **개별 파일 (flat namespace)**, NOT tar.gz. 사용자가 "Upload from Local" 으로 파일 한 개씩 보냄.
- Entry point = `run.sh` (플랫폼이 `bash run.sh` 호출).
- 데이터는 플랫폼이 `TRAIN_DATA_PATH` env로 주입. 우리 코드는 plain parquet 디렉토리만 받으면 됨.
- Storage cap = 100 MB. 우리 bundle은 ≤ 200 KB라 cap 여유 있음.
- Final Round 시 `README.md` (technical report) 의무 — `build_cloud_package.py` 가 자동 생성.

**규칙**:
1. **모든 T2/T3 실험**은 실행 직전 `experiments/{exp_id}/upload/` (flat dir, 9 files) + `experiments/{exp_id}/upload.tar.gz` (backup) + `training_request.md` 생성 (skill: `cloud-handoff`). 패킷 없이 사용자에게 실행 요청 금지.
2. 사용자가 결과 들고 돌아오면 `training_result.md` 채우기 + INDEX/verdict/progress 갱신 (skill: `cloud-intake`).
3. INDEX.md 행은 intake 완료 전까지 `pending`. intake 누락된 결과로 다음 H 시작 금지.
4. `metrics.json` 의 `git_sha`, `config_sha256` 이 expected와 다르면 사용자에게 변경 사유 확인 후 적용.
5. seed 1회 결과는 신뢰구간 계산 불가 → §17.3 binary 임계 미달이면 자동 refuted, **단** 사용자가 seed×3 재학습 명시 요청 시 보류 가능.
6. **upload/ 디렉토리는 평문 파일만**: subdirectories 금지, venv/git/pycache/data/ckpt/logs 자동 exclusion.
7. **plat run.sh는 local-dev defaults 금지**: `${ROOT}/data` 같은 자동 fallback 제거. `TRAIN_DATA_PATH:?` (필수) / `TRAIN_CKPT_PATH:?` (필수). LOG/TF/WORK은 CKPT에서 derive.

### §17 PreToolUse hook 후보 (P1+ 활성화)
- `experiments/{id}/card.yaml` 의 `compute_tier:` 누락 시 차단.
- `card.yaml` 의 `claim_scope:` 누락 시 차단.
- 같은 `primary_category` 가 직전 2개 H에서 등장 + `challengers.md`에 "재진입 정당화" 섹션 부재 시 차단.
- T2/T3 실험에서 `training_request.md` 누락 시 차단 (cloud-handoff skill 미통과).
- `hypotheses/HXXX/{problem,transfer,predictions,lit_refs}.md` 안에 §3 의 "후보" / "TBD" / "미검증" 라벨이 붙은 사실 직접 인용 검출 시 차단. `competition/ns_groups.json` 또는 `eda/out/*.json` 검증값으로 §3 본문 갱신 후 진입 권유. (CLAUDE.md §4.9 enforcement.)
