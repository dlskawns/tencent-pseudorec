# H013 — Challenger Frames

> CLAUDE.md §10.1, §10.7, §17.2. **Parametric mutation** (hyperparameter
> calibration) — §17.2 의 "structural not parametric" 룰 위반 가능성 →
> 정당화 필수 섹션 추가.

## Frame A (default — what we're proposing)

- **Claim**: H006~H012 의 ceiling 0.82~0.8408 은 hyperparameter artifact.
  사용자 batch=2048 (default 256 의 8×) + lr=1e-4 default → effective
  lr 1/8 underpowered. Linear scaling rule (lr 8e-4) 적용 + IO 완화
  (num_workers 2→4, buffer_batches 4→8) 시 H010 mechanism 위 lift ≥ +0.001pt
  (measurable) 또는 ≥ +0.005pt (strong).
- **Mechanism**: H010 byte-identical, run.sh 4 hyperparameter 만 calibrate.
- **Why this could be wrong**:
  1. lr 8e-4 가 너무 커서 발산 (NaN abort 또는 P3 등 instability).
  2. Linear scaling rule 이 항상 작동 안 함 (큰 batch 에선 generalization
     gap 더 벌어질 수 있음). batch 2048 자체가 generalization 면에서 sub-
     optimal 일 가능성 — 그럴 경우 batch 256 복귀 (sub-H) 가 더 좋음.
  3. Mechanism ceiling 이 진짜고 hyperparameter 영향 작을 가능성 (Frame B).
  4. Sparse_lr (Adagrad 0.05) 가 별도 issue 이지만 본 H 는 안 건드림.

## Frame B (counter-frame — mechanism ceiling 진짜)

- **Claim**: Ceiling 은 mechanism limit. H010 NS xattn entropy 0.81 (~2 of
  384 tokens attended) 이 dominant signal sparse capture 의 한계. lr 변경해도
  sparse attention 자체가 정보 bottleneck.
- **What evidence would prove this?**:
  - H013 결과 Δ vs H010 ∈ (−0.001, +0.001pt) noise → lr 변경 효과 없음.
  - 또는 attn_entropy 변화 미세 (H012 패턴) — proper lr 시도해도 selective
    routing 유지.
- **Distinguishing experiment**: H013 그 자체. Δ ≥ +0.005pt → A (artifact),
  Δ ≤ 0 → B (mechanism limit), 사이 → mixed.

## Frame C (orthogonal frame — split definition issue)

- **Claim**: Ceiling 의 정체는 OOF-Platform gap 의 cohort drift. 7개 H
  모두 OOF AUC ~0.86 일관 / Platform AUC 만 변동 → train/eval 의 distribution
  자체가 다름. label_minus_event_seconds gap, recency drift, user cohort
  변화. 어떤 lr / mechanism 에서도 Platform AUC 의 hard ceiling 존재.
- **Mechanism**: temporal cohort 모델링 (recency-aware loss 또는 distribution
  alignment), 또는 OOF 자체를 platform 분포에 맞게 재정의.
- **Cost vs A**: cohort H 는 더 fundamental, design cost 큼. A 가 cheap
  diagnostic (run.sh 만 수정) 라 먼저 분리.

## Decision

**A 선택 — first-pass diagnostic**:
1. **Measurement integrity 가 prerequisite**: 7개 H 의 paired Δ 가 같은
   underpowered regime 에서 측정 → 절대 lift 의미를 해석하기 위해서는
   regime 정합성 검증 필요.
2. **Cost 매우 작음**: run.sh 4 flags 변경, ~3h 학습, mechanism 추가 H
   만들기보다 압도적으로 cheap.
3. **결과가 모든 분기 결정**: A confirm → mechanism ranking 재해석. B
   confirm → Track B (long-seq) 우선. C confirm → cohort H 우선. 어느
   결과든 다음 step 명확.
4. **Linear scaling rule 은 standard practice** — arbitrary tuning 아님,
   §17.2 의 "promising trends with more tuning" anti-pattern 회피.

**B/C 미루기 조건 (즉, B/C 로 돌아갈 trigger)**:
- A 결과 Δ ≤ +0.001pt (noise) → B confirm. H014 = long-seq retrieval (P2
  entry).
- A 결과 OOF-Platform gap 여전히 > 2pt → C confirm. cohort H 우선.
- A 결과 NaN abort → lr 4e-4 sub-H 또는 batch 256 복귀.

## (조건부) §17.2 정당화 — Parametric mutation justified

§17.2 원문: "한 실험은 한 component 클래스를 교체. ... 하이퍼파라미터 (focal γ,
lr, dropout, init scale) 튜닝은 P2까지 명시 금지."

**위반 우려 적용 가능성**:
- 본 H 가 hyperparameter (lr) 튜닝 → §17.2 표면적 위반.

**정당화 근거**:
1. **§17.2 의 의도** = "promising trends with more tuning" anti-pattern
   회피 (사후 fishing). 본 H 는 **사전 가설 + measurement integrity check**
   — H012 F-2 가 이미 명시 trigger 제공.
2. **결과가 prior H 결과 모두에 영향** → mechanism H 가 아닌 **measurement
   H**. CLAUDE.md 가 measurement H 별도 분류 안 했으나 §17.2 의 의도와
   상충 안 함.
3. **Linear scaling rule (Goyal et al. 2017)** = standard practice, 단일
   변수 튜닝 아닌 batch-lr 의 mathematical relationship. arbitrary 튜닝
   아님.
4. **단일 concern**: 4 changes (batch, lr, num_workers, buffer_batches)
   모두 single concern (training efficiency under batch 2048). focal γ /
   dropout / init scale 같은 architecture-level tuning 아님.
5. **결과 분기에서 mechanism limit 가 진짜로 입증되면 prior H mechanism
   ranking 가 valid 확정** → mechanism 추가 H 정당.

**§17.4 rotation 룰**: H013 primary_category 가 없음 (measurement H). rotation
룰 적용 안 됨. 다음 mechanism H (H014) 시 직전 2 H 카테고리 = H011 feature_
engineering + H012 multi_domain_fusion (H013 은 카운트 안 함). 둘 다 차단.
