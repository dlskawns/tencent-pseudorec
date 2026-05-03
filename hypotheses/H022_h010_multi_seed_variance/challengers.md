# H022 — Challengers (≥ 2 reverse frames)

> §10.3 — H022 = first measurement H (no mutation), category
> `measurement` first-touch. rotation auto-justified. mutation-H rules
> 일부 exempt (§17.2 단일 component 교체 = mutation H 만 적용).

---

## §17.4 rotation 정당화 (auto-justified)

- 카테고리 = `measurement`, first-touch.
- 직전 H (H015~H018) = temporal_cohort, H019 = retrieval_long_seq,
  H020a/b/H021 = backbone_replacement / debiasing (미시도). H022 =
  완전히 다른 axis (no mutation).
- §10.7 unexplored 카테고리 list 에 측정 infrastructure 자체가 빠져
  있었음 — H022 가 그 gap 채움.

---

## Frame A — "3 seeds 부족, variance estimate noisy"

**가설**: 3 seeds 의 stdev 자체가 high variance estimator. 5+ seeds 또는
bootstrap resampling 없이 σ 추정 자체가 ±factor 2 의 uncertainty.

**근거**:
- statistics 표준: ≥ 5 sample 후 stdev 신뢰 가능. 3 sample 의 stdev 는
  itself wide CI.
- Welford / sample variance unbiased estimator: n=3 → df=2, t-distribution
  CI 매우 넓음.
- machine learning replicate studies (예: Bouthillier et al. 2021,
  "Accounting for Variance in Machine Learning Benchmarks") 가 5-10 seed
  권장.

**Falsification 조건**: 3 seeds 의 σ 가 sub-microsecond level (~1e-5pt) →
Frame A REFUTED, 3 seeds 충분.

**Frame A confirmed (σ wide / inconsistent) 시 carry-forward**: H022-sub
= 5 seeds 추가 (총 8 seeds, $40 cost). 또는 paired bootstrap CI 도입
(seed 마다 prediction-level resample).

---

## Frame B — "Multi-seed variance != ranking variance"

**가설**: 3-seed σ 가 absolute platform AUC variance 만 측정. **paired Δ
variance** (H015 vs H010, H012 vs H010 등) 는 같은 seed 비교 시 더 작음
(systematic noise cancels). 따라서 H022 의 σ ≥ paired Δ σ → conservative
threshold 만 줌.

**근거**:
- paired bootstrap (Efron 1979) 은 same-data-different-method 비교 시
  same-seed correlation 활용 → 더 tight CI.
- H015 vs H010 paired (same seed 42) 시 σ_paired < σ_individual.

**Falsification 조건**: 3 seeds 의 σ 가 매우 작아 (≤ 0.001pt) 모든 marginal
Δ 가 명확히 outside ±2σ → Frame B 무관 (어쨌든 trustworthy).

**Frame B confirmed (σ moderate, paired Δ unclear) 시 carry-forward**:
H022-sub = paired bootstrap CI implementation (more complex, prediction-
level resample). 또는 다른 H 의 multi-seed paired Δ direct 측정.

---

## Frame C — "Variance baseline 의 cost ROI 가 paradigm shift 보다 낮다"

**가설**: $15 (3 seeds × $5) 로 variance estimate 만 얻는 게, 그 돈으로
H019 (TWIN, $15) 또는 H020b (debiasing, $5) 추가 mechanism test 보다
expected information gain 작음. variance 가 어떻든 paradigm shift 시도
는 어차피 해야.

**근거**:
- 9 H 누적 marginal/REFUTED 가 variance 와 무관하게 ceiling 신호.
  variance 측정해도 mechanism 결정 안 바뀜 가능성.
- $15 으로 H019 + H020b 동시 launch 가능 (TWIN $15 만 — H020b $5 추가 시 $20).
  더 많은 mechanism test.

**Falsification 조건**: H022 σ 가 large (> 0.005pt) → 기존 9 H 의 단일
seed 측정 모두 INVALID → variance baseline 이 catastrophic 발견 → Frame C
REFUTED (variance 측정 없으면 기존 모든 결론 무효).

**Frame C confirmed (σ tight) 시 carry-forward**: H022 가 "validation
only" 였음 — 신호 없음 confirm. 단 historical Δ 들 의 신뢰 levels 정량화
(future H 평가 framework 보강) 자체는 가치.

---

## Counter-argument 종합 (왜 그래도 H022 진행)

1. **Cheapest critical infrastructure**: $15 = single H019 cost ≤. 모든
   future Δ 해석 framework 결정 → cost ÷ value 매우 낮음.
2. **Retroactive impact**: 9 H 의 marginal Δ 들 (H015 +0.0002, H012 −0.003,
   etc.) 의 신뢰도 retroactive 갱신. 기존 데이터 재해석 = "free" lift in
   decision quality.
3. **Subset A 의 logical foundation**: H018/H019 의 결과 해석에 σ 필수.
   variance 모르면 H018 noise vs measurable 구분 못 함, H019 strong vs
   measurable 구분 못 함.
4. **paradigm shift cost cap audit**: H020a HSTU $25 / H020 series cost
   cap 위협. variance 가 large → multi-seed 강제 → cost 3× → cap 이미
   초과. variance 가 tight → single-seed 유지 → cap 안전. 이 결정 자체가
   $15 짜리.
