# H021 — Challengers (≥ 2 reverse frames)

> §10.3 + §17.4 — H021 = TWIN sub-H, quantity axis. H019/H020 same category 3회 연속 — RE_ENTRY 정당화 carry-forward + per-domain K mutation reverse frame 명시.

---

## §17.4 rotation re-entry 정당화 (H020 carry-forward + H021 추가 사유)

**카테고리**: `retrieval_long_seq` (3회 연속 — H019 + H020 + H021).

**§10.7 audit — 같은 카테고리 3회 연속 룰 위반 검토**:
- H019 (paradigm shift first entry, retrieval class) → H020 (scoring axis sub-H) → H021 (quantity axis sub-H).
- 3회 연속 trigger.

**re-entry 정당화 (H020 의 4 사유 + H021 추가 1 사유)**:
1. (carry from H020) sweep saturation: top_k/seq/gate/batch axis 모두 saturated.
2. (carry from H020) paper-faithful: H019 simplified GSU 의 paper-faithful learnable form (H020) + per-domain K (paper TWIN 의 single-history 와 다른 본 데이터 specific) 직접 검증.
3. (carry from H020) cost-effective: paradigm shift class 안 가장 cheap.
4. (carry from H020) falsification value: 4 outcome 모두 다음 H 결정 명확.
5. **(H021 추가) §3.5 quantitative motivation 직접**: domain d p90=2215 가 K=64 의 top 2.9% — 4 도메인 중 가장 under-served. uniform K=64 sweep 으로 검증 불가능 (per-domain isolation 필요).

**rotation_status**: `RE_ENTRY_JUSTIFIED` (carry from H020 + H021 추가 사유).

---

## Frame A — "K=64 가 globally saturated, per-domain 변경 NOOP"

**가설**: uniform K sweep (32/64/128) 결과 = K=64 가 모든 도메인 평균 sweet spot. domain d 만 K=96 으로 늘려도 추가 token 들이 noise (relevance 낮음) → ESU attention 이 무시 → output 변화 없음. uniform K=128 의 flat 결과가 직접 신호.

**근거**:
- top_k=128 uniform sweep 시 OOF 0.8612 ≈ H019 0.8611 — K↑ 의 추가 token 들이 lift 안 만듦.
- domain d 추가 token 들 (rank 65~96) 도 같은 saturation 일 가능성.
- ESU MultiheadAttention 은 K-token 모두에 attend 하지만 attention weight 가 낮은 token 은 자동 무시 → effective K 가 score-driven.

**Falsification 조건**: H021 Δ vs H019 ∈ (−0.001, +0.001pt] (noise band) → Frame A confirmed.

**Frame A confirmed 시 carry-forward**: per-domain K 가 lever 아님. uniform K 가 이미 effective limit. retrieval quantity axis 의 모든 lever (uniform K + per-domain K) saturation. ESU capacity (A3) 또는 cohort drift (C) pivot.

---

## Frame B — "Domain d 의 long history 가 K=64 로도 top-relevance 충분 cover"

**가설**: domain d 의 long history (p90=2215) 안에 candidate-relevant token 은 실제로는 적음 (예: top 10~30 정도). K=64 가 이미 top 2.9% 면 충분히 cover, K=96 의 추가 token 들 은 noise.

**근거**:
- §3.5 도메인 b/d Jaccard overlap 0.022 — 도메인 d item-id 가 다른 도메인과 거의 disjoint.
- target item 이 user 의 어느 도메인 seq 에도 거의 안 등장 (any_domain 0.4%) — TWIN 의 candidate-history attention 가정이 본 데이터 에 약함.
- domain d frac_empty = 0.080 (가장 높음) — 8% user 는 history 자체 비어 있음. K 늘려도 의미 없음.

**Falsification 조건**: H021 Δ vs H019 < −0.001pt (degraded) + ESU attention entropy domain d 만 증가 (uniform 방향) → Frame B confirmed.

**Frame B confirmed 시 carry-forward**: domain d 의 K 확장이 noise injection. sub-H = K=80 (더 보수) 또는 domain d 만 K=64 유지 + 다른 도메인 (a) 만 확장 (a:96, b:64, c:64, d:64).

---

## Frame C — "Per-domain K 이 inter-domain consistency 깨서 fusion 단계 instability"

**가설**: 4 도메인의 TWIN output 이 `TwinRetrievalAggregator` 에서 mean pool. K=64 vs K=96 → ESU output 의 norm/scale 미세하게 다를 수 있음. mean pool 시 domain d output 이 다른 3 도메인과 다른 분포 → aggregator 학습 불안정.

**근거**:
- ESU MultiheadAttention 의 output norm 은 K 와 무관 (attention weight 가 normalized) — 이론적 안정.
- 그러나 LayerNorm 후의 scale 분포는 K 다양성에 영향 받을 수 있음 (top-K 의 score 분포 differ).
- TwinRetrievalAggregator 의 `proj` Linear 는 4 도메인 평균 input 받음 → 도메인 d 의 다른 분포가 학습 어렵게 만듦.

**Falsification 조건**: H021 Δ vs H019 < −0.001pt (degraded) + train_loss epoch trajectory 가 H019 보다 noisy → Frame C confirmed.

**Frame C confirmed 시 carry-forward**: per-domain K 변경 + per-domain LayerNorm rescale (각 TWINBlock 의 output 을 표준화). 또는 TwinRetrievalAggregator 의 mean → weighted average (학습 weight) 변경. H022 후보.

---

## Counter-argument 종합 (왜 그래도 H021 진행)

1. **H020 과 직교 axis**: scoring (H020) vs quantity (H021) 동시 검증. 둘 다 PASS 면 stack 가능, 한 쪽만 PASS 면 그쪽 anchor.
2. **§3.5 정량 motivation 강함**: domain d under-collecting 가설은 실측 기반 — uniform sweep 으로 검증 불가능.
3. **Single-domain change**: 한 도메인 (d) 만 K 변경 → paired Δ 해석 가장 깔끔. multi-domain 변경 시 어느 도메인 효과인지 분리 어려움.
4. **Pre-build 가치**: 사용자 명시 — 학습 시간 길어서 H020 결과 wait 비용 높음. H021 도 동시 cloud submit 가능.
5. **Cost-effective**: T2.4 ~3.5h × $5-7. H019/H020 동급.
