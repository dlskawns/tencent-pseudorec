# H020 — Challengers (≥ 2 reverse frames)

> §10.3 + §17.4 — H020 = TWIN sub-H, mechanism class re-entry (H019 family).
> rotation re-entry 정당화 + structural change 진입 timing 정당화 명시.

---

## §17.4 rotation re-entry 정당화

**카테고리**: `retrieval_long_seq` (re-entry from H019).

**§10.7 audit — 같은 카테고리 2회 연속 금지 룰 위반 검토**:
- H019 (retrieval_long_seq, paradigm shift first entry) → H020 (retrieval_long_seq, sub-H scoring 변경).
- 2회 연속 같은 카테고리 trigger.

**re-entry 정당화 (§10.7 명시 예외 조건 충족)**:
1. H019 result (cloud) 직전 단계 = 사용자 local sweep 으로 mechanism 진위 partial confirm. top_k / seq / gate / batch axis 모두 saturated (4 axis sweep 결과 H019 미만 또는 동급).
2. H019 GSU = paper 의 simplified form (parameter-free inner product) — paper-faithful learnable scorer 직접 검증이 mechanism class 진위 측정 의미.
3. 다음 paradigm shift (HSTU / OneTrans / cohort attack) cost cap 위협 → cheap-and-meaningful sub-H 가 ROI 높음.
4. 미경험 카테고리 (`debiasing`, `backbone_replacement`) 와 비교 시 H020 의 falsification value 더 크고 cost 더 작음.

**rotation_status**: `RE_ENTRY_JUSTIFIED` (not auto). card.yaml 에 명시.

---

## Frame A — "Scoring 학습이 inner product 와 다르지 않음 (NOOP mutation)"

**가설**: Backbone 이 이미 학습한 embedding space 자체가 candidate-history relevance 에 잘 맞아서, 별도 projection 추가해도 학습 후 W_q/W_k 가 거의 identity 로 수렴. 즉 추가 64K params 가 NOOP. inner product 가 이미 optimal.

**근거**:
- H010 (NS→S xattn) attn entropy 0.81 = highly selective routing. NS-tokens 이 384 S tokens 중 e^0.81 ≈ 2 tokens 만 attend → backbone embedding 이 이미 sparse selection 에 잘 작동.
- H011 (input-stage aligned encoding) REFUTED = backbone embedding 변경이 platform 악화 (overfit signature). embedding space 가 이미 cohort 에 fit 된 상태에서 추가 학습 = 부정 효과 가능성.
- TWIN paper 의 learnable GSU 는 100M+ user 환경에서만 의미 있을 수 있음 — sample-scale 이 아닌 full-data 환경에서도 demo size 에서는 NOOP 가능.

**Falsification 조건**: H020 Δ vs H019 ∈ (−0.001, +0.001pt] (noise band) → Frame A confirmed.

**Frame A confirmed 시 carry-forward**: retrieval scoring axis retire. H022′ = ESU capacity 개선 (A3) 또는 cohort drift 직접 attack (C). retrieval mechanism class 의 selection policy 는 더 이상 lever 아님.

---

## Frame B — "Projection 의 rank reduction 이 정보 손실 (degraded mutation)"

**가설**: W_q, W_k 가 d_model → d_model//4 = 64 → 16 으로 dim 축소. inner product 의 expressiveness 가 16-dim 으로 bottleneck → original 64-dim inner product 대비 정보 손실. 학습이 잘 돼도 capacity 자체 부족.

**근거**:
- TWIN paper 의 GSU projection dim 은 보통 d_model 같거나 절반 (paper-specific). d_model//4 는 §10.6 sample budget 친화 결정.
- H019 inner product 는 d_model=64 full dim 사용 → score signal 이 64-dim space 에서 정의됨.
- d_model//4 = 16 으로 축소 시 candidate 와 history 가 같은 16-dim subspace 에 projection 되면서 다른 candidate 끼리 구분 약화 가능.

**Falsification 조건**: H020 Δ vs H019 < −0.001pt (degraded) + ESU attention entropy 가 H019 대비 증가 (uniform 방향) → Frame B confirmed.

**Frame B confirmed 시 carry-forward**: sub-H = projection dim d_model//2 (= 32) 재시도. 또는 W_k only learnable (W_q = identity, candidate 쪽은 그대로) — params 절반, candidate space 보존.

---

## Frame C — "Sub-H 진입이 시기상조 (H019 cloud result 회수 전)" — **RESOLVED 2026-05-06**

> **Frame C REFUTED 2026-05-06**: H019 cloud measurable PASS CONFIRMED (platform 0.839674, Δ vs H010 corrected +0.001868pt = §17.3 measurable band). Paradigm shift family ceiling-breaker confirmed. H020 launch 정당.
>
> 아래는 historical record (Frame C 가 raise 됐던 당시의 우려).



**가설**: H019 자체의 cloud platform AUC 측정 미완료. local sweep 만으로 mechanism 진위 단정 위험. H019 cloud 가 noise/REFUTED 면 H020 도 noise/REFUTED 가능성 매우 높음 → sub-H 비용 낭비.

**근거**:
- H019 verdict = pending (cloud submission ready, 결과 회수 전).
- F-A baseline: best_val 이 platform 의 conservative estimate (~−0.003pt). H019 best_val 0.8372 → platform expectation 0.834~0.836. F-G ceiling band 안.
- H019 가 cloud noise 면 retrieval mechanism class 자체 ceiling 못 풂 → H020 도 같은 운명.

**Falsification 조건**: H019 cloud platform Δ vs H010 corrected ≥ +0.001pt (PASS measurable+) → Frame C REFUTED.

**Frame C confirmed (H019 cloud noise) 시 carry-forward**: H020 launch 보류. paradigm shift class (HSTU trunk / cohort attack) 으로 직접 pivot. retrieval class 은 ceiling 못 푸는 family confirmed.

**Counter to Frame C (왜 그래도 H020 진행 가능)**: 사용자 결정 — H019 cloud 결과 wait 비용 vs H020 사전 build 의 카드 작성 cost (학습 비용 발생 안 함). H020 카드 + upload_patch 미리 ready, 실제 cloud submit 은 H019 cloud 회수 후 결정 가능.

---

## Counter-argument 종합 (왜 그래도 H020 진행)

1. **Cheap structural exploration**: scoring axis 가 retrieval mechanism 안 가장 명확한 lever. H019 sweep saturation 후 다음 자연스러운 깊이.
2. **Paper-faithful test**: H019 의 simplified GSU 가 paper 와 다른 점 직접 검증 — mechanism class 진위 측정의 정밀도 향상.
3. **Single mutation 깔끔**: 64K params 추가, 다른 모든 hyperparam byte-identical. 결과 해석 ambiguity 없음.
4. **Decision tree clean**: 4 outcome (strong/measurable/noise/degraded) 모두 다음 H 결정 명확 (H021 stack / H022′ ESU / cohort pivot).
5. **Cost-effective**: T2.4 ~3.5h × $5-7. cost cap (campaign $100) 친화.
