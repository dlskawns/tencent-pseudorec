# H014 — Verdict (REFUTED — L4 도 ceiling, 4-layer diagnosis 의 L1+L3+L4 모두 retire)

> 클라우드 학습 + inference 완료 (2026-05-03, OOM mitigation iter-4 setup).
> Platform AUC **0.833587** vs H010 corrected anchor 0.837806 **−0.0042pt**
> = REFUTED. Long-seq dense expansion 가 ceiling 풀지 못함. **L4 retire
> confirmed**. 4-layer ceiling diagnosis 의 L1 + L3 + L4 모두 retire →
> **L2 (cohort drift) 만 남음**. OOF-Platform gap 2.59pt = 9 H 중 가장 큼,
> Frame B (cohort hard ceiling) 강한 confirm.

## Status
`done` — **REFUTED — L4 retire**. paradigm shift inevitable.

## Critical context (eval data correction)
- Organizer 가 eval data 수정 (2026-05-02~03). H010 재측정 = **0.837806**
  (이전 잘못된 eval 0.8408 대비 −0.003 shift).
- **prior H 들 (H011/H012/H013) 은 corrected eval 로 재측정 안 됨** —
  ranking 비교 정밀도 낮음. 단 H014 = corrected eval 로 직접 측정.
- H010 corrected 가 새 anchor.

## Source data
- 학습 setup: batch=1024, lr=1e-4, seq_max_lens "192-192-192-192" uniform,
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (OOM mitigation iter-4
  최종).
- 학습 wall: **3시간 30분 49초** (H010 3:44:54 −6%).
- Inference wall: **229.87초** (H010 corrected 92.74s **+148%**).

## P1 — Code-path success
- Measured: NaN-free 완주. OOM 없음 (iter-4 setup 안전 확인).
- Verdict: **PASS**.

## P2 — Primary lift (§17.3 binary)
- Measured: Platform AUC **0.833587**, OOF **0.8595**.
- Δ vs H010 corrected (0.837806): **−0.0042pt**.
- Δ vs H010 prior (0.8408, 잘못된 eval): −0.0072pt.
- Δ vs H012 prior (0.8380): −0.0044pt.
- Δ vs H013 (0.8344): −0.0008pt.
- Predicted classification: strong | measurable | noise | **degraded** ← 적용.
- **Verdict: REFUTED (degraded)**.

## P3 — NS xattn entropy 변화 (mechanism interpretation)
- Measured: `attn_entropy_per_layer = [0.9915, 0.9920]`.
- H010 baseline: [0.8127, 0.8133].
- Δ: **+0.18** (덜 sparse).
- 새 threshold: 5.9264 (= 0.95 × log(seq_total)). 변화 미세 위반 안 함.
- **해석**: long-seq (seq total 384 → 768) 에서 attention 후보 2× 증가.
  e^0.99 / 768 ≈ 0.35% vs e^0.81 / 384 ≈ 0.58% → 비율적으로 더 sparse,
  절대 token 수 ~2-3 유지. **dominant signal sparse capture pattern 유지** —
  long-seq 의 정보 회복이 routing 에 흡수 안 됨.
- Verdict: 변화 측정 가능 (덜 sparse), 단 mechanism class 효과는 marginal.

## P4 — §18 인프라 통과
- Measured: inference 229.87s, eval auc 0.833587 ≠ 0.5.
- Verdict: **PASS**.

## P5 — val ↔ platform 정합
- TBD.

## P6 — OOF-platform gap (cohort drift 모니터)
- Measured: 0.8595 − 0.8336 = **2.59pt**.
- 9 H 중 **가장 큼** (H010 1.88 / H011 2.42 / H012 2.10 / H013 2.29 / H014 2.59).
- **Frame B (cohort drift hard ceiling) 강한 confirm**: long-seq 가 OOF 만 보존
  (0.8595 = H010 0.8596 거의 동일), Platform 더 악화 (cohort drift 증폭).

## P7 — Wall efficiency
- Measured: 학습 3:30:49 (H010 3:44:54 −6%), inference 229.87s (H010 92.74s **+148%**).
- batch 절반 (effective per-step compute 절반) + seq 3× (attention compute
  per-step ~9× for a/b, 2.25× for c/d) = 학습 거의 같음.
- inference 는 batch 영향 적고 seq 만 영향 → +148% 일관.

## Findings (F-N carry-forward)

- **F-1 (L4 retire)**: dense long-seq expansion (seq 192 uniform) 가 H010
  anchor 위 −0.0042pt 악화. truncate 정보 손실이 ceiling 의 진짜 정체 아님.
  **4-layer diagnosis 의 L4 retire confirmed**. carry-forward: 단순 envelope
  expansion 무용 → retrieval/compression (TWIN/SIM/HSTU) 도 효과 의문.
- **F-2 (cohort drift hard ceiling 강한 confirm)**: OOF-Platform gap 2.59pt
  = 9 H 중 가장 큼. long-seq 가 OOF 만 향상, Platform 악화 — cohort drift
  증폭 신호. **Frame B 가 ceiling 의 진짜 정체** 가능성 매우 높음.
- **F-3 (4-layer diagnosis 종료)**: L1 (hyperparameter) ❌ + L3 (NS xattn
  sparse) ❌ + L4 (truncate) ❌ retire. **L2 (cohort drift) 만 남음**.
  paradigm shift inevitable.
- **F-4 (NS xattn routing pattern 유지)**: entropy 0.99 (vs H010 0.81) 절대값
  증가지만 비율 (e^entropy / total) 더 sparse. dominant signal sparse capture
  patterns long-seq 에서도 유지. mechanism class 의 한계 강한 confirm.
- **F-5 (anchor 갱신)**: H010 corrected (0.837806) = 새 baseline (organizer
  eval data 수정). prior H 측정값들 (잘못된 eval) 직접 비교 invalid.
  H011/H012/H013 corrected 재측정 권장.
- **F-6 (eval data correction 영향)**: 사용자 직관 "H012 가 가장 높다" 는
  H010 corrected (0.837806) vs H012 prior (0.838047) 비교 = invalid (다른
  eval data). 단 H012 가 corrected 로 재측정 시 H010 와 비슷한 −0.003 shift
  적용 시 H012 corrected 추정 ~0.835 < H010 corrected 0.838.
- **F-7 (cost — 누적 ~32h)**: H006~H014. §17.6 cap 임박 — 한계 도달.

## Surprises
- **OOF AUC 0.8595 = H010 0.8596 와 거의 동일**: 9 H 모두 OOF 0.857~0.860.
  모델 capacity 의 OOF 한계 hit. **Platform 만 변동 = cohort drift 가 진짜
  ceiling** 가설 강한 confirm.
- **L4 가 cohort drift 증폭** 가능성: long-seq 가 train cohort 더 fit (OOF
  유지), platform cohort transfer 더 악화 (gap 2.59pt). 이전 H 들 패턴
  반복.
- **eval data correction shift**: H010 prior 0.8408 → corrected 0.837806 =
  −0.003. 만약 같은 shift 가 모든 H 에 적용 시 ranking 유지. 단 H012
  prior 0.838047 가 corrected 시 어떻게 변할지 불확실.

## Update to CLAUDE.md?
- §17 새 룰 후보: "**4-layer ceiling diagnosis 종료 시 backbone replacement
  또는 cohort 처리 paradigm shift mandatory**" — H014 결과로 trigger.
- §10.3 challenger rule 강화 후보: "3회 연속 mechanism mutation REFUTED 후
  envelope 도 REFUTED 시 paradigm shift inevitable."
- 본문 갱신 보류 (H015 결과 누적 후 결정).

## Carry-forward to H015

- F-1 → L4 retire. retrieval/compression 도 동일 cohort drift hit 가능성.
- F-2 → **L2 (cohort drift) 가 마지막 가설**. H015 = cohort H 우선.
- F-3 → 4-layer diagnosis 종료, paradigm shift inevitable.
- F-5 → H011/H012/H013 corrected 재측정 mandatory (verify-claim sanity gate
  정합성). H015 진행 전 또는 병렬.
- F-6 → 사용자 "H012 가장 높다" 결론은 invalid comparison. corrected ranking
  명확히 한 후 결정.

## Decision applied (per predictions.md decision tree)

- "noise (Δ ≤ +0.001pt)" 또는 "degraded (Δ < −0.001pt)" 분기 적용.
- card.yaml `noise` decision tree → "Dense expansion 효과 없음. retrieval/
  compression mandatory. H015 = TWIN/SIM (retrieval) 또는 cohort H (Frame B).
  paradigm shift inevitable."
- **결정**: L4 retire + L2 만 남음 → **H015 = cohort drift 처리 H** 우선.
- 단 retrieval (TWIN/SIM) 도 valid 후보 — L4 가 dense expansion 형태로만
  retire, retrieval form 으로는 새 axis 가능.
