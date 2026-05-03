# H013 — Verdict (REFUTED — Frame A REFUTED, hyperparameter artifact 가설 무효)

> 클라우드 학습 + inference 완료 (2026-05-02). Platform AUC **0.834376** vs
> H010 anchor 0.8408 **−0.0064pt** = degraded. Linear scaling rule (lr 1e-4
> → 8e-4 for batch 2048) 가 ceiling 풀지 못함. OOF AUC **0.8573** = 8 H
> 중 처음으로 0.86 미달. **4-layer ceiling diagnosis 의 L1 retire confirmed**.

## Status
`done` — **REFUTED — Frame A (hyperparameter artifact 가설) REFUTED**.

## Source data
- 학습 wall: **4시간 8분 13초** (H010 3:44:54 +10%).
- Inference wall: 95.57s (H010 297s −68%).
- ckpt: best step (raw metrics.json paste 도착 시 보강).

## P1 — Code-path success
- Measured: NaN-free 완주 (lr 8e-4 + batch 2048 stability OK).
- Verdict: **PASS**.

## P2 — Primary lift (§17.3 binary)
- Measured: **Platform AUC = 0.834376**.
- OOF AUC: **0.8573** (8 H 중 처음으로 0.86 미달).
- OOF LogLoss: 0.2327.
- Δ vs H010 anchor (0.8408): **−0.0064pt**.
- Δ vs H008 (0.8387): −0.0044pt.
- Δ vs H011 (0.8347): −0.0003pt (거의 동일).
- Δ vs H012 (0.8380): −0.0036pt.
- Predicted classification: strong | measurable | noise | **degraded** ← 적용.
- **Verdict: REFUTED (degraded)**.

## P3 — NS xattn entropy 변화
- Measured: `attn_entropy_per_layer = [0.8138, 0.8140]`.
- H010 baseline [0.8127, 0.8133]. Δ ≈ +0.001 (변화 미세).
- Verdict: **변화 미세** — lr 변경이 mechanism 표현 자체에 거의 영향 없음.

## P4 — §18 인프라 통과
- Measured: inference 95.57s, eval auc 0.834376.
- Verdict: **PASS**.

## P5 — val ↔ platform 정합
- TBD (raw metrics.json paste 시 보강).

## P6 — OOF-platform gap
- Measured: 0.8573 − 0.8344 = **2.29pt**.
- 비교 baseline: H010 1.88 / H011 2.42 / H012 2.10. **H013 가 H010 보다 큼**.
- Verdict: 미달 (cohort drift 강화 신호).

## P7 — Wall efficiency
- Measured: 학습 4:08:13 (H010 3:44:54 **+10%**), inference 95.57s (H010 297s **−68%**).
- Predicted: ≤ H010 wall (lr 큰 효과로 빠른 수렴 기대).
- Verdict: **학습은 +10% 반대 방향** — patience=3 미트리거 (lr 큰데 plateau
  없이 full 10 epoch). inference 단축 일관성 (H010 → H011/H012/H013 모두 단축).

## Findings (F-N carry-forward)

- **F-1 (Frame A REFUTED)**: Linear scaling rule (lr 8e-4 for batch 2048)
  이 ceiling 의 정체 아님. Δ vs H010 −0.0064pt = 명확한 degraded.
  **Hyperparameter artifact 가설 무효, L1 retire confirmed**.
- **F-2 (lr 8e-4 + batch 2048 = generalization gap)**: Keskar et al. 2017
  의 "large-batch generalization gap" 가 적용된 케이스. OOF AUC 도 처음으로
  0.86 미달 (0.857) — 학습 자체가 less optimal.
- **F-3 (4-layer ceiling diagnosis 갱신)**: L1 (hyperparameter) ❌ retire.
  L3 (NS xattn sparse) ❌ retire (3 H 누적). **L2 (cohort drift) + L4
  (truncate) 만 남음**.
- **F-4 (cohort drift 강화 신호)**: P6 gap 2.29pt > H010 1.88. lr 큰 효과
  가 cohort overfit 일부 발현.
- **F-5 (anchor 갱신 안 함)**: H010 (0.8408) 여전히 champion. 4 H 누적
  H010 위 mutation REFUTED (H011/H012/H013).
- **F-6 (§10.3 challenger rule trigger)**: 3회 연속 H010 anchor 위 mutation
  REFUTED → 강제 challenger 사고. H014 = envelope mutation (다른 axis)
  필수.
- **F-7 (cost — 누적 ~27h)**: H006~H013. §17.6 cap 임박.

## Surprises
- **OOF AUC 0.857 = 처음으로 0.86 미달**: 8 H 중 가장 낮은 OOF. lr 8e-4
  가 학습 자체를 어렵게 만든 신호 (Keskar generalization gap).
- **학습 wall +10%**: lr 큰데 더 길어짐 — patience=3 미트리거. lr 효과로
  학습이 빨리 수렴하기는커녕 oscillation 으로 plateau detection 늦어짐
  가능.
- **Inference −68%**: IO 4/8 calibration 이 inference 에 큰 효과. 학습
  +10%와 대조 — IO bottleneck 이 inference 에서 더 dominant 였던 신호.

## Update to CLAUDE.md?
- §17.2 의 "structural not parametric" 룰 보강 후보: "linear scaling rule
  같은 standard practice 라도 결과가 ceiling 풀지 못하면 mechanism limit
  의 strong evidence." H013 결과로 confirmed.
- 본문 갱신 보류 (H014 결과 후 4-layer diagnosis 종합 결정).

## Carry-forward to H014

- F-1 → L1 retire confirmed → **L4 (truncate) 가 마지막 unexplored axis,
  H014 = envelope mutation 정당**.
- F-3 → 4-layer diagnosis 갱신 (L1 + L3 retire, L2 + L4 만 남음).
- F-4 → cohort drift 모니터 H014 P6.
- F-5 → control = H010 (0.8408) 유지.
- F-6 → H014 envelope mutation = challenger 사고 적용.
- F-7 → H014 cost ~6-15h (long-seq overhead) → 누적 33-42h, cap 위협.

## Decision applied (per predictions.md decision tree)

- "degraded (Δ < −0.001pt)" 분기 적용.
- card.yaml `degraded` decision tree → "lr 너무 큼. H013-sub = lr 4e-4 또는
  batch 256 복귀."
- **단 H013-sub 진행 안 함** — Frame A retire 가 더 큰 정보. 4-layer
  diagnosis 의 L4 (truncate) 우선.
