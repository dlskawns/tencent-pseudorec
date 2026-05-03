# H012 — Verdict (REFUTED — Frame B confirmed, uniform routing + slight degradation)

> 클라우드 학습 + inference 완료 (2026-05-02). Platform AUC **0.838047** vs
> H010 anchor 0.8408 **−0.0028pt** → predictions.md decision tree 의 noise/degraded
> 경계. **MoE gate entropy = [1.378, 1.363]** vs uniform threshold log(4)=1.386
> → **uniform routing (Frame B confirmed)**. expert 가 specialization 학습
> 못함 — H010 NS xattn 이 이미 dominant signal 을 mixed 형태로 학습 중이라
> explicit MoE 가 redundant + 약간의 noise 추가.

## Status
`done` — **REFUTED — Frame B (uniform routing)**. Platform AUC **0.838047**
vs H010 0.8408 **−0.0028pt**. Δ vs H008 (0.8387) **−0.0007pt**. expert
specialization 안 일어남, mechanism class 의 효과 입증 실패.

**Critical context — measurement regime**:
- 사용자가 batch_size=2048 (default 256 의 8×) 로 학습 진행.
- lr 동일 (1e-4 추정) → linear scaling rule 미적용 → effective lr 1/8.
- H006~H012 모두 같은 underpowered regime — 상대 비교 (paired Δ) valid 하지만
  **절대 lift 작은 것이 mechanism 한계가 아닐 수 있음**.
- → H013 = hyperparameter calibration 으로 ceiling 의 정체 결정.

## Source data
- 학습: 10 epoch, train_ratio=0.3, batch_size=2048, label_time split + 10% OOF,
  **2시간 49분 43초 wall** (H011 2:46:54 와 거의 동급, H010 3:44:54 보다
  −24%).
- Inference: §18 정상 통과 **137.39초 wall** (H010 297초 −54%).
- ckpt: best step (raw metrics.json paste 도착 시 보강).

## P1 — Code-path success
- Measured: 학습 NaN-free 완주. `Training complete!` 로그 + metrics.json
  생성. MultiDomainMoEBlock dispatch 정상.
- Verdict: **PASS**.

## P2 — Primary lift (§17.3 binary)
- Measured: **Platform AUC = 0.838047** (eval auc).
- OOF AUC: **0.8589**, OOF LogLoss: **0.2335**.
- Δ vs H010 anchor (0.8408): **−0.0028pt**.
- Δ vs H008 (0.8387): **−0.0007pt**.
- Δ vs H011 (0.8347): +0.0033pt (H011 보다는 낫지만 anchor 미달).
- Predicted classification (predictions.md):
  - strong_pass ≥ +0.005pt: 미달.
  - measurable [+0.001, +0.005pt]: 미달.
  - noise (−0.001, +0.001pt): 경계 근접.
  - **degraded < −0.001pt** + uniform routing: **현재 결과**.
- **Verdict: REFUTED**. expert mechanism 효과 없음 + 약간의 noise 추가.

## P3 — MoE gate entropy (mechanism check) **Critical**
- Measured: **`moe_gate_entropy_per_block = [1.3779, 1.3629]`**.
- collapse_threshold: 0.6931 (= 0.5 × log(4)).
- uniform_threshold: 1.3863 (= log(4)).
- **결과 = [1.378, 1.363] ≈ uniform threshold** (격차 0.009 ~ 0.024).
- Predicted classification: specialized [0.69, 1.30] | **uniform > 1.34** | collapse < 0.69.
- **Verdict: uniform** — gate routing 가 4 expert 에 거의 균등 분배. expert
  specialization 학습 못 함.
- **Frame B 강한 confirm**: H010 NS xattn 의 selective routing 이 이미
  dominant signal 을 mixed 형태로 학습 중. explicit MoE 가 redundant 일
  뿐만 아니라 ~66K extra params 가 noise 추가 → 약간의 degradation.

## P4 — §18 인프라 통과
- Measured: inference 137초 wall, eval auc 0.838047 산출 (≠ 0.5 fallback).
- Verdict: **PASS**.

## P5 — val ↔ platform 정합 (보너스)
- val_AUC: TBD (raw metrics.json paste 시 보강).
- platform_AUC: 0.838047.
- TBD.

## P6 — OOF-platform gap (보너스, cohort drift 모니터)
- Measured: OOF 0.8589 − Platform 0.838 = **2.10pt**.
- 비교 baseline: H006 3.5 → H010 1.88 → H011 2.42 → **H012 2.10**.
- Predicted: ≤ 2pt. **경계 미달** (2.10 > 2).
- Cohort drift hard ceiling 가설 (H011 F-5) 7개 H 누적 데이터로 강화.

## Findings (F-N carry-forward)

- **F-1 (Frame B confirmed — uniform routing)**: MoE gate entropy [1.378, 1.363]
  ≈ uniform threshold 1.386. expert 가 specialization 학습 못 함. H010 NS
  xattn 의 selective routing (entropy 0.81) 이 이미 dominant signal 을
  mixed/sparse 형태로 capture 중 → explicit expert routing 이 redundant.
  carry-forward: **NS xattn 위에 어떤 NS-token level mechanism 도 marginal
  가능성 큼**. H013+ 부터 다른 axis (long-seq / cohort / capacity) 우선.
- **F-2 (Hyperparameter measurement bias 노출)**: 사용자 batch=2048 +
  default lr=1e-4 → effective lr 1/8 underpowered. 7개 H 모두 같은 regime
  → 상대 paired Δ 는 valid 하지만 절대 lift 작은 게 ceiling 의 진짜 정체
  미확정. carry-forward: **H013 = hyperparameter calibration** (lr linear
  scaling 또는 batch 복귀) 우선 수행 후 ceiling 정체 결정.
- **F-3 (4-layer ceiling diagnosis)**:
  - L1 hyperparameter regime (F-2).
  - L2 OOF-platform gap 1.9~2.4pt 일관 (cohort/temporal drift, H011 F-5
    누적 confirm).
  - L3 H010 NS xattn 의 sparse selective routing 이 dominant signal 캡처
    (~2 tokens / 384) → 후속 mechanism marginal.
  - L4 truncate 64-128 의 95%+ 정보 손실 (§3.5 p90 1393~2215).
  carry-forward: **H013 = L1 (calibration) → 결과 따라 L2 (cohort) 또는
  L4 (long-seq) 진행**.
- **F-4 (multi_domain_fusion 카테고리 retire 후보)**: explicit expert routing
  이 sample-scale (또는 underpowered regime) 에서 specialization 못 함.
  category 재진입 시 강한 정당화 필요 (예: PLE progressive separation,
  hard top-K routing). carry-forward: backlog 후순위.
- **F-5 (cost — 누적 ~24h)**: H006~H012 누적 ~24시간. §17.6 cap 압박
  지속. H013 calibration 은 H010 envelope (~3h). 누적 ~27h.
- **F-6 (anchor 갱신 안 함)**: H010 (Platform 0.8408) 여전히 champion.
  H011/H012 둘 다 REFUTED. H013+ control = H010.

## Surprises
- **Gate entropy 1.378 = uniform 거의 확정**: 0.024 격차 (1.386 − 1.362)
  로 학습이 uniform 에서 거의 안 멀어짐. random init (sanity test 1.378)
  와 거의 동일 → **gate 가 거의 학습 안 됨**. underpowered regime (F-2)
  의 직접 신호일 가능성. 또는 NS-token 7개의 input 자체가 expert
  specialization 학습할 정보 부족.
- **Δ vs H011 = +0.0033pt**: H012 가 H011 보다는 나음. 그래도 H010 anchor
  미달. H011 (input-stage) 보다 H012 (NS-level) 가 안전한 위치 stacking
  이라는 점은 confirm.
- **Inference wall −54%**: H010 297초 → H012 137초. params 추가 66K 인데
  학습/infer 모두 단축. **IO bound 신호** (GPU idle, 데이터 로딩 bottleneck).
  carry-forward: H013 의 num_workers / buffer_batches 늘리는 데이터 로딩
  최적화도 같이 시도.

## Update to CLAUDE.md?
- §17.3 의 binary 임계 (current Δ ≥ +0.5pt full-data, +0.001pt sample-scale
  relaxed) 가 **measurement regime 정합성 검증 의무 추가** 필요. 다음 §X
  추가 후보: "**hyperparameter regime 변경 시 anchor 재측정 의무**" (linear
  scaling rule 미적용 시 모든 prior 결과 재해석 필요).
- 본문 갱신은 H013 결과 후 결정 (calibration 효과 정량 확보 후).
- §3.5 의 4-layer ceiling 요약 추가 후보 (현재까지 7 H 누적 evidence).

## Carry-forward to H013

- F-1 → NS-token level mechanism 추가 retire. H013 = hyperparameter calibration
  또는 long-seq 또는 cohort 처리 우선.
- F-2 → **H013 1순위 = hyperparameter calibration** (lr 1e-4 → 8e-4 또는
  batch 2048 → 256). 측정 base 정합성 확보가 다음 mechanism H 의 전제.
- F-3 (4-layer ceiling) → H013 결과 따라 L2/L4 결정.
- F-5 → H013 cost ~3h, 누적 ~27h. §17.6 cap 임박.
- F-6 → control = H010, H008 carry-forward 유지.

## Decision applied (per predictions.md decision tree)

predictions.md table:
- noise (Δ ∈ (−0.001, +0.001) + uniform): 경계 근접 (Δ −0.0028 가 noise 임계
  살짝 미달).
- **uniform routing 자체로 Frame B 강한 confirm** → effective verdict:
  REFUTED, expert mechanism 효과 입증 실패.

card.yaml decision_tree_post_result `noise` 분기 + slight degradation
hybrid 적용:
- → "Frame B 채택. anchor 갱신 안 함. H013 = NS xattn sub-H 또는 cohort
  처리 (H011 F-5)."
- 추가 (F-2 carry-forward): **H013 = hyperparameter calibration 우선**.
