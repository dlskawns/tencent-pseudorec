# H010 — Predictions

> Pre-registered before run.

## P1 — Code-path success
- Quantity: train.py NaN-free 완주 (또는 patience=3 early stop 정상), `metrics.json` 생성.
- Predicted: NaN 0건, finite val/OOF AUC. NSToSCrossAttention dispatch + DCN-V2 fusion 둘 다 정상.
- Falsification: NaN abort, OOM, NSToSCrossAttention 라우팅 오류 → REFUTED + 인프라 디버깅.

## P2 — Primary lift (§17.3 binary)
- Quantity: extended envelope **platform AUC** vs anchor (original_baseline) ~0.83X.
- Predicted: Δ ≥ **+0.5pt** (binary 임계).
- Falsification: Δ < +0.5pt → REFUTED. target_attention 일반화 가설 약화.

## P2-sub — Paired vs H008 strongest single
- Quantity: Δ vs H008 (Platform 0.8387, current champion).
- Predicted classification:
  - **super-additive** (Δ vs H008 ≥ +0.005pt): paper-grade lift, NS×S layer-level 통합 가치 confirmed. Platform ≥ 0.8437.
  - **additive** (Δ vs H008 ∈ [+0.001, +0.005pt]): mechanism class 강화 confirm. Platform ∈ [0.8397, 0.8437].
  - **noise** (Δ vs H008 ∈ [−0.001, +0.001pt]): NS xattn 효과 marginal. H007 일반화 가설 약화.
  - **interference** (Δ vs H008 < −0.001pt): H009 와 같은 위치 충돌 패턴. transfer.md 의 위치 충돌 회피 설계 가설 무효 — REFUTED, mechanism class 한계.

## P3 — §10.9 attention entropy
- Quantity: NSToSCrossAttention 의 layer 별 평균 attn entropy.
- Threshold: 0.95 × log(384) ≈ **5.65**.
- Predicted: entropy < 5.65 (uniform collapse 부재). H004 OneTrans backbone smoke 결과 (entropy [3.49, 3.91] / threshold 5.67) 의 sparse pattern 재현 expected.
- Falsification: entropy ≥ 5.65 → §10.9 룰 abort signal. mechanism degenerate to uniform attention. REFUTED + sample-scale 한계 분류.

## P4 — §18 인프라 통과
- Quantity: inference 시 §18 룰 모두 충족 (batch heartbeat + `[infer] OK` 로그 + platform AUC ≠ 0.5).
- Predicted: PASS (H008 패키지 inherit + 두 cfg key 추가).
- Falsification: P4 fail → §18 회귀, infer.py cfg.get 추가 부분 디버깅.

## P5 — val ↔ platform 정합 (보너스)
- Quantity: |val_AUC − platform_AUC|.
- Predicted: ≤ 0.05 (H006/H007/H008 패턴 재현). val_AUC 가 platform 의 useful proxy.
- Falsification 아님 — bonus.

## P6 — Mechanism check (NS xattn 작동)
- Quantity: NSToSCrossAttention 의 attention weight 분포 (가능 시).
- Predicted: 각 NS token 이 4 도메인 S tokens 에 nontrivial attention spread (도메인 자동 routing).
- Falsification 아님 — 보조 신호.

## Reproducibility
- compute_tier: T2.4 extended (10 epoch × 30%, patience=3) ~3시간.
- seed: 42.
- split: label_time + 10% OOF (anchor 동일).
- expected wall: H008 (3.7h) ~ H009 (3.6h) 범위. patience=3 + plateau early 가정 시 ~2.5-3.5시간.
- code: `experiments/H010_ns_to_s_xattn/upload/` (12 파일, run.sh 가 H010 + H008 flags baked).

## Negative-result interpretation (§17.7 falsification-first)

본 H REFUTED 시:

- **P2 fail with Δ vs H008 ∈ (−0.001, +0.001pt)** (noise): NS xattn 일반화 효과 marginal. H007 의 1-token candidate 가 sufficient 였고 7-token 일반화 가치 작음. **carry-forward**: NS xattn layer 수 (1→2) 또는 num_heads 변경 sub-H. 또는 NS tokens 의 routing 부족 가설 → 도메인 ID embedding 추가 sub-H.
- **P2 fail with Δ vs H008 < −0.001pt** (interference): H009 와 같은 위치 충돌 패턴 — transfer.md 의 회피 설계 가설 무효. **carry-forward**: NS xattn 의 통합 위치를 query decoder 이후 (decoded_q 와 enriched NS tokens 가 같은 fusion block 입력) 로 변경 sub-H. 또는 mechanism class (target_attention) 자체가 single-token (H007) 으로만 작동하는 한계 가설.
- **P3 fail (attn entropy ≥ 5.65)**: §10.9 룰 abort. attention pattern uniform collapse — sample-scale 한계 또는 NS tokens 의 query semantic 약함. **carry-forward**: NS-token granularity 변경 (5+2 → 8+4) 또는 hard routing (top-K) sub-H.
- **P4 fail**: §18 회귀. infer.py 의 cfg.get 추가 부분 검증.

## Decision tree (post-result)

| Result | Next action |
|---|---|
| Δ vs anchor ≥ +0.5pt + super-additive vs H008 + P3/P4 PASS | H010 PASS paper-grade. **anchor 갱신**: H010 = 새 baseline. H011 = orthogonal axis (multi_domain_fusion MMoE/PLE 또는 aligned `<id, weight>` pair encoding). |
| Δ vs anchor ≥ +0.5pt + additive vs H008 | H010 PASS. anchor 갱신 (H010 = 새 baseline). H011 = 다른 axis 탐험. |
| Δ vs anchor ≥ +0.5pt + noise vs H008 | H010 PASS marginal. anchor 갱신 결정 보류 (H008 vs H010 paired Δ 작음). H011 = orthogonal axis 또는 NS xattn sub-H (layer 수 / num_heads). |
| Δ vs anchor ≥ +0.5pt + interference vs H008 | H010 PASS at anchor but interference vs champion. anchor 갱신 안 함. H011 = NS xattn 통합 위치 변경 sub-H 또는 다른 mechanism class. |
| Δ vs anchor < +0.5pt | REFUTED. target_attention mechanism class 의 일반화 가치 의문. H011 = aligned pair encoding (orthogonal axis, interference 위험 0) 또는 multi_domain_fusion. |
| P3 fail (attn entropy ≥ 5.65) | sample-scale 한계 또는 NS query semantic 약함. abort. NS-token granularity 또는 hard routing sub-H. |
| P4 fail | §18 회귀. infer.py 디버깅 우선. |
