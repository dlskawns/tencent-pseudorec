# H011 — Predictions

> Pre-registered before run. 반증 가능.

## P0 — Audit gate (학습 시작 전, ns_groups.json 검증값 reference)

verified aligned set (출처: `competition/ns_groups.json` `_note_shared_fids`)
= **`{62, 63, 64, 65, 66, 89, 90, 91}`** (8 fids). 본 P0 는 새 EDA 가 아닌
documented mapping 의 implementation 일관성 검증.

- **Quantity 1 (offset/dim 추출)**: aligned fid k 별 `user_dense_feats` slice
  의 (offset_k, dim_k) 를 `competition/dataset.py:_user_dense_plan` 에서
  추출 → `eda/out/aligned_offsets.json` 산출.
- **Quantity 2 (array length 검증)**: `demo_1000.parquet` 의 per-row
  `user_int_feats_k` array length `n_k` 측정 (mean / max / per-row 분포).
  `n_k == dim_k` 검증 필수.
- **Quantity 3 (dense-only 제외 확인)**: `{61, 87}` 가 user_int 측에 없음
  재확인 (CLAUDE.md §3 verified facts 일치).
- Predicted: PASS — verification 일치 (ns_groups.json 의 `_note_shared_fids`
  = ground truth).
- Falsification:
  - `n_k != dim_k` for any k → binding semantics 불명. **INVALID**, H011
    retract 또는 mechanism 재설계 (broadcast 또는 group-level reduction).
  - aligned fid k 의 `user_dense_feats` slice 가 dataset 에서 사라짐 →
    schema 변경 의심, ns_groups.json stale 가능성. 사용자 확인.

## P1 — Code-path success

- Quantity: train.py NaN-free 완주, `metrics.json` 생성.
- Predicted: NaN 0건, finite val/OOF AUC. weighted embedding multiply +
  H010 NS xattn + DCN-V2 fusion 모두 정상.
- Falsification: NaN abort, OOM, dense feature scale issue → REFUTED +
  scale handling 디버깅 (LayerNorm 추가 또는 sigmoid 게이팅).

## P2 — Primary lift (§17.3 binary)

- Quantity: extended envelope **platform AUC** vs anchor (H010 0.8408).
- Predicted classifications:
  - **strong PASS** (Δ ≥ +0.005pt vs H010): aligned binding 의 explicit
    handling 효과 큼. Platform ≥ 0.8458.
  - **measurable lift** (Δ ∈ [+0.001, +0.005pt]): mechanism 작동, marginal
    confirmed. Platform ∈ [0.8418, 0.8458].
  - **noise** (Δ ∈ [−0.001, +0.001pt]): baseline 이 이미 implicit binding
    학습 충분. Frame B (NS-level binding) 채택.
  - **degraded** (Δ < −0.001pt): scale issue 또는 fid 매핑 잘못 (negative
    control 역할). REFUTED + 매핑 재확인.
- Falsification (binary): Δ < +0.001pt → §17.3 strict 미달. mechanism class
  retire 또는 sub-form 시도.

## P3 — NS xattn entropy 변화 (mechanism check)

- Quantity: H010 NS xattn `attn_entropy_per_layer` 와 비교. H010 baseline
  = [0.8127, 0.8133].
- Predicted classifications:
  - **더 sparse** (entropy < 0.7): weighted embedding 이 NS-token routing
    더 집중시킴. input-stage binding 이 NS 표현을 더 정보 집약. A frame
    (input-stage 효과) 강한 신호.
  - **변화 미세** (|Δ| < 0.05): NS-level routing 이 input 변경 흡수, NS
    output 거의 동일. B frame (NS-level 충분) 신호.
  - **더 uniform** (entropy > 1.0): weighted embedding 이 NS-token 표현
    혼란 → routing 약화. negative signal — scale 또는 매핑 issue.
- Falsification 아님 — mechanism 작동 진단.
- Threshold: §10.9 룰 violation (entropy ≥ 5.65) 모니터.

## P4 — §18 인프라 통과

- Quantity: inference 시 §18 룰 모두 충족 (batch heartbeat + `[infer] OK`
  로그 + platform AUC ≠ 0.5).
- Predicted: PASS (H010 패키지 inherit + 3 cfg key 추가).
- Falsification: P4 fail → §18 회귀, infer.py cfg.get 추가 부분 디버깅.

## P5 — val ↔ platform 정합 (보너스)

- Quantity: |val_AUC − platform_AUC|.
- Predicted: ≤ 0.05 (H006/H007/H008/H010 패턴 재현).
- Falsification 아님 — bonus.

## P6 — OOF-platform gap (보너스)

- Quantity: OOF AUC − platform AUC. H010 baseline = 1.88pt.
- Predicted: ≤ 2pt (H010 패턴 유지). input-stage enrichment 가 cohort
  effect 증폭 안 시키면 PASS.
- Falsification 아님 — bonus, capacity-overfit signal.

## Reproducibility

- compute_tier: T2.4 extended (10 epoch × 30%, patience=3) ~3-3.5시간.
- seed: 42 (seed×3 검토 — H012+ 부터 cost cap 압박으로 단일 seed 만 가능).
- split: label_time + 10% OOF (anchor 동일).
- expected wall: H010 (3.7h) 동급 또는 약간 짧음 (params 추가 0).

## Negative-result interpretation (§17.7 falsification-first)

본 H REFUTED 시:

- **P0 fail (audit fail)**: `n_k != dim_k` (user_int_feats_k array length
  vs user_dense slice dim mismatch) 검출. binding semantics 불명. **carry-
  forward**: mechanism 재설계 후보 — (a) `dim_k` 가 fixed 통계 summary 면
  broadcast (1 weight × n_k IDs); (b) group-level reduction (mean/sum) 후
  단일 weight 곱셈; (c) 매핑 자체가 ns_groups.json 과 다름 → 사용자 확인
  필요. eda/out/aligned_offsets.json + array_lengths.json 산출 mandatory.
- **P1 fail (NaN)**: dense feature scale issue 가능성 큼. weight 가 raw [0,
  1] 가 아닌 unbounded → embedding × weight 가 발산. **carry-forward**:
  sub-H = sigmoid 게이팅 또는 LayerNorm(weight) form.
- **P2 fail with Δ ∈ (−0.001, +0.001pt) (noise)**: NS-level cross-attention
  이 이미 binding 학습 충분 (Frame B). **carry-forward**: H012 = MMoE/PLE
  (Frame C 전환) 또는 aligned pair 의 multi-form (gated_multiply, log_weight)
  sub-H. 단 sub-form 은 §10.7 rotation 으로 H012 직후 차단 (feature_engineering
  재진입은 1회 PASS 또는 mechanism 직접 신호 필요).
- **P2 fail with Δ < −0.001pt (degraded)**: scale issue 또는 매핑 잘못 또는
  baseline 이 이미 implicit binding 더 정확하게 학습. **carry-forward**:
  매핑 재확인 + scale handling sub-form (sigmoid, log).
- **P3 entropy 변화 미세 + P2 noise**: B frame 강한 confirm. NS-level binding
  충분. H012 = MMoE 우선 (Frame C 전환).
- **P3 entropy 더 uniform + P2 degraded**: scale 또는 매핑 issue. retract
  + 재가설.
- **P4 fail**: §18 회귀, infer.py 디버깅.

## Decision tree (post-result)

| Result | Next action |
|---|---|
| Δ vs anchor (H010) ≥ +0.005pt + P0/P1/P3/P4 PASS | H011 strong PASS. **anchor 갱신**: H011 = 새 baseline. H012 = orthogonal axis (multi_domain_fusion 또는 NS xattn sub-H, rotation 룰 따라). |
| Δ vs anchor ≥ +0.001pt + measurable | H011 PASS. anchor 갱신 검토. H012 = 다른 axis 또는 aligned pair sub-form. |
| Δ vs anchor ∈ (−0.001, +0.001pt) (noise) | H011 noise. anchor 갱신 안 함. **Frame B 채택** → H012 = MMoE/PLE (Frame C 전환). |
| Δ vs anchor < −0.001pt (degraded) | H011 REFUTED + scale/매핑 issue. retract + 재가설 또는 sub-form. |
| P0 fail (audit) | H011 INVALID. 데이터 사실 재정립 (eda/out/aligned_fids.json) 후 재가설. |
| P3 entropy 더 uniform | scale 또는 매핑 issue 가능. P2 와 무관하게 매핑 재확인 + 다음 sub-H 시 sigmoid 게이팅. |
| P4 fail | §18 회귀. infer.py 디버깅 우선. |
