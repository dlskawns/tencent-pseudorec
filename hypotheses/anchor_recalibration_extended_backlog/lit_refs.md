# H010 — Literature References

> Anchor recalibration H. mechanism mutation 0 → 새 paper 인용 없음. prior H
> verdicts 인용만으로 minimal.

## Internal references (prior H verdicts)

- `hypotheses/H006_longer_encoder_d_domain/verdict.md` F-3:
  > "paired Δ 는 platform AUC 으로만."
  → 본 H 의 measurement target 도 platform AUC.

- `hypotheses/H007_candidate_aware_xattn/verdict.md` F-3 (직접 동기):
  > "H007 (extended envelope 3 epoch × 30%) Δ vs anchor (smoke 1 epoch × 5%)
  > 비교 unfair — envelope 도 변수. mechanism 효과만 isolate 하려면 anchor 도
  > extended 에서 측정 필요. 현재 anchor smoke 0.83X / H007 extended 0.8352 =
  > **mechanism 효과 + envelope 효과 합산 +0.005pt**. mechanism 단독 효과
  > separate 하려면 추후 anchor recalibration H 필요."
  → 본 H 가 정확히 이 미래 H 충족.

- `hypotheses/H007_candidate_aware_xattn/verdict.md` F-2:
  > "val ↔ platform 정합 confirm 두 번째 — 우리 split 의 val_AUC 가 platform 과
  > 정합."
  → 본 H 도 같은 패턴 expected. val_AUC 가 platform 의 useful proxy.

- `hypotheses/H008_dcn_v2_block_fusion/verdict.md` F-4:
  > "patience=3 + early stop aggressive 으로 cost 절약. extended 에서 plateau
  > 일찍 도달 패턴 H006/H007 모두 confirm."
  → 본 H envelope 에 patience=3 적용.

- `hypotheses/H008_dcn_v2_block_fusion/verdict.md` F-5:
  > "OOF-platform 갭 H006 3.5pt → H008 1.98pt 로 좁아지는 패턴. mechanism
  > 강화될수록 cohort effect 의 noise 가 상대적으로 줄어듦."
  → 본 H 결과의 OOF-platform 갭이 anchor 의 cohort effect baseline 측정.
  H006 시점으로 다시 벌어지면 mechanism 의 cohort effect 보정 효과 직접 측정.

- `hypotheses/H009_combined_xattn_dcn_v2/verdict.md` F-3 (정량 동기):
  > "anchor 정확값 의존성이 결론 분류 흔드는 것 직접 확인 — anchor 정확값
  > 0.83 가정 시 marginal pass, 0.835 가정 시 fail."
  → 본 H 가 이 정확값 의존성 해소.

## Method-class references (no new paper)

- 본 H 는 PCVRHyFormer baseline (organizer 코드) + label_time split + 10% OOF +
  §18 인프라 룰 그대로 재사용. paper 인용 없음.
- envelope (10ep × 30%, patience=3) 는 H006~H009 와 byte-identical envelope 정의
  → paper 가 아니라 우리 sweep 의 internal convention.

## Related (carry-forward)

- §17.5 sample-scale 룰: smoke = code-path verification, extended = 측정용. 본
  H 가 measurement objective (mechanism lift 측정 아님) 라 §17.3 binary
  ≥ +0.5pt 임계 적용 안 됨. measurement-only H 패턴 첫 적용.
- §10.6 sample budget cap: anchor 면제 (original_baseline 와 동일).
