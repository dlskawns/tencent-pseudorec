# H016 — Verdict (PENDING)

> 실행 후 작성. status = `BUILT`.

OOF 재정의 (label_time future-only holdout) — L2 (cohort drift) 의 다른 form.

## P0 — Sanity gate
- TBD: train cohort 약 10% 감소 (label_time cutoff vs random user). 학습
  가능 row 수 sufficient 확인.

## P1 — Code-path success
- TBD.

## P2 — Primary lift (Platform AUC)
- Anchor: H010 corrected (0.837806).
- Predicted classification: strong | measurable | noise | degraded.

## P3 — NS xattn entropy
- TBD (변화 미세 expected).

## P4 — §18 PASS
- TBD.

## P5 — val ↔ platform
- TBD (val 새 정의 인지).

## P6 — OOF-Platform gap (다른 의미)
- prior H 들과 비교 invalid (OOF 정의 다름).
- 새 정의 하의 단일 측정값.

## P7 — train cohort 변화
- TBD (10% 감소 영향).

## Findings TBD

## Carry-forward to H018
- strong / measurable: anchor = H016 검토. H018 = OOF + recency combo.
- noise (+ H015 / H017 도 noise): **paradigm shift mandatory (Frame B)**.
  H018 = backbone replacement.
- degraded: oof_user_ratio 5% sub-H.
- (Triple-H 종합 결과로 결정)
