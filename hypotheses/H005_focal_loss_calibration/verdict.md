# H005 — Verdict (PENDING)

> 실행 후 작성. 현재 status = `scaffold` (가설 문서 6개 + 패키지 빌드 완료, launch 대기).

## Status
`scaffold` — 가설 문서 6개 완비, 코드 패키지 (`experiments/H005_focal_loss_calibration/upload/`) 빌드 완료.

## P1 — Code-path success
- Measured: TBD
- Predicted: NaN 0건, finite val_AUC.
- Verdict per P1: TBD.

## P2 — Primary lift (§17.3 binary)
- Measured: TBD (val_AUC).
- Control (E_baseline_organizer): 0.8251.
- Δ vs control: TBD.
- Predicted: Δ ≥ +0.5 pt ⇒ val_AUC ≥ 0.8301.
- Verdict per P2: TBD.

## P3 — Logloss 검증 (보조)
- Measured: TBD (best_val_logloss).
- Control (E_baseline_organizer): 0.2538.
- Predicted: ≤ 0.2792 (악화 ≤ 10%).
- Verdict per P3: TBD.

## P4 — Submission round-trip
- Measured: TBD (G1–G6 PASS count).
- Predicted: 5/5.
- Verdict per P4: TBD.

## P5 — AUC vs Logloss trade-off (보너스)
- Measured: TBD.
- Verdict per P5: TBD.

## Findings (F-N carry-forward)
TBD.

## Surprises
TBD.

## Update to CLAUDE.md?
TBD — H005 결과로 §10 신규 carry-forward rule 추가 여부 결정.
