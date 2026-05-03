# H015 — Verdict (PENDING)

> 실행 후 작성. 현재 status = `code_build_pending`.

## Status
`code_build_pending` — 가설 문서 6 files 완비. 코드 패키지 빌드 완료
(dataset.py + trainer.py + train.py + run.sh + README 변경, model.py /
infer.py / utils.py byte-identical with H010). tar.gz + ast.parse sanity.

## P1 — Code-path success
- Measured: TBD.
- Predicted: PASS (mean weight = 1.0 보존).

## P2 — Primary lift (§17.3 binary, sample-scale relaxed)
- Measured: TBD.
- Anchor: H010 corrected (Platform 0.837806).
- Δ vs anchor: TBD.
- Predicted classification: strong | measurable | noise | degraded.

## P3 — NS xattn entropy 변화
- Measured: TBD (H010 baseline [0.8127, 0.8133]).
- Predicted: 변화 미세 (mechanism stack byte-identical, loss 가중치만 영향).

## P4 — §18 인프라 통과
- Measured: TBD.
- Predicted: PASS (infer.py 변경 0).

## P5 — val ↔ platform 정합
- Measured: TBD.
- Predicted: ≤ 0.05.

## P6 — OOF-platform gap (핵심 진단)
- Measured: TBD.
- 9 H baseline: 1.88~2.59pt.
- Predicted classification: gap 줄어듦 | 유지 | 더 벌어짐.

## P7 — OOF AUC 변화
- Measured: TBD.
- 9 H baseline: 0.857~0.860.
- Predicted classification: 약간 하락 | 유지 | 향상.

## Findings (F-N carry-forward)
TBD.

## Surprises
TBD.

## Update to CLAUDE.md?
TBD — H015 결과에 따라:
- L2 confirmed → §17 에 "cohort drift mitigation 필수 룰" 추가 후보.
- L2 retire → §17 에 "paradigm shift trigger condition" 추가 후보.

## Carry-forward to H016
TBD — predictions.md decision tree:
- strong / measurable → H016 = recency variants (exp decay, larger range, per-dataset).
- noise → **paradigm shift mandatory** → H016 = backbone replacement.
- degraded → H015-sub = weight range 좁힘 또는 exp decay.
- P6 gap 더 벌어짐 → recency direction 잘못 → H016 = OOF 재정의 (Frame C).
