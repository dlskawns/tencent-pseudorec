# H050_train_ratio_60 — Verdict (NOOP, ARCHIVED 2026-05-08)

## Status
`NOOP` — Val Δ vs H019 = −0.00048pt (band 안). Platform 미제출 (eval slot 보호).

## Mutation
- CLI only: `--train_ratio 0.3 → 0.6`. 모든 .py byte-identical to H019.

## Source data (cloud T2.4, seed=42)

| Epoch | Best Val | Notes |
|---|---|---|
| 1 | 0.83057 | — |
| 2 | 0.83446 | — |
| 3 | 0.83415 | — |
| 4 | 0.83593 | — |
| 5 | 0.83655 | — |
| **6** | **0.83672** ★ | best |
| 7 | 0.83584 | — |
| 8 | 0.83625 | — |
| 9 | 0.83468 | early stop |

| Metric | H019 anchor | H050 | Δ |
|---|---|---|---|
| Best Val | 0.83720 | 0.83672 | **−0.00048pt** |
| Best LogLoss | 0.28080 | 0.28069 | −0.00011 |
| Platform | 0.839674 | not submitted | — |

## Verdict
- Val −0.0005 → §17.3 noise band, NOT submission-worthy.
- H019 ratio=1 underperform 패턴 (eval 0.837785) 와 일치 — 0.3 sweet spot 약 confirm.
- Platform 미제출 (slot 보호, theoretical EV 낮음).

## Carry-forward
- F-A 패턴 + ratio scaling: train_ratio=0.3 가 본 데이터 sweet spot likely.
- 더 expand 시도 (0.7+) 가치 낮음.
- Local source code DELETED 2026-05-08 (recycled to H053). Cloud ckpt 보존.
