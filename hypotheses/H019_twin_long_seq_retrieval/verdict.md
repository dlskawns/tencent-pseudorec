# H019 — Verdict (RESOLVED 2026-05-07)

## Status
`PASS measurable` (platform). Val/OOF backfilled 2026-05-07 from cloud train log
(verify-claim 누락 retroactive 정리). 본 anchor = post-paradigm-shift 첫 실측치,
이후 H020/H021/H033/H034/H035/H041 의 control_exp_id.

## Source data (cloud T2.4, seed=42, 2026-05-05~06)

| Epoch | Train Loss | Val AUC | Val LogLoss |
|---|---|---|---|
| 1 | 0.16282 | 0.83105 | 0.28388 |
| 2 | 0.15893 | 0.83487 | 0.28391 |
| 3 | 0.15782 | 0.83497 | 0.28143 |
| **4** | **0.15733** | **0.83720** ★ | **0.28080** ★ |
| 5 | 0.15660 | 0.83612 | 0.28133 |
| 6 | 0.15669 | 0.83654 | 0.28112 |
| 7 | 0.15665 | 0.83602 | 0.28177 |

→ Best at epoch 4, early-stop after patience=3.

| Metric | Value |
|---|---|
| Best Val AUC | **0.83720** |
| Best Val LogLoss | **0.28080** |
| OOF AUC | **0.8611** |
| OOF LogLoss | **0.2309** |
| Platform AUC (eval) | **0.839674** |

## Paired Δ vs control = H010 corrected (0.837806)

- **Platform Δ = +0.001868pt** → §17.3 measurable band [+0.001, +0.005pt). PASS measurable.
- val→platform gap: 0.83720 → 0.839674 = +0.00247pt platform overestimate.
- OOF→platform gap: 0.8611 → 0.839674 = **−0.02143pt** (OOF over platform).

## P1 — Code-path success
- ✅ T0 sanity ALL PASS (TWIN forward NaN-free, defensive empty-seq guard).
- ✅ Cloud full-data 7 epoch 정상 수렴, early stop ep4.

## P2 — Primary lift (§17.3 binary)
- **PASS measurable** — Δ = +0.001868pt. retrieval_long_seq class 첫 valid measurement.

## P3 — TWIN mechanism 작동 검증
- TWIN gate sigmoid(-2.0)≈0.12 init → 학습 후 활성도 측정 미수집 (gate logging hook 미추가).

## P4 — §18 인프라 통과
- ✅ §18.7 label_time fill_null + zero_copy_only=False
- ✅ §18.8 SUMMARY block emit (carry-forward H023)

## P5 — val ↔ platform gap
- +0.00247pt (val UNDER platform).

## P6 — OOF (legacy) ↔ Platform gap
- −0.02143pt (OOF OVER platform, F-A/F-G 일관 패턴).

## P7 — Cost cap audit
- T2.4 ~3.5h, ~$5-7 (campaign cap $100 안).

## Findings (F-N carry-forward)

- **F-G 부분 부정 (paradigm shift 첫 신호)** — H019 platform 0.839674 가 12 H ceiling band [0.832, 0.836] 의 **0.838~0.840 영역** 확장. mechanism class change (retrieval_long_seq) 에 한해 부분 깨짐.
- **F-A 갱신** — val→platform gap 6 H 누적 mean ≈ −0.0017pt (H019 +0.0025 추가). retrieval class 가 cohort 우호.
- **F-G ceiling Val 측면 여전 — Val 0.8372 도 12 H band 안**. platform 만 깨짐 → val/OOF 측정 신뢰성 의문 재확인.

## Surprises
- **OOF ≫ Platform** (+0.0214pt) — H010 의 +0.0188 패턴 강화. OOF metric 이 platform generalization 의 noisy proxy.

## Update to CLAUDE.md?
- §10 anti-bias rule 추가 후보: "OOF AUC 단독 verdict 금지 — platform AUC 동시 회수 필수".

## Carry-forward to 다음 H

- **H019 anchor 확정** — Val 0.8372 / OOF 0.8611 / Platform 0.839674.
- **H020/H021/H033/H034/H035**: 결과 대기.
- **H041 (cold-start branch)**: REFUTED — Val Δ −0.0022, OOF Δ −0.0010 → R2 reframe wrong frame.
