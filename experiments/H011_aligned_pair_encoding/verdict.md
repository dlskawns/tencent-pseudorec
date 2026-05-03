# E_H011 — Experiment Verdict (REFUTED — degraded vs H010 anchor)

> Mirror of `hypotheses/H011_aligned_pair_encoding/verdict.md`. 자세한 P0–P6
> + Findings + Decision tree = hypothesis verdict.

## Status
`done` — **REFUTED (degraded vs H010 anchor)**.

## Headline numbers
- **Platform AUC: 0.834699** (eval auc) — anchor 대비 악화.
- OOF AUC: 0.8589, OOF LogLoss: 0.2342.
- attn_entropy_per_layer: [0.8130, 0.8132], threshold 5.6531, violation=False.
- Δ vs H010 anchor (0.8408): **−0.0061pt**.
- Δ vs H008 carry-forward (0.8387): **−0.0040pt**.
- Δ vs original_baseline (~0.83X): +0.4~+0.5pt (anchor 정확값 의존).
- Wall: 2:46:54 학습 (H010 −25%) + 178.31초 inference (H010 −40%).
- OOF-platform gap: 2.42pt (H010 1.88 → 다시 벌어짐).

## Falsification check (predictions.md P0–P6)

| P | Predicted | Measured | Verdict |
|---|---|---|---|
| P0 | aligned mapping audit | eda/out/aligned_audit.json PASS (pre-train) | **PASS** |
| P1 | NaN-free 완주 | Training complete, metrics dumped | **PASS** |
| P2 | Δ vs anchor ≥ +0.001pt (binary measurable) | −0.0061pt | **REFUTED (degraded)** |
| P2 strong | Δ ≥ +0.005pt | 미달 | — |
| P3 | NS xattn entropy 변화 (sparse / 미세 / uniform) | [0.8130, 0.8132] (변화 미세) | **변화 미세** (Frame B 신호) |
| P4 | §18 인프라 PASS | eval auc 0.834699 ≠ 0.5 | **PASS** |
| P5 | val ↔ platform |val−plat| ≤ 0.05 | TBD (raw paste 후) | TBD |
| P6 | OOF-platform gap ≤ 2pt | 2.42pt | **미달** (capacity-overfit cohort) |

## Decision applied
predictions.md table → "Δ vs anchor < −0.001pt (degraded)" 분기 →
**REFUTED**. anchor 갱신 안 함. card.yaml `degraded` 분기 적용.

## Next actions
1. `experiments/INDEX.md` H011 row 갱신 (val/OOF/platform/wall/status).
2. `hypotheses/INDEX.md` Active Pipeline → Archive, Active Phase + Recent
   Findings prepend (F-1 ~ F-6).
3. `progress.txt` iter block append.
4. H012 후보 = **multi_domain_fusion (MMoE/PLE)** 신규 카테고리 first-touch.

## Pending paste (사용자 보강 시 추가)
- raw `metrics.json` blob (best_step, val_AUC, config_sha256, git_sha 확인).
- 학습 log tail 200줄 (peak epoch, batch heartbeat, `H011 aligned_pair_encoding ENABLED: 8 fids ...` 로그 검증).
- inference log (`[infer] OK: torch path produced N predictions`).
