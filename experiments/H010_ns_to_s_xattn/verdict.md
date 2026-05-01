# E_H010 — Experiment Verdict (PASS — additive)

> Mirror of `hypotheses/H010_ns_to_s_xattn/verdict.md`. 본 파일은 experiments/
> 트리에서 E### 단위 reference 용. 자세한 P1–P6 + Findings + Decision tree =
> hypothesis verdict.

## Status
`done` — **PASS (additive vs H008)**.

## Headline numbers
- **Platform AUC: 0.840771** (eval auc, 새 champion).
- OOF AUC: 0.8596, OOF LogLoss: 0.2323.
- attn_entropy_per_layer: [0.8127, 0.8133], threshold 5.6531, violation=False.
- Δ vs anchor (~0.83X): +0.7~+1.1pt.
- Δ vs H008 (0.8387): **+0.0021pt** (additive 분류).
- Δ vs H009 (0.8364): +0.0044pt.
- Δ vs H007 (0.8352): +0.0056pt.
- Wall: 3:44:54 학습 + 297.02초 inference.

## Falsification check (predictions.md P1–P6)

| P | Predicted | Measured | Verdict |
|---|---|---|---|
| P1 | NaN-free 완주 | Training complete, metrics dumped | **PASS** |
| P2 | Δ vs anchor ≥ +0.5pt | +0.7~+1.1pt | **PASS** |
| P2-sub | additive vs H008 [+0.001, +0.005pt] | +0.0021pt | **additive** |
| P3 | attn entropy < 5.65 | [0.8127, 0.8133] (sparse) | **PASS** |
| P4 | §18 인프라 PASS | eval auc 0.840771 (≠ 0.5) | **PASS** |
| P5 | val ↔ platform |val−plat| ≤ 0.05 | TBD (val_AUC paste 후) | TBD |
| P6 | nontrivial NS routing | entropy 0.81 = highly selective | **indirect PASS** |

## Decision applied
predictions.md table → "Δ vs anchor ≥ +0.5pt + additive vs H008" 분기 →
**H010 PASS. anchor 갱신 (H010 = 새 baseline). H011 = 다른 axis 탐험.**

## Next actions
1. `experiments/INDEX.md` H010 row 갱신 (val/OOF/platform/wall/status).
2. `hypotheses/INDEX.md` Active Pipeline → Archive, Active Phase + Recent
   Findings prepend.
3. `progress.txt` iter block append.
4. anchor 갱신: H011+ control = H010 (Platform 0.8408). H008 carry-forward.
5. H011 후보 = orthogonal axis (aligned `<id, weight>` pair encoding 권장).

## Pending paste (사용자 보강 시 추가)
- raw `metrics.json` blob (best_step, val_AUC, config_sha256, git_sha 확인).
- 학습 log tail 200줄 (peak epoch, batch heartbeat).
- inference log (`[infer] OK: torch path produced N predictions`).
