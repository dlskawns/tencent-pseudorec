# E_H012 — Experiment Verdict (REFUTED — Frame B confirmed, uniform routing)

> Mirror of `hypotheses/H012_multi_domain_fusion/verdict.md`. 자세한 P1–P6
> + Findings + Decision tree = hypothesis verdict.

## Status
`done` — **REFUTED — Frame B (uniform routing)**.

## Headline numbers
- **Platform AUC: 0.838047** (eval auc) — anchor H010 0.8408 대비 −0.0028pt.
- OOF AUC: 0.8589, OOF LogLoss: 0.2335.
- attn_entropy_per_layer: [0.8130, 0.8139] (H010 baseline 변화 없음).
- **moe_gate_entropy_per_block: [1.378, 1.363]** (uniform threshold 1.386
  거의 동일 → uniform routing).
- collapse_violation: False (collapse 위험 없음, 단 specialization 없음).
- Δ vs H010 anchor (0.8408): **−0.0028pt**.
- Δ vs H008 carry-forward (0.8387): −0.0007pt.
- Δ vs H011 (0.8347): +0.0033pt.
- Wall: 2:49:43 학습 (H010 −24%) + 137.39초 inference (H010 −54%).
- OOF-platform gap: 2.10pt.

## Falsification check (predictions.md P1–P6)

| P | Predicted | Measured | Verdict |
|---|---|---|---|
| P1 (code-path) | NaN-free 완주 | Training complete | **PASS** |
| P2 (primary lift) | Δ vs anchor ≥ +0.001pt | −0.0028pt | **REFUTED** |
| P2 strong | Δ ≥ +0.005pt | 미달 | — |
| P3 (gate entropy) | specialized [0.69, 1.30] | [1.378, 1.363] | **uniform (Frame B)** |
| P3 collapse | < 0.69 | 미발생 | safe |
| P4 (§18 인프라) | PASS | eval auc 0.838 ≠ 0.5 | **PASS** |
| P5 (val↔platform) | gap ≤ 0.05 | TBD | TBD |
| P6 (OOF-platform gap) | ≤ 2pt | 2.10pt | **경계 미달** |

## Decision applied
predictions.md table → "noise + uniform" 경계 + slight degradation → **Frame
B 채택**. anchor 갱신 안 함. expert mechanism 효과 입증 실패.

card.yaml decision_tree → noise 분기 + F-2 (hyperparameter measurement bias
노출) → **H013 = hyperparameter calibration 우선**.

## Critical context (measurement regime)
- 사용자 batch_size=2048 (default 256 의 8×). lr default 1e-4 추정 → linear
  scaling rule 미적용 → effective lr 1/8 underpowered.
- H006~H012 모두 같은 regime — 상대 paired Δ valid, 절대 lift 작은 게
  mechanism 한계인지 hyperparameter artifact 인지 미확정.
- H013 = calibration 으로 결정.

## Next actions
1. `experiments/INDEX.md` H012 row 갱신.
2. `hypotheses/INDEX.md` Active Pipeline → Archive, Active Phase + Recent
   Findings prepend, H013 후보 = hyperparameter calibration.
3. `progress.txt` iter block append.
4. H013 scaffold = anchor recalibration (batch/lr) — measurement H.

## Pending paste (사용자 보강 시 추가)
- raw `metrics.json` blob (best_step, val_AUC, config_sha256, git_sha 확인,
  user-provided batch/lr 기록).
- 학습 log tail 200줄 (peak epoch, MultiDomainMoEBlock init 로그).
- inference log (`[infer] OK`).
