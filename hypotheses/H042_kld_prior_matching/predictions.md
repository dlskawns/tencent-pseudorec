# H042 — predictions.md

## Outcome distribution (pre-cloud, prior)
- Strong (≥+0.003pt OOF vs H038): 15% — KLD direct attack 이 MSE 보다 명확히 강함.
- Measurable (+0.001~+0.003pt): 35% — additive 약 effect.
- Noise (-0.001~+0.001pt]: 35% — KLD MSE 와 redundant 또는 λ 너무 작음.
- Degraded (<-0.001pt): 15% — KLD 가 BCE 와 fight, prediction collapse 시작.

## Decision tree (post-result)

| Δ vs H038 OOF (0.8623) | Action |
|---|---|
| ≥ +0.003pt | output-distribution axis main lever 강한 confirm. sub-H = λ_kld sweep (0.005/0.02/0.05) + per-domain prior 또는 per-cohort prior. |
| [+0.001, +0.003pt] | additive 작동, sub-H 가치. |
| (-0.001, +0.001pt] | KLD MSE redundant — sub-H 우선순위 낮음, 다음 axis pivot (β DANN 또는 γ item-side). |
| < -0.001pt | λ_kld 0.01 너무 큼. λ=0.005 retry 또는 retire. |

추가: Δ vs H019 (0.8611) 도 동시 측정. H038+H042 axis 누적 vs single-stage H019:
- H042 OOF − H019 OOF ≥ +0.003pt → output-distribution axis 누적 lever 강함.
- H042 OOF − H019 OOF ≥ +0.001pt → 누적 measurable.

## Risk
- KLD term 이 BCE 와 fight 가능 — BCE 는 sharp prediction 권장, KLD prior
  matching 은 prior 쪽 (uncertainty) 권장. λ tuning 잘못시 학습 destabilize.
- T0 sanity 에서 KLD/BCE 비율 0.96 → λ=0.01 면 KLD 기여 ~1%, 안전 범위.
- AUC rank-invariance: 단순 global bias shift 면 AUC lift 0. per-sample form
  이 individual logit regularize → AUC affect 가능 (label smoothing 처럼).
- §0.5 step 2 (diagnose root cause) — §3.4 의 12.4% prior 가 실제로 prediction
  bias 의 원인인지 직접 검증 안 됨. 만약 model 이 이미 internal calibration
  잘 하고 있으면 KLD 무용 (H041 의 R2 reframe 결함 패턴 가능).
