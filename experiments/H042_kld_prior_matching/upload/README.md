# H042_kld_prior_matching — Technical Report

> CLAUDE.md §0.5 data-signal-driven: H038 (BCE+aux MSE on log1p(timestamp))
> 결과 OOF Δ vs H019 = +0.00120pt, Val Δ +0.00015pt (LogLoss −0.0013).
> 14 H 모두 sample-level loss (BCE/BPR/focal) 만 — H038 가 처음으로 *output
> distribution-level supervision* axis 작동 confirm. H042 = H038 base 위에
> KLD prior matching 추가. 사용자 §0.5 의 shift-scheduling KLD pattern 의
> binary 적용.

## 1. Hypothesis & Claim
- Hypothesis: H038 의 aux MSE (간접 calibration signal) 보다 KLD prior
  matching (직접 prob distribution attack) 가 더 강한 lift. **per-sample
  Bernoulli KL(σ(logit) ‖ Bernoulli(0.124))** 가 overconfident prediction
  을 prior 쪽으로 regularize → label smoothing-like generalization.
- Falsifiable: Δ vs H038 (Val 0.83735 / OOF 0.8623) ≥ +0.001pt → KLD direct
  attack 이 MSE 보다 강함, output-distribution axis 진짜 lever.
- Compute tier: T2.4 (~3.5h, ~$5-7).

## 2. Mutation — KLD prior matching term

```python
# H038 base (BCE main + MSE aux on log1p(timestamp)) 위에 추가:
if self.kld_lambda > 0:
    prob = torch.sigmoid(logit_main).clamp(min=1e-6, max=1.0 - 1e-6)
    prior = float(self.kld_target_prior)              # 0.124 (§3.4)
    kld_per_sample = (
        prob * (torch.log(prob) - math.log(prior))
        + (1.0 - prob) * (torch.log(1.0 - prob) - math.log(1.0 - prior))
    )
    loss = loss + self.kld_lambda * kld_per_sample.mean()
```

KL(Bern(σ(logit_i)) ‖ Bern(0.124)) per sample → mean → λ_kld = 0.01.

**왜 per-sample**: batch-mean 만 KLD = 단순 global bias shift = AUC rank-invariant.
per-sample 이 individual prediction 을 regularize → label smoothing 변형, AUC affect.

**왜 prior=0.124**: §3.4 class prior (label_type=2: 124/1000).

**왜 λ=0.01**: T0 sanity 에서 random init KLD ≈ 0.68, BCE ≈ 0.7. λ=0.01 →
KLD 기여 ~1% of BCE. 너무 크면 prediction collapse to prior.

## 3. Decision tree (post-result)

| Δ vs H038 (OOF 0.8623) | Action |
|---|---|
| ≥ +0.003pt | super-additive → output-distribution axis main lever, sub-H = λ sweep + per-domain prior |
| [+0.001, +0.003pt] | additive → axis 작동, sub-H 가치 |
| (-0.001, +0.001pt] | KLD MSE 와 redundant 또는 λ 너무 작음 |
| < -0.001pt | KLD BCE 와 fight, λ↓ 또는 retire |

추가 비교: Δ vs H019 (Val 0.83720 / OOF 0.8611) — H038+KLD axis 누적 vs single-stage.

## 4. Files
| File | H038 대비 | Purpose |
|---|---|---|
| `trainer.py` | + math import, + 2 __init__ params, + KLD term in 'aux_timestamp' branch (~10 lines) | Training loop |
| `train.py` | + 2 argparse + 2 plumbing line | CLI |
| `run.sh` | + `--kld_lambda 0.01 --kld_target_prior 0.124` | Entry |
| `README.md` | new | Doc |
| `model.py / dataset.py / infer.py / utils.py / local_validate.py / make_schema.py` | byte-identical (md5 verified) | unchanged |

trainable params 추가: **0** (loss term 만, no new module).

## 5. T0 sanity (local) — ALL PASS
1. ✅ AST parse PASS for trainer.py + train.py
2. ✅ random logits → KLD=0.6763 (positive, math correct)
3. ✅ logits at prior logit → KLD=−1.31e-08 ≈ 0 (math correct)
4. ✅ kld_lambda=0 → byte-identical to H038 (safe carrier)
5. ✅ gradient flow: |grad|.sum=0.2528 (KLD trainable)
6. ✅ md5 verify: 6 unchanged files identical to H038

## 6. Carry-forward
- §17.2 single mutation: KLD term 추가, model graph byte-identical to H038, infer.py byte-identical to H038.
- §17.4 rotation: NEW first-touch (output_distribution_supervision axis sub-H), AUTO_JUSTIFIED.
- §10.6 sample budget: trainable params +0.
- §0.5 data-signal-driven: §3.4 class prior 12.4% 직접 attack, paper transplant 0.
- §18.7 + §18.8: H038/H019 carry.
