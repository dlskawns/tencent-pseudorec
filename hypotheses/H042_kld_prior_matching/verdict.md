# H042 — Verdict (PENDING — BUILT 2026-05-07)

## Status
`pending` — H042 upload 패키지 BUILT 2026-05-07. Cloud submit 대기.

**Build approach** (H038 base + KLD term):
- trainer.py 'aux_timestamp' branch 에 KLD prior matching 추가 (~10 lines, behind `if self.kld_lambda > 0`).
- per-sample Bernoulli KL(σ(logit) ‖ Bern(0.124)) → mean → λ_kld weighted.
- λ_kld=0.01 (보수적, KLD/BCE 비율 ~1%).
- prior=0.124 (§3.4 class prior).
- kld_lambda=0 시 H038 byte-identical (safe carrier).

**Cloud submission**: ready (control=H038). `bash run.sh --seed 42`

## T0 sanity (local) — ALL PASS
1. ✅ AST parse PASS (trainer.py + train.py)
2. ✅ random logits → KLD=0.6763 (positive, math correct)
3. ✅ logits at prior logit → KLD=−1.31e-08 ≈ 0 (math correct)
4. ✅ kld_lambda=0 → byte-identical to H038 (safe carrier)
5. ✅ gradient flow: |grad|.sum=0.2528 (KLD trainable)
6. ✅ md5 verify: 6 unchanged files (model.py / dataset.py / infer.py / utils.py / local_validate.py / make_schema.py) identical to H038

## Source data
- TBD (post-cloud).

## P1 — Code-path success
- TBD.

## P2 — Primary lift (§17.3 binary)
- TBD. control=H038 (Val 0.83735 / OOF 0.8623). Cut: ≥ +0.003pt OOF strong / ≥ +0.001pt measurable / < +0.001pt noise.
- 추가: Δ vs H019 (Val 0.83720 / OOF 0.8611) — output-distribution axis 누적 effect.

## P3 — KLD mechanism 작동 검증
- TBD. predicted prob 의 batch mean 이 epoch 진행 따라 prior=0.124 쪽으로 이동하는지 측정 (logging hook 추가 candidate).

## P4 — §18 인프라 통과
- TBD.

## Carry-forward to 다음 H
- TBD per Decision tree (predictions.md).

## Decision applied (per predictions.md decision tree)
- TBD (post-cloud).
