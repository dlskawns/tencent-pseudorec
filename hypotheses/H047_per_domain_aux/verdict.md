# H047 — Verdict (PENDING — BUILT 2026-05-08)

## Status
`pending` — H047 upload 패키지 BUILT 2026-05-08. Cloud submit 대기.

**Build approach** (multi-task per-domain aux):
- model.py 에 4 per-domain heads (nn.Linear(d_model, 1)) 추가.
- forward: per-domain seq masked mean-pool → aux_logit per domain.
- forward returns (logits, per_domain_aux_list) tuple.
- trainer: aux_loss = mean(BCE per domain) × 0.25.
- 260 trainable params.
- dataset.py / infer.py / utils.py / local_validate.py / make_schema.py byte-identical to H019.

**Cloud submission**: ready (control=H019). `bash run.sh --seed 42`

## T0 sanity (local) — ALL PASS
1. ✅ AST parse PASS (model.py + train.py + trainer.py)
2. ✅ shellcheck PASS (run.sh)
3. ✅ md5 verify: 5 unchanged files identical to H019

## Source data / Decisions
- TBD (post-cloud).
