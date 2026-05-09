# H049 — Verdict (PENDING — BUILT 2026-05-08)

## Status
`pending` — H049 upload 패키지 BUILT 2026-05-08. Cloud submit 대기.

**Build approach**:
- model.py: ns_type_emb (Embedding(4, D)) + register_buffer ns_type_ids +
  _build_token_streams 에서 broadcast ADD.
- run.sh: --item_ns_tokens 2→6, --use_ns_slot_type_emb.
- ~16K trainable params 추가.
- trainer.py / dataset.py / utils.py byte-identical to H019.
- **infer.py flag parity** verified.

**Cloud submission**: ready (control=H019). `bash run.sh --seed 42`

## T0 sanity (local) — TBD post AST verify

## Source data / Decisions
- TBD (post-cloud).
