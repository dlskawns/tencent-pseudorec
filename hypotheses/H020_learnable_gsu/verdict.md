# H020 — Verdict (PENDING — SCAFFOLDED 2026-05-06)

## Status (SCAFFOLDED 2026-05-06)
`pending` — H020 hypothesis docs (6 files) + experiment card SCAFFOLDED. upload package NOT YET BUILT.

**Build approach** (TWIN sub-H, learnable GSU on H019 base):
- TWINBlock GSU 의 parameter-free inner product → `nn.Linear(d_model, d_model//4, bias=False)` projection 1쌍 추가 (W_q on candidate, W_k on history)
- score = (W_q · candidate) · (W_k · history)
- ESU / top_k=64 / aggregator / gate=-2.0 / num_heads=4 전부 H019 byte-identical
- `--twin_learnable_gsu` argparse flag (default False, H019 호환 유지)
- params 추가: 4 도메인 × (W_q + W_k) × (64×16) = 8,192 params

**Single mutation compliance** (§17.2): GSU scoring function 의 단일 변경. 다른 모든 부분 byte-identical to H019.

**Defensive considerations**:
- W_q / W_k init Xavier — projection 직후 score variance 적정.
- T0 sanity 에서 GSU score mean/std 측정 (H019 와 비교).
- top_k filter NaN guard H019 carry-forward.

**T0 sanity (local) — TBD**:
1. TWINBlock direct forward (learnable GSU) shape (B, D), NaN-free
2. W_q / W_k grad flow
3. Ablation H019 vs H020 forward output max abs diff > 0.001
4. GSU score distribution (mean / std) H019 vs H020 비교
5. Full PCVRHyFormer params (~161M + 8K vs H019 161M + 0)

## Source data
- TBD (post-cloud).

## P1 — Code-path success
- TBD.

## P2 — Primary lift (§17.3 binary)
- TBD. Δ vs H019 (cloud actual 회수 후). Cut: ≥ +0.003pt strong / [+0.001, +0.003pt] measurable / (−0.001, +0.001pt] noise / < −0.001pt degraded.
- Secondary: Δ vs H010 corrected (0.837806) = H019 cloud + H020 Δ.

## P3 — Learnable GSU mechanism 작동 검증
- TBD. GSU score distribution (H019 vs H020), W_q/W_k weight norm, ESU attention entropy.

## P4 — §18 인프라 통과
- TBD. dataset.py / infer.py / make_schema.py 변경 없음 → auditor 범위 model.py + train.py 만.

## P5 — val ↔ platform gap
- TBD. F-A baseline mean −0.003pt.

## P6 — OOF (redefined) ↔ Platform gap
- TBD. H016 framework −0.004pt baseline.

## P7 — Cost cap audit
- TBD. T2.4 ~3.5h × $5-7. campaign cap $100 친화.

## Findings (F-N carry-forward)
- TBD.

## Surprises
- TBD.

## Update to CLAUDE.md?
- TBD.

## Carry-forward to H### (다음 H)
- TBD per Decision tree (predictions.md).

## Decision applied (per predictions.md decision tree)
- TBD.
