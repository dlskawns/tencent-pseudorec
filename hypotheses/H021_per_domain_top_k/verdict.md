# H021 — Verdict (PENDING — SCAFFOLDED 2026-05-06)

## Status (SCAFFOLDED 2026-05-06)
`pending` — H021 hypothesis docs (6 files) + experiment card SCAFFOLDED. upload package NOT YET BUILT.

**Build approach** (TWIN sub-H, per-domain top_k on H019 base):
- TWINBlock class **byte-identical to H019** (이미 top_k 를 init param 으로 받음).
- PCVRHyFormer 의 `twin_top_k` argument type 변경 (int → int | dict). 4 TWINBlock instantiation 시 도메인별 K 적용.
- per-domain K = `{a:64, b:64, c:64, d:96}` — domain d 만 50% 확장 (uniform K=128 sweep flat 영역의 75% conservative end).
- ESU / GSU / aggregator / gate=-2.0 / num_heads=4 byte-identical to H019.
- `--twin_top_k_per_domain "64,64,64,96"` argparse flag (default None, H019 호환 유지).
- params 추가: **0** (top_k 는 hyperparam).

**Single mutation compliance** (§17.2): top_k policy 의 uniform → domain-aware. TWINBlock 내부 변경 0. PCVRHyFormer wiring + train.py argparse + run.sh 만 변경.

**Defensive considerations**:
- Backward compat: `--twin_top_k_per_domain` 없으면 H019 동일 동작.
- Argparse string 파싱: 4-int 검증 (assert len == 4).
- T0 sanity 에서 도메인별 TWINBlock 의 `self.top_k` 값 print 확인.

**T0 sanity (local) — TBD**:
1. PCVRHyFormer instantiation log: `H021 TWIN per-domain K: a=64 b=64 c=64 d=96`
2. 4 TWINBlock 각각의 `self.top_k` 값 (64/64/64/96) 확인
3. domain d forward (history_len=128, K=96) shape (B, D), NaN-free
4. Ablation H019 (uniform K=64) vs H021 (d=96) forward output max abs diff > 0.001 (domain d 만)
5. Full PCVRHyFormer params (~161M, H019 와 동일 — parameter-free 변경)

## Source data
- TBD (post-cloud).

## P1 — Code-path success
- TBD.

## P2 — Primary lift (§17.3 binary)
- TBD. Δ vs H019 (cloud actual 0.839674). Cut: ≥ +0.003pt strong / [+0.001, +0.003pt] measurable / (−0.001, +0.001pt] noise / < −0.001pt degraded.
- Secondary: Δ vs H010 corrected (0.837806) = +0.00187 + H021 Δ.
- Triple-paired: H019 vs H020 vs H021 비교.

## P3 — Per-domain K mechanism 작동 검증
- TBD. ESU attention entropy per-domain (domain d 의 entropy 변화), GSU score distribution per-domain (top 64-96 slot score), top-K filter activity per-domain.

## P4 — §18 인프라 통과
- TBD. dataset.py / infer.py / make_schema.py / TWINBlock class 변경 없음 → auditor 범위 좁음 (PCVRHyFormer wiring + train.py + run.sh).

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
- TBD per Decision tree (predictions.md). H020 결과와 paired 비교 framework.

## Decision applied (per predictions.md decision tree)
- TBD.
