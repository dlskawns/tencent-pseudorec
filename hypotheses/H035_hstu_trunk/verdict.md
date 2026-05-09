# H035 — Verdict (PENDING — BUILT 2026-05-06)

## Status (BUILT 2026-05-06)
`pending` — H035 hypothesis docs (6 files) + experiment card + upload package COMPLETE. cloud submit ready (`bash run.sh --seed 42`). 단 H020/H021 결과 회수 후 paradigm shift 우선순위 재평가 권장 (H035 noise 시 H036_cohort_embed pivot).

**Build approach** (paradigm shift sub-H — backbone replacement):
- HSTUEncoder class 신규 (~50 lines, paper-faithful core: silu-attention + U gate).
- create_sequence_encoder dispatch 에 `'hstu'` 분기 추가.
- train.py argparse choices 에 `'hstu'` 추가.
- run.sh: `--seq_encoder_type hstu` flag bake.
- 다른 모든 부분 H019 byte-identical (TWIN GSU+ESU, NS xattn, DCN-V2, gate=-2.0, top_k=64, seq 256, batch 1024).
- params 변화: HSTU 가 transformer 보다 작음 (per layer 21K vs 54K, 0.38×). **−264K 절약** (4 도메인 × 2 block).

**Single mutation compliance** (§17.2): per-domain seq encoder type 의 단일 변경. NS xattn / DCN-V2 / TWIN / aggregator / gate / num_heads 전부 H019 byte-identical.

**T0 sanity (local) — PASS**:
1. HSTUEncoder forward shape (4, 64, 64), NaN-free, Inf-free
2. mask passthrough 정상
3. params per layer = 20,736 (TransformerEncoder 54,144 의 0.38×)
4. HSTU vs Transformer ablation diff 1.267 (mechanism active, 매우 다른 representation)
5. create_sequence_encoder('hstu') dispatch 정상

## Source data
- TBD (post-cloud).

## P1 — Code-path success
- TBD.

## P2 — Primary lift (§17.3 binary)
- TBD. Δ vs H019 (cloud actual 0.839674). Cut: ≥ +0.005pt strong / [+0.001, +0.005pt] measurable / (-0.001, +0.001pt] noise / < -0.001pt degraded.
- Secondary: Δ vs H010 corrected (0.837806) = +0.001868 + H035 Δ.

## P3 — HSTU mechanism 작동 검증
- TBD. silu(score) 분포, U gate magnitude, per-layer entropy (정보용, abort 적용 안 함).

## P4 — §18 인프라 통과
- TBD. dataset.py / infer.py / make_schema.py byte-identical. model.py state dict key 변경 (transformer → hstu) → infer.py 가 cfg 기반 재구성 (H019 carry 패턴).

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
- TBD (HSTU 가 PASS 시 §0 backbone 표 갱신 + §10.9 entropy abort threshold 의 silu-attention 적용 룰 추가 가능).

## Carry-forward to H### (다음 H)
- TBD per Decision tree (predictions.md).
- noise → H036_cohort_embed 강제 attack.
- strong → sub-H = HSTU full form 또는 stack.

## Decision applied
- TBD.
