# H002 — Literature References

## Primary references
- [papers/unified_backbones/interformer_meta.md](../../papers/unified_backbones/interformer_meta.md) — InterFormer (Meta CIKM 2025, arXiv:2411.09852). 본 H 의 메커니즘 출처. R1 (bidirectional bridges), R2 (rank=4 bottleneck), R3 (gate init = -2). H002 의 transfer.md ⑦ 가 R1–R3 직접 인용.

## Secondary references
- [papers/unified_backbones/pcvrhyformer_baseline.md](../../papers/unified_backbones/pcvrhyformer_baseline.md) — Organizer baseline = control. `MultiSeqHyFormerBlock` 의 step 1 (per-domain seq encoding) 직후가 본 H의 bridge 삽입 위치.
- [papers/unified_backbones/onetrans_tencent.md](../../papers/unified_backbones/onetrans_tencent.md) — H003 후보 (OneTrans single-stream mixed causal). 본 H 가 도메인간 정보 흐름 시도하는 첫 step. OneTrans는 동일 목적의 다른 메커니즘.

## Counter-evidence references
- 없음. paper 의 bridge mechanism 에 대한 외부 반증 evidence 미발견. Paper 자체의 ablation 이 bridges OFF 시 성능 저하를 보고 (R1 strict expressive superset claim).
