# H001 — Literature References

## Primary references
- [papers/unified_backbones/pcvrhyformer_baseline.md](../../papers/unified_backbones/pcvrhyformer_baseline.md) — Organizer baseline, structure of `MultiSeqHyFormerBlock` + `RankMixerNSTokenizer` + dual optimizer pattern.

## Secondary references
- [papers/unified_backbones/onetrans_tencent.md](../../papers/unified_backbones/onetrans_tencent.md) — OneTrans (arXiv:2510.26104) NS-token equal-split chunking 가 PCVRHyFormer의 RankMixer NS tokenizer와 동일 메커니즘 (parameter-free chunking). Baseline 의 NS path 가 ablation 대상 component 가 됨을 시사 (다음 H 후보).
- [papers/unified_backbones/interformer_meta.md](../../papers/unified_backbones/interformer_meta.md) — InterFormer (arXiv:2411.09852) bridge gating σ(−2) initialization. 본 H는 새 bridge 추가 안 함이라 미적용. 다음 H (DAMTB) 의 1순위 transfer source.

## Counter-evidence references
- 없음 (anchor 가설은 기준점이지 주장이 없으므로 반대 evidence가 의미 없음).
