# H012 — Literature References

## Primary references

- [papers/multi_domain_fusion/mmoe_ma2018.md](../../papers/multi_domain_fusion/mmoe_ma2018.md)
  — MMoE: Multi-gate Mixture-of-Experts. expert tower + per-task gate.
- [papers/multi_domain_fusion/ple_tang2020.md](../../papers/multi_domain_fusion/ple_tang2020.md)
  — PLE: Progressive Layered Extraction. shared + task-specific expert
  분리, cross-task interference 감소.

## Secondary references

- [papers/unified_backbones/pcvrhyformer_baseline.md](../../papers/unified_backbones/pcvrhyformer_baseline.md)
  — PCVRHyFormer 의 단일 fusion block 한계.
- STAR (Sheng et al. CIKM 2021) — multi-domain CTR star topology.
- MiNet (Ouyang et al. KDD 2020) — multi-domain CTR with domain-specific
  base towers.
- Switch Transformer (Fedus et al. JMLR 2022) — sparse expert routing
  scaled, top-1 routing.
- DCN-V2 ([papers/sparse_feature_cross/](../../papers/sparse_feature_cross/))
  — H008 anchor, post-MoE interaction layer.

## Counter-evidence references (Frame B 근거)

- H010 verdict.md F-3 — NS xattn entropy 0.81 = 384 tokens 중 ~2 attend
  (highly selective routing). NS xattn 이 이미 implicit 도메인 routing
  학습 중 가능성.
- Switch Transformer expert collapse paper section — sample-scale 에서
  expert collapse 위험 정량.
- MoE training instability surveys — gate softmax 학습 불안정성 보고.

## Audit references (motivation 정량 출처)

- **`eda/out/domain_facts.json`** — domain_vocab, domain_jaccard_overlap,
  target_item_in_domain_seq. 본 프로젝트 카피 (출처: tencent-cc/eda/out/deep.json,
  같은 1000-row flat snapshot).
- CLAUDE.md §3.5 — domain seq length p50/p90/max/frac_empty per domain.
- HF README (`TAAC2026/data_sample_1000`) — flat layout, 4 도메인 시퀀스.
