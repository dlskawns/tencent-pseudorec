# H013 — Literature References

## Primary references

- **Goyal et al. 2017** (arXiv:1706.02677, "Accurate, Large Minibatch SGD:
  Training ImageNet in 1 Hour") — Linear scaling rule canonical
  reference. batch K× → lr K× 동시 scaling. warmup 권장. Facebook AI.
- **Smith et al. 2018** (arXiv:1711.00489, "Don't Decay the Learning Rate,
  Increase the Batch Size") — batch-lr equivalence theory. lr decay 대신
  batch increase 가 같은 dynamics.

## Secondary references

- [papers/unified_backbones/pcvrhyformer_baseline.md](../../papers/unified_backbones/pcvrhyformer_baseline.md)
  — H013 anchor (H010 mechanism + H008 fusion + organizer baseline).
- He et al. 2019 (arXiv:1908.01878, "Accurate, Efficient and Scalable
  Training of Graph Neural Networks") — large-batch GNN training, sparse
  feature scaling.
- McCandlish et al. 2018 (arXiv:1812.06162, "An Empirical Model of Large-Batch
  Training") — gradient noise scale ~ batch / lr.

## Counter-evidence references (Frame B / C 근거)

- **H010 verdict.md F-3** — NS xattn entropy 0.81 = 384 tokens 중 ~2 만
  attended. Mechanism ceiling 가설 — sparse routing 자체가 정보 bottleneck.
- **H011 verdict.md F-5** — cohort drift hard ceiling 가설. OOF/Platform
  분포 다름.
- **H012 verdict.md F-1** — Frame B confirmed (uniform routing). NS-token
  level mechanism class 한계.
- Keskar et al. 2017 (arXiv:1609.04836, "On Large-Batch Training for Deep
  Learning") — large-batch generalization gap. batch 2048 자체가 sub-optimal
  generalization 위험. Frame A 의 risk.

## Audit references (measurement integrity)

- **`hypotheses/H012_multi_domain_fusion/verdict.md`** — F-2 carry-forward
  (hyperparameter measurement bias 노출). H013 의 핵심 trigger.
- **`experiments/INDEX.md`** — H006~H012 누적 cost 24h, batch override
  명시 (이 turn 갱신).
- **CLAUDE.md §17.2** — one-mutation rule + parametric mutation 정당화
  논의 (challengers.md §재진입정당화).
- **CLAUDE.md §17.3** — binary success threshold (Δ ≥ +0.5pt or +0.001pt
  sample-scale relaxed).
