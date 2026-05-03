# H017 — Literature References

H015 lit_refs.md 와 동일 (sibling H, same category). 추가 없음.

## Primary references

H015 와 같음:
- Pan & Yang 2010 ("A Survey on Transfer Learning").
- Gama et al. 2014 ("A Survey on Concept Drift Adaptation").
- Sugiyama et al. 2007 ("Covariate Shift Adaptation by Importance Weighted CV").
- Production CTR engineering (Meta / Google / Tencent).

## H017 specific

- TWIN (Pan et al. RecSys 2024) — lifelong behavior modeling. Production
  recency handling 은 geometric decay 가 표준 — exp form 이 production-aligned.

## Counter-evidence (Frame B)

H015 lit_refs counter 와 동일.

## Audit references

- `hypotheses/H015_recency_loss_weighting/` — sibling H, paired 비교 base.
- `experiments/H015_recency_loss_weighting/upload/trainer.py` — H017 exp
  branch 구현 base.
