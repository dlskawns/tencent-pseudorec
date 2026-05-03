# H017 — Method Transfer

> H015 sub-form variant. mechanism class identical (recency loss weighting),
> form parameter only changed (linear → exp).

## ① Source

같음 — H015 references (Pan & Yang 2010, Gama et al. 2014, Sugiyama et al. 2007).
exp form 추가 motivation: production CTR 의 standard recency handling 은
geometric decay (TWIN, lifelong behavior modeling) — exp 가 더 production-aligned.

## ② Original mechanism

Linear: `weight = w_min + (w_max - w_min) × pct`. mean = (w_min + w_max) / 2.
Exp: `weight_raw = w_min × (w_max / w_min)^pct` (geometric). 단순 form 에선
mean ≠ 1.0 (e.g., [0.5, 1.5] → mean ≈ 0.91). **Auto-normalize** → `weight =
weight_raw / weight_raw.mean()` → mean = 1.0 (loss scale 보존).

## ③ What we adopt

H015 byte-identical 외:
- trainer.py `recency_weight_form` arg ('linear' | 'exp', default 'linear').
- exp branch: geometric scale + auto-normalize.
- train.py CLI flag.
- run.sh: `--recency_weight_form exp`.

## ④ What we modify (NOT a clone)

- Range 동일 [0.5, 1.5] (form 효과만 isolated).
- per-batch normalization (per-dataset 은 sub-H).

## ⑤ UNI-REC alignment

H015 와 동일.

## ⑥ Sample-scale viability

params 추가 0. mean=1.0 보존 → loss scale 영향 없음.

## ⑦ Carry-forward rules to honor

H015 와 동일.
