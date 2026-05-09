# H033 — Literature References

## Primary
- **Chang, J. et al. 2024** — TWIN (Tencent RecSys 2024). H019 base.
- **H019** — TWIN paradigm shift, cloud measurable PASS 0.839674.
- **H020** — learnable GSU (paper-faithful form). H033 의 component 1.
- **H021** — per-domain top_k (paper-uncovered extension). H033 의 component 2.

## Stacking literature
- **H010 (NS→S xattn) PASS additive** — anchor 0.837806. H008 anchor 위 stacking 성공 사례 (interaction + sequence axis).
- **H009 (combined xattn + DCN-V2) REFUTED — interference** — combined < strongest single. block-level fusion 위치 충돌. H033 의 Frame A reverse 사례.

## Carry-forward refs (from H019/H020/H021)
- ETA (Chen 2021), HSTU (Zhai 2024), SASRec (Kang 2018), DIN (Zhou 2018).
- OneTrans (Tencent WWW 2026) — 별도 mechanism class.

## H033 의 직교성 reference
- H020 / H021 / H033 / H034 = retrieval class 안 4 axis sub-H (scoring quality / quantity / stacking / capacity).
