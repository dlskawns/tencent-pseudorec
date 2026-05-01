# H008 — Literature References

## Primary

- **DCN-V2** (Improved Deep & Cross Network) — Wang, Shivanna, Lin, He, Singh, Mehrotra, Cheng. WWW 2021. arXiv:2008.13535.
  - Production CTR 표준 lever. Low-rank polynomial cross with x₀ residual.
  - Mechanism: `xₗ₊₁ = x₀ ⊙ (UV ᵀ xₗ + b) + xₗ`.

## Mechanism class family (sparse_feature_cross)

- **DCN** (Deep & Cross Network) — Wang et al. ADKDD 2017. arXiv:1708.05123. DCN-V2 의 origin.
- **FwFM** (Field-weighted Factorization Machine) — Pan et al. WWW 2018. arXiv:1806.03514.
- **FmFM** (Field-matrixed FM) — Sun et al. WSDM 2021.
- **AutoDis** (Automatic Discretization for Embeddings) — Liu et al. CIKM 2021.
- **CAN** (Co-Action Network) — Bian et al. KDD 2022. candidate-aware variant.

## Implementation source

- 신규 클래스 `DCNV2CrossBlock` (model.py 확장).
- `MultiSeqHyFormerBlock` step 3 에서 `RankMixerBlock` 또는 `DCNV2CrossBlock` 으로 dispatch.

## Comparison anchor (control)

- **original_baseline** (anchor 2026-04-28). PCVRHyFormer + transformer encoder + **RankMixer fusion** + BCE + smoke envelope.
- 본 H 가 paired 비교 대상. 같은 split, 같은 envelope, **`fusion_type` flag 만 변수**.

## Carry-forward from prior H

- **H006 verdict F-3**: paired Δ는 platform AUC 으로만.
- **H007 verdict F-1**: target_attention mechanism PASS marginal. orthogonal axis (interaction = 본 H) 검증 차례.
- **H007 verdict F-2**: val ↔ platform 정합 expected.
- **H007 verdict F-4**: extended envelope cost — 본 H smoke 우선.
- **H007 verdict F-5**: §18 인프라 룰 inherit.

## Carry-forward rules referenced

- **§10.5 LayerNorm on x₀ MANDATORY** — `DCNV2CrossBlock` 의 첫 step = Pre-LN on x₀. 직접 적용 영역.
- **§10.7 카테고리 rotation** (sparse_feature_cross 첫 충족).
- **§17.2 one-mutation** (fusion 클래스 swap).
- **§17.3 binary success** (Δ ≥ +0.5pt).
- **§17.4 카테고리 rotation 추가 충족**.
- **§17.5 sample-scale**.
- **§17.6 cost cap** (smoke 우선).
- **§17.7 falsification-first**.
- **§17.8 cloud handoff discipline**.
- **§18 inference infrastructure**.

## External inspirations (§10.4 P1+ 의무 주입)

- 본 H 미충족 (sparse_feature_cross 카테고리). H009+ carry-forward (Switch Transformer load balance loss 등).

## Future ablation candidates (post-H008)

- **DCN-V2 layer 수 tuning**: 2 → 4 → 6.
- **rank tuning**: 8 → 16 → 32.
- **Per-domain DCN-V2 cross** (current global → domain-specific).
- **DCN-V2 + RankMixer parallel arm**: 둘 다 적용 후 gate-combine. InterFormer-style.
- **다른 explicit cross**: FwFM, FmFM, AutoDis. sparse_feature_cross 카테고리 변형.
- **H009 = combined H007 + H008**: candidate xattn + DCN-V2 cross stack. additive 가정 검증.
