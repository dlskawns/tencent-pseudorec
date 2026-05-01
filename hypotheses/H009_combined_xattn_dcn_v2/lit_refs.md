# H009 — Literature References

## Primary

본 H 는 **새 paper source 도입 없음** — H007 + H008 단독 PASS mutation stacking.

## H007 reference (sequence axis component)

- **DIN** (Zhou et al. KDD 2018, arXiv:1706.06978) — origin paper of candidate-as-attention-query.
- **CAN** (Bian et al. KDD 2022) — candidate × history co-action (candidate-aware variant).
- **SIM** (Pi et al. CIKM 2020) — long-history candidate-aware retrieval.
- **TWIN** (Chang et al. KDD 2023) — SIM + ESU joint training.
- **HSTU** (Meta 2024) — hierarchical candidate-aware multi-layer.
- **OneTrans** (Tencent WWW 2026, arXiv:2510.26104) — single-stream candidate token + mixed-causal mask.

H007 implementation: modern multi-head cross-attention (Pre-LN), candidate = (item_ns + item_dense_tok) mean pool, per-domain ModuleDict, prepend to seq.

## H008 reference (interaction axis component)

- **DCN-V2** (Wang et al. WWW 2021, arXiv:2008.13535) — production CTR explicit polynomial cross.
- **DCN** (Wang et al. ADKDD 2017) — DCN-V2 의 origin.
- **FwFM** (Pan et al. WWW 2018), **FmFM** (Sun et al. WSDM 2021) — field-weighted FM 변형.
- **AutoDis** (Liu et al. CIKM 2021) — automatic discretization for embeddings.

H008 implementation: low-rank cross W = U V^T (rank=8), 2 stacked layers (degree 3), Pre-LN on x_0, token-wise application.

## Comparison anchor (control)

- **original_baseline** (anchor 2026-04-28). PCVRHyFormer + transformer encoder + RankMixer fusion + BCE + smoke envelope. Platform AUC ~0.83X.
- 본 H 의 paired 비교 대상.

## Carry-forward from prior H

- **H007 verdict F-1**: target_attention mechanism PASS marginal. 본 H sequence axis component.
- **H008 verdict F-1**: sparse_feature_cross mechanism PASS, +0.0035pt vs H007. 본 H interaction axis component.
- **H008 verdict F-2**: §0 P1 룰 (block-level gradient 공유) active 검증. 본 H 도 같은 패턴.
- **H008 verdict F-3**: additivity 검증 필요 → **본 H 의 핵심 측정**.
- **H008 verdict F-4**: patience=3 + early stop aggressive → 본 H 적용.
- **H008 verdict F-5**: OOF cohort 갭 좁아지는 패턴.

## Carry-forward rules referenced

- §10.5 LayerNorm on x_0 (DCN-V2 자동 충족).
- §10.6 sample budget cap (anchor 면제).
- §10.7 카테고리 rotation (§17.4 stacking 정당화).
- §10.9 OneTrans softmax-attention entropy abort (H007 candidate xattn 도 softmax — instrumentation sub-H).
- §17.2 one-mutation (stacking 정당화).
- §17.3 binary success.
- §17.4 카테고리 rotation 정당화.
- §17.5 sample-scale.
- §17.6 cost cap.
- §17.7 falsification-first.
- §17.8 cloud handoff discipline.
- §18 inference infrastructure.

## External inspirations (§10.4 P1+ 의무 주입)

본 H 미충족 (stacking 만, 새 mechanism 아님). H010+ carry-forward — Switch Transformer load balance loss, MoE gating, etc.

## Future ablation candidates (post-H009)

- **multi-seed × 3 ablation**: H009 결과 confirm. paper-grade 가능성 검증.
- **H010 = multi_domain_fusion** (MMoE/PLE) — 또 다른 axis 의 mutation.
- **H010 = external_inspirations** (Switch Transformer load balance) — §10.4 의무.
- **H010 = aligned <id, weight> pair encoding** — feature engineering, 데이터 구조 활용.
- **H010 = onetrans_anchor full-data revisit** — H004 archive 재평가.
- **DCN-V2 layer/rank tuning** — H008 sub-H.
- **CAN co-action** — H007 변형.
- **lr scaling, weight_decay, dropout 0.01→0.1** — regularization H.
- **anchor recalibration at extended envelope** — envelope vs mechanism 효과 isolation.
