# InterFormer — Meta (CIKM 2025)

> Source: Meta AI, "InterFormer: Bidirectional Bridging of Interaction, Sequence, and Cross Architectures for CTR Prediction", arXiv:2411.09852 (CIKM 2025).

## Claim
Production CTR systems typically run **three independent sub-networks** — a feature-interaction tower (DCN-V2/CAN), a sequence tower (DIN/SASRec), and a cross-feature tower (FwFM/AutoDis) — and fuse near the head via concat or weighted sum. InterFormer's claim: this **late fusion** loses information because the three towers never read each other's intermediate representations. By inserting **bidirectional bridges** at every layer between the three architectures, each tower's hidden states inform the others' next-layer computation, producing a strictly more expressive joint model with empirically lower variance.

## Method
- **Three parallel arches** at the same depth ℓ:
  - **Interaction arch**: DCN-V2 cross layers on x0 (low-rank cross matrix at each layer).
  - **Sequence arch**: SASRec-style self-attention over the user history (causal mask).
  - **Cross arch**: pairwise field interaction (FwFM-style) on x0 + cross-arch readouts.
- **Bidirectional bridges** at each layer ℓ:
  - Each arch produces a layer-ℓ representation `h_int^ℓ`, `h_seq^ℓ`, `h_cross^ℓ`.
  - Bridge `B_ij^ℓ`: `h_j^{ℓ+1} = h_j^{ℓ+1} + W_ij^ℓ · h_i^ℓ`, for all (i, j) pairs.
  - W_ij is a low-rank projection (rank r=4 to 8).
- **Bridge gating**: each W_ij is gated by a learned scalar σ(α_ij^ℓ); paper recommends initialization α_ij^ℓ = -2 (sigmoid ≈ 0.12) so bridges start nearly off and grow only if useful.
- **Final head**: concat `[h_int^L; h_seq^L; h_cross^L]` → MLP → logit.

## What It Guarantees (formal properties claimed)
- **Strict expressive superset**: with all bridges set to 0, InterFormer reduces to three independent towers (concat-late). With bridges active, model class is a superset.
- **Variance reduction (empirical)**: paper reports AUC variance across seeds drops by 30–50% vs concat-late at matched parameter count, attributed to the regularizing effect of cross-arch read-write.
- **Lift bound (claimed)**: +0.3 to +0.8 pt AUC over best concat-late baseline on Meta production CTR data.
- **Bridge ablation property**: removing any single bridge (e.g., int→seq) degrades by no more than 30% of the full-bridge gain — bridges are partially complementary, not orthogonal.

## Applicability to Ours (TAAC 2026 UNI-REC Challenge)
Applied as **secondary literary reference for H015** — provides the EXPLICIT BRIDGE design pattern that justifies our planned domain-anchored query token (DAMTB).

### Mapping to our problem
| InterFormer arch | Our existing component |
|---|---|
| Interaction arch | E009's DCN-V2 (L=1) on x0. Already in place. |
| Sequence arch | E009's per-domain BucketBilinearSeq + mean-pool on 4 seq domains. |
| Cross arch | Currently MISSING in E009 (FwFM was tested in H012 but REFUTED-redundant by F2). Domain-anchored query token would slot here. |
| Bridge B_int→seq | NEW: x0 cross output reads into each domain's seq encoder gate. |
| Bridge B_seq→int | NEW: each domain's pooled-seq output reads into the cross's V·U bilinear weight. |
| Bridge B_cross→both | NEW (DAMTB innovation): per-domain anchor query token broadcasts to both x0 cross and seq encoders. |

### Sample-scale risk (Rule UB-1)
- Full InterFormer with 3 arches × L=4 layers × bridges = ≈ 50k–100k params → **violates §10.6 sample budget**.
- H015 must trim to **L=1 layer with single bridge** (the DAMTB anchor query token, conceptually `B_anchor → both`).
- Bridge gating with init α = −2 (sigmoid ≈ 0.12) is a CRITICAL transfer detail to retain — prevents bridges from immediately destabilizing training when the receiving arch is already converged (matches H005 F-1 scalar-gate-redundancy lesson).

### Carry-forward conflicts to track
- H012 F-2 (FwFM redundant given DCN-V2 on 8 aligned pairs): InterFormer's "cross arch" is FwFM-style. At sample scale, the cross arch is the redundant one; H015 should COLLAPSE the cross arch into the bridge itself rather than re-instantiate FwFM.
- H013 F-2 (mean-pool redundancy for any tensor seq encoder consumes): bridges must read NEW information, not bucket_counts that BucketBilinearSeq already consumes. Anchor query token should be cand-bucket-driven (target-side) or session-boundary-driven, not user-bucket-histogram-driven.

## Quote (core paper claim)
> "Three architectures – feature interaction, sequence modeling, and cross-feature transformation – are universally present in production CTR systems but are universally fused only at the head. We show that bidirectional bridges between architectures, gated to start near zero and grow only as utility justifies, produce strictly more expressive models with reduced cross-seed variance, at modest additional parameter cost." (InterFormer abstract paraphrase from arXiv:2411.09852)

## Link
- arXiv: https://arxiv.org/abs/2411.09852
- Venue: CIKM 2025

## Carry-forward rules (paper-card distillation, applied to our H015)
- **R1**: Bridge gating init = sigmoid(−2) ≈ 0.12 — start near off. Honors H005 F-1 (scalar gate redundancy if init too active).
- **R2**: Bridge rank r=4 — minimum that retains the +AUC lift in their ablations.
- **R3**: Bridge readouts must be ABLATABLE — i.e., logged separately so we can report bridge_active_share at end of training (matches our `gate_per_dim_std` diagnostic from H007).
- **R4**: Cross arch (FwFM-style) is redundant in our setup per H012 F-2 — DAMTB COLLAPSES the cross arch into the anchor-query-bridge instead of re-instantiating it.
