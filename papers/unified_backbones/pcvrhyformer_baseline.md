# PCVRHyFormer — TAAC 2026 Organizer Baseline

> Source: TAAC 2026 organizers, code at `competition/{dataset.py, model.py, trainer.py, train.py, run.sh, utils.py, ns_groups.json}` (received 2026-04-19, audited via Explore agent).

## Claim
The competition organizers ship a **HyFormer-named** production baseline named **PCVRHyFormer** = "Per-domain CVR HyFormer". The claim implicit in shipping it as the baseline: a multi-stage architecture with **per-domain seq encoders → per-domain query decoders → joint token fusion → RankMixerBlock** is a strong starting point for the UNI-REC task. The dual optimizer (Adagrad for sparse / AdamW for dense) and parameter-free NS-token equal-split are presented as production-tested choices.

## Method
### MultiSeqHyFormerBlock (`competition/model.py:850-980`) — the core architecture
1. **Per-domain seq encoders** (one per domain A/B/C/D), three options:
   - `swiglu`: attention-free SwiGLU residual block (low-cost default for short seqs).
   - `transformer`: Pre-LN self-attention + FFN (default for `train.py`).
   - `longer`: top-K compressed cross-attention (for D's 1100-event right tail).
2. **Per-domain query decoders**: each domain emits a `query_token` (1 token of d_model) representing the domain's sequence summary, conditioned on candidate side.
3. **Joint token fusion**: 4 query_tokens are concatenated with NS-tokens (parameter-free RankMixer equal-split chunking on x0 — see below) into a flat token sequence.
4. **RankMixerBlock** (`model.py:1070-1189`): MLP-mixer-style block over the token sequence — parameter-free token-mixing then channel-mixing MLP. Acts as the final fusion before the head.

### CrossAttention (`model.py:295-296`)
- `ln_mode='pre'` confirms Pre-LN convention — matches our §10.5 LN(x0) MANDATORY rule (H002 F-1 carry-forward) at the architectural level.

### RankMixer NS tokenizer (`model.py:1070-1189`)
- **Parameter-free equal-split chunking**: given x0 of dim d_x and a target N_NS tokens, chunk x0 into N_NS contiguous slices, each becomes 1 token of d_model via a learned linear (the only learnable params are the linear projection, NOT the chunking itself).
- **Decouples NS token count from feature grouping**: feature groups in `ns_groups.json` map fids to logical groups (e.g., `user_ns_groups: {U1: [1,15], U2: [48,49,...]}`), but the chunking IGNORES the groups and just takes equal-size slices. This is the novel trick — it is "parameter-free" relative to alternatives like Set Transformer or Perceiver IO that require learned query tokens per NS chunk.
- Default config: `user_ns_tokens=5`, `item_ns_tokens=2` (from `run.sh`).

### Trainer (`competition/trainer.py`)
- Dual optimizer pattern:
  - **Adagrad** on sparse params (embeddings) with `lr=0.05` — high LR per-param adaptive for embedding tables.
  - **AdamW** on dense params with `lr=1e-4`, `betas=(0.9, 0.98)` — Pre-LN-friendly betas (matches H002 F-1 lineage).
- Validation tracks **AUC ONLY** (no GAUC, no HR@10, no NDCG@10) — same as our reports/EDA harness.

### Train script (`competition/train.py`)
- Loss: **Focal loss with γ=2.0, α=0.1** defaults — matches our H008 F-1 carry-forward (focal γ=2.0 robust across ±1 grid per H011) almost verbatim. α=0.1 differs from our class-balanced β=0.9999 derivation (Cui 2019); equivalent in spirit but H008 (Cui) is theoretically grounded for our 7.1:1 imbalance.
- Batch size: 256, epochs: 999 with patience: 5 — early-stopping-driven.
- Default `seq_encoder_type=transformer`.

## What It Guarantees (production stipulations)
- The organizers' baseline AUC on the (full, not yet released) target leaderboard is the de-facto floor competitors must beat.
- The dual optimizer + RankMixer NS tokenizer combination has been empirically validated at Tencent's full-data scale.
- Pre-LN throughout is consistent with our H002 F-1 finding (LayerNorm MANDATORY).

## Applicability to Ours (TAAC 2026 UNI-REC Challenge)
Applied as **production reference for H015** AND as a **method-transfer source** for the RankMixer NS tokenizer specifically.

### CLAUDE.md violations in baseline (must patch before any local run)
- **§4.3 violated**: `dataset.py` splits by **Row Group position**, NOT by `label_time`. This re-introduces the exposure→feedback leakage that §4.3 was designed to prevent.
  - Patch: replace Row Group split with sklearn `train_test_split` keyed on `label_time` percentile.
- **§4.4 violated**: NO OOF holdout. Every user is potentially in train.
  - Patch: apply our 10% user OOF holdout (seed=42) BEFORE any of the dataset's other split logic.

### Carry-forward CONFIRMATIONS (production validates our 16-iter findings)
- Pre-LN everywhere (`ln_mode='pre'`) → confirms **§10.5 H002 F-1 LayerNorm on x0 MANDATORY**.
- Focal γ=2.0 is the production default → confirms **H008 F-1 focal_cb γ=2.0 is full-data handoff candidate**, γ-robust per H011.
- Dual optimizer with AdamW betas=(0.9, 0.98) → matches Pre-LN convention; consistent with our `.venv-arm64` configs.

### Method-transfer kernel: RankMixer NS tokenizer
This is the **direct extractable mechanism** for H015:
- It is parameter-free relative to alternatives.
- It restructures x0 without adding new params (the only learnable is the projection from chunk to d_model, which is a small linear).
- Application to E009: replace x0's flat-vector path into DCN-V2 with an N_NS-token chunked path; DCN-V2 cross still consumes the same total dimensions.

### Sample-scale viability (Rule UB-1 check)
- Full PCVRHyFormer at d_model=64, num_hyformer_blocks=2, hidden_mult=4 ≈ **250k–500k params** → **violates §10.6 sample budget** by 100×+.
- We CANNOT run PCVRHyFormer end-to-end on `demo_1000.parquet`. Treat as paper-level reference only.
- Effort estimate to make runnable on demo_1000 (post-§4.3/§4.4 patch): ~2 hours of patch work, but the resulting model would still be 100× over budget — ineffective without full data.

## Quote (implicit baseline claim)
> "Per-domain sequence encoders feed query decoders that produce one token per domain; these tokens are mixed with parameter-free NS-token chunks of the non-sequence feature vector via a RankMixerBlock; the resulting token bag is read by the candidate-aware head. Sparse parameters use Adagrad at 0.05; dense parameters use AdamW at 1e-4 with betas=(0.9, 0.98)." (Synthesized from `competition/model.py:850-980` and `competition/trainer.py`.)

## Link
- Local source: `competition/` directory in this repo. No external link (organizers do not yet publish the baseline as a paper).

## Carry-forward rules (baseline-card distillation, applied to our H015)
- **R1**: Extract the RankMixer NS tokenizer (`model.py:1070-1189`) as the SAMPLE-SCALE method-transfer kernel. Re-apply to E009's x0 path; that alone is the iter-18 minimal viable experiment.
- **R2**: Do NOT replicate the full MultiSeqHyFormerBlock at sample scale — params blow §10.6.
- **R3**: Adopt dual optimizer pattern (Adagrad/AdamW) ONLY when sparse embedding tables are introduced; current E009 has no sparse table large enough to justify it.
- **R4**: Patch the dataset.py split logic before any reproducibility run; CLAUDE.md §4.3/§4.4 are firm.
- **R5**: Confirm `ln_mode='pre'` on every cross-attention path we ever ADD — production agrees with H002 F-1.
