# OneTrans — Tencent (WWW 2026)

> Source: Tencent UNI-REC team, "OneTrans: Single-stream Transformer for Unified Recommendation with Mixed-Token Causal Attention", arXiv:2510.26104 (WWW 2026 to appear).

## Claim
Recommendation systems traditionally separate sequence modeling (SASRec/DIN/HSTU) from feature interaction (DCN-V2/CAN/PLE), then concatenate near the head. OneTrans claims this concat-late pattern leaves UNI-REC unification gradient on the table. By unifying both as **tokens in a single transformer stream** — S-tokens for sequence positions and NS-tokens for non-sequence features — and routing them through a single attention block with a **mixed causal attention mask** (sequence tokens attend causally; non-sequence tokens attend bidirectionally to all S-tokens up to the candidate position), the model captures seq×int interactions natively at every layer. Pyramid pruning progressively drops low-attention S-tokens deeper in the stack to bound compute.

## Method
- **Tokenization**:
  - S-tokens: per-event embeddings of sequence history (per-domain or merged), positionally encoded.
  - NS-tokens: scalar/dense/multi-val features chunked into N parameter-free token groups (similar to RankMixer's equal-split chunking), each becomes 1 token of d_model.
- **Mixed causal attention mask**:
  - S-token → S-token: causal upper-triangular (no future leakage).
  - NS-token → S-token: full bidirectional attention up to the candidate position.
  - NS-token → NS-token: full self-attention.
  - Candidate token: attends to all S/NS tokens up to its time.
- **Pyramid pruning**:
  - Layer ℓ retains top-K_ℓ S-tokens by attention probability mass; K_ℓ is monotone decreasing in ℓ.
  - NS-tokens are NEVER pruned.
- **Output head**: take the candidate token's final-layer hidden state, project to logit.

## What It Guarantees (formal properties claimed)
- **Universality**: by setting NS-tokens to zero and removing pruning, OneTrans reduces to SASRec; by setting S-tokens to zero, it reduces to a feature-interaction transformer (FT-Transformer-like). UNI-REC concat-late blocks are a strict subset.
- **Compute bound**: with pyramid pruning at rate r, layer-ℓ FLOPs are O(r^ℓ · L · d). Authors report 3–5× FLOPs reduction at fixed AUC vs full-attention.
- **Empirical**: AUC lift of +0.4 to +1.2 pt over CAN/DIN/DIEN baselines on Tencent internal CTR datasets at scale ≥ 100M users.

## Applicability to Ours (TAAC 2026 UNI-REC Challenge)
Applied as the **primary literary reference for H015** (iter 17, scouting only).

### Mapping to our problem
| OneTrans token type | Our data |
|---|---|
| S-tokens | 4-domain seq (A=701, B=571, C=449, D=1100 events/user). Per-domain causal embeddings; cross-domain attention via shared d_model. |
| NS-tokens | 35 user_int scalar + 13 item_int scalar + 8 aligned `<id, weight>` pairs + 10 user dense + 12 item multi-val. Chunked into N groups via RankMixer-style equal-split (parameter-free). |
| Candidate token | The candidate item embedding + its bucket id; attends to all S/NS up to `timestamp`. |
| Mixed causal mask | Honors our `label_time` train/val split (CLAUDE.md §4.3) — no exposure→feedback leakage. |

### Sample-scale risk (Rule UB-1)
- Full OneTrans at d_model=64, 4 layers, L=64+L_NS would be ≈ 250k–500k params → **violates §10.6 sample budget** (≤ 2146).
- H015 must trim to a **delta-on-E009 minimal backbone**: ≤ 2 attention layers at d_model=8 OR token-restructuring of x0 only without new attention block. The OneTrans **structural lesson** (token-level UNI-REC fusion) survives the trim; the **scale claim** does not transfer.

### Carry-forward conflicts to track
- H010 F-1 (softmax routing → uniform collapse at sample scale): OneTrans uses softmax attention with N_S+N_NS≈100 tokens per row; at 124 positives across 4 layers, attention probabilities risk uniform collapse. Diagnostic to log: `attn_entropy_per_layer` ≥ 0.95·log(N) ⇒ abort.
- H014 F-2 (arm-conditional antagonism w/o x0 LN anchor): OneTrans replaces x0 LN with token-level LayerNorm. Must preserve a SCALAR-anchored side branch (e.g., DCN-V2 cross on raw x0 vector parallel to the transformer) for stability.

## Quote (core paper claim)
> "We unify sequential and non-sequential features as tokens in a single transformer stream, with a mixed causal attention mask that lets feature tokens attend bidirectionally to the user history while preserving the autoregressive structure of the sequence itself. This eliminates the architectural barrier between sequence modeling and feature interaction modeling that has defined production recommender systems for the past decade." (OneTrans abstract, paraphrase from arXiv:2510.26104)

## Link
- arXiv: https://arxiv.org/abs/2510.26104
- Venue: WWW 2026 (to appear)

## Carry-forward rules (paper-card distillation, applied to our H015)
- **R1**: Token-level UNI-REC fusion is the design paradigm; at sample scale, fuse via NS-token equal-split on x0 ONLY, defer attention layers.
- **R2**: Mixed causal mask is honored automatically by `label_time` split; no extra masking code needed.
- **R3**: Pyramid pruning is a compute optimization; not relevant at N=1000.
- **R4**: Candidate-token-as-output is structurally similar to our existing `cand_*` features feeding the cross — adoption is incremental.
