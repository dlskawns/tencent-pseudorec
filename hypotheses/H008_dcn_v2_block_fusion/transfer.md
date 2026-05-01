# H008 — Method Transfer

## ① Source

- **DCN-V2** (Improved Deep & Cross Network) — Wang, Shivanna, Lin, He, Singh, Mehrotra, Cheng. WWW 2021. arXiv:2008.13535. "DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems".
  - Production CTR 시스템 (Google) 에서 검증된 explicit polynomial cross.
  - Low-rank cross (rank r) 로 params 절약.
- 참고 (mechanism class family — sparse_feature_cross 카테고리):
  - **DCN** (Wang et al. ADKDD 2017) — original Deep & Cross Network.
  - **FwFM** (Pan et al. WWW 2018) — field-weighted factorization machine.
  - **FmFM** (Sun et al. WSDM 2021) — field-matrixed factorization.
  - **AutoDis** (Liu et al. CIKM 2021) — automatic discretization for dense features.
  - **CAN** (Bian et al. KDD 2022) — co-action network. candidate-aware variant.

## ② Original mechanism (DCN-V2)

DCN-V2 cross layer:
```
xₗ₊₁ = x₀ ⊙ (Wₗ · xₗ + bₗ) + xₗ
```
- `x₀` ∈ ℝᴰ — input vector (Pre-LN 적용 후).
- `xₗ` ∈ ℝᴰ — l번째 layer state.
- `Wₗ` ∈ ℝᴰˣᴰ — weight matrix (full-rank).
- `bₗ` ∈ ℝᴰ — bias.
- `⊙` — element-wise product.

Stack `L` layers → polynomial degree `L+1` of feature crossing.

**Low-rank variant** (params 절약):
```
Wₗ = Uₗ Vₗᵀ,  Uₗ ∈ ℝᴰˣʳ, Vₗ ∈ ℝᴰˣʳ
```
rank `r` (e.g., 8 ≪ D=64) → params 약 8x 절약.

## ③ What we adopt

- **DCN-V2 low-rank cross block** (`DCNV2CrossBlock`) — token-wise application:
  - Input: `(B, T, D)` (T = num_queries × num_sequences + num_ns).
  - Output: `(B, T, D)`.
  - Pre-LN on `x₀` (CLAUDE.md §10.5 MANDATORY).
  - `num_cross_layers = 2` (default, polynomial degree 3).
  - `rank = 8` (default, low-rank approx).
- **Block-level integration**: `MultiSeqHyFormerBlock` step 3 (token fusion) 에서 `RankMixerBlock` 을 swap. 같은 위치, 같은 역할, 다른 mechanism.
- **Per-token application**: 각 token (decoded_q + NS) 에 cross 독립 적용. token 간 interaction 은 cross 가 아니라 추후 RankMixer 와 같은 token mixer 가 추가 layer 에서 처리 — 단 본 H 는 swap 이라 token mixer 도 없어짐 (여기서 trade-off 노출 — Frame 1 risk).
  - 대안: token mixer + DCN-V2 cross 둘 다 → 2-mutation, §17.2 위배.
  - 본 H 의 단일 mutation 깔끔성 우선.
- CLI flag: `--fusion_type dcn_v2`.

## ④ What we modify (NOT a clone)

- **paper 의 Production setting (huge embedding tables, multi-task) 미반영**: DCN-V2 paper 는 web-scale 환경, 다양한 추가 mechanism 동반. 본 H 는 fusion swap minimum viable form.
- **token-wise application**: paper 는 일반적으로 single feature vector → cross stack. 우리는 (B, T, D) token sequence → 각 token 독립 cross. 다중 token 환경의 적합성은 추후 ablation.
- **rank=8 default**: paper 권장 r ≈ D/4 ~ D/8. d_model=64 → rank 8 (D/8). 변경은 sub-H.
- **num_cross_layers=2**: paper 권장 2-4. 우리는 minimal degree 3 polynomial. 변경은 sub-H.
- **§17.2 one-mutation 엄격**: token mixer + DCN-V2 cross 동시 추가 = 2-mutation. swap 단일 mutation 으로 한정.

## ⑤ UNI-REC alignment

- **Sequential reference**: 변경 없음 — TransformerEncoder + 기존 query decoder 그대로.
- **Interaction reference**: DCN-V2 가 sparse_feature_cross 카테고리의 정수. token-wise polynomial cross.
- **Bridging mechanism**: `MultiSeqHyFormerBlock` step 3 에서 (decoded_q × S domains + NS tokens) 의 cross block 통과 → seq 결과 (decoded_q) + interaction tokens (NS) 가 같은 block 안에서 polynomial cross gradient 공유. **§0 P1 조건 직접 충족** ("시퀀스 인코더와 explicit interaction cross가 같은 블록에서 gradient 공유"), concat-late anti-pattern 회피.
- **primary_category**: `sparse_feature_cross` (§17.4 rotation 추가 충족).
- **Innovation axis**: token-mixing (RankMixer) → explicit polynomial cross (DCN-V2). interaction mechanism class 변경.

## ⑥ Sample-scale viability (Rule UB-1, §10.6)

- 추가 trainable params:
  - 2 cross layers × (Uₗ + Vₗ + bₗ) per layer = 2 × (D × r + r × D + D) = 2 × (64×8 + 8×64 + 64) ≈ 2,176 params.
  - **Removed**: RankMixerBlock params — Linear projections × token-mixing capacity. 비슷한 영역 (~few thousand).
  - Net: ~0 또는 약간 감소.
- Total params: ~198M (변화 무시).
- Sample-scale (5%-data 47k rows): 2K params 학습 가능 — paper-grade cross block 라 데이터 의존성 작음. anchor envelope 동일.
- §10.6 cap 면제 (anchor 와 동일).

## ⑦ Carry-forward rules to honor

- **§10.5 LayerNorm on x₀ MANDATORY**: `DCNV2CrossBlock` 의 첫 step = Pre-LN on x₀. ✓ 직접 충족.
- **§10.6 sample budget cap**: anchor envelope 동일.
- **§10.7 카테고리 rotation**: H008 = 첫 sparse_feature_cross. 추가 충족.
- **§10.9 OneTrans softmax-attention entropy abort**: 본 H 는 OneTrans 미사용 — 미적용. DCN-V2 cross 는 attention 아님 — 룰 적용 안 됨.
- **§10.10 InterFormer bridge gating σ(−2)**: 본 H 는 새 bridge/gate 추가 없음 — 미적용.
- **§17.2 one-mutation**: fusion 클래스 swap 한 mechanism. ✓
- **§17.3 binary success**: Δ ≥ +0.5pt platform AUC vs anchor.
- **§17.4 카테고리 rotation 추가 충족**.
- **§17.5 sample-scale = code-path verification only**: smoke 결과는 mechanism 효과 measurement.
- **§17.6 cost cap**: smoke ~5분 wall, ≪ T2 cap.
- **§17.7 falsification-first**: predictions.md 에 negative-result interpretation.
- **§17.8 cloud handoff discipline**: training_request.md + flat upload + git_sha pin.
- **§18 inference 인프라 룰**: original_baseline 패키지에서 그대로 inherit.
