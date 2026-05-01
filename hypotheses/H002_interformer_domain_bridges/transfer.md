# H002 — Method Transfer

## ① Source
[`papers/unified_backbones/interformer_meta.md`](../../papers/unified_backbones/interformer_meta.md) — InterFormer (Meta CIKM 2025, arXiv:2411.09852).

## ② Original mechanism
InterFormer 는 production CTR 시스템에 보편적으로 존재하는 3 architectures (Interaction × Sequence × Cross) 를 하나의 모델에서 매 layer 마다 **bidirectional bridges** 로 연결. 각 bridge `B_ij^l` 은 source arch `i` 의 layer-l 출력 `h_i^l` 을 low-rank projection (W_down, W_up) 과 sigmoid-gate (`α_ij^l`) 통과시켜 target arch `j` 의 layer-(l+1) 출력에 더한다: `h_j^{l+1} += sigmoid(α_ij^l) · W_ij · h_i^l`. Gate init `α=-2` (sigmoid≈0.12) — bridges 가 near-off 로 시작해 utility 따라 grow. Paper 결과: 동일 param 수에서 +0.3–0.8 pt AUC + cross-seed variance 30–50% 감소.

## ③ What we adopt
- Low-rank bottleneck: `W_down ∈ R^{d×r}`, `W_up ∈ R^{r×d}`, **rank r=4** (paper 권장).
- Scalar gate per bridge: `gate ∈ R^1`, init **-2.0** → sigmoid(-2) ≈ 0.119.
- Init: `W_down` xavier(gain=0.1), `W_up` zeros → init 시점 bridge 출력 0 (gate 와 무관, 학습으로 grow).
- Per-block bridges (매 `MultiSeqHyFormerBlock` 안에서 개별 bridge module).

## ④ What we modify (NOT a clone)
- **Source/target = arch가 아니라 도메인**: paper 의 3 arch (Interaction × Sequence × Cross) 가 우리는 4 sequence domains (a/b/c/d). HyFormer 의 sequence encoder 들이 paper 의 "arch" 자리.
- **Bridge 위치**: paper 는 per-layer cross-arch. 우리는 `MultiSeqHyFormerBlock` 안 step 1 (per-domain seq encoding) 직후, step 2 (cross-attention query decoding) 직전. 이유: query decoding 이 per-domain seq 를 collapse 시키므로 그 전에 enrich 해야 함.
- **Pooling**: paper 는 모든 token-level 간 bridge. 우리는 mask-aware **mean-pool to (B, D)** 후 broadcast — 도메인 별 length 가 다름 + memory 절감.
- **3rd arch (Cross arch) 미적용**: paper 의 explicit cross arch (FwFM-style) 는 H005 로 분리. H002 는 sequence axis 안 inter-domain bridges 만.
- **Bridge gate 모니터링 의무**: 학습 종료 시 12개 bridge 의 final gate value 분포 측정 → verdict.md F-N. 모두 0.05 이하면 paper claim 비응답 신호.

## ⑤ UNI-REC alignment
- **Sequential reference**: HyFormer per-domain `TransformerEncoder` (model.py:544).
- **Interaction reference**: 변경 없음 — RankMixer NS path 보존.
- **Bridging mechanism**: 4 sequence encoder 간 bridges 가 **layer-level cross-domain gradient 공유** 추가. CLAUDE.md §0 P1 정의 ("seq + interaction 한 블록 gradient 공유") 의 "seq" 측면 강화.
- **primary_category**: `unified_backbones` (재진입 — challengers.md §재진입정당화).
- **Innovation axis**: paper 의 cross-arch bridges 를 cross-domain bridges 로 변환. arch dimension → domain dimension. paper 의 메커니즘 클래스 (low-rank gated bridge + α=-2 init) 그대로, 적용 대상만 다름.

## ⑥ Sample-scale viability (§10.6 Rule UB-1)
- 추가 trainable params: **12,312** (12 bridges × 1,026 params each, 측정값).
- Total params (production): ~198M base + 12k bridge = +0.006% (negligible).
- Sample-scale (5% data, 94k rows): 12k 새 params 학습 가능 여부 불확실. paper 의 효과는 더 큰 data 에서 검증된 것. 단, gate near-off init 으로 **bridge 가 학습 안 되어도 baseline 은 보존** (degenerate to no-op).
- demo_1000 smoke run 검증 완료: forward + backward OK, gate init 정확 (0.1192).

## ⑦ Carry-forward rules to honor
- **§10.5 LayerNorm on x0 MANDATORY**: bridge 가 next_seqs[j] 에 update 더할 때 LN 통과 안 함. 그러나 add 후 다음 cross_attn block 의 `ln_mode='pre'` 가 자동 정규화 → §10.5 spirit 보존.
- **§10.10 InterFormer bridge gating σ(-2) init**: 본 H 가 그 rule 의 **첫 적용**. 정확 -2.0 사용.
- **§10.9 OneTrans softmax-attention entropy abort**: 본 H는 새 softmax 안 추가. 미적용.
- **§10.7 카테고리 rotation**: 위 challengers.md §재진입정당화 충족.
- **§17.2 one-mutation-per-experiment**: 한 component 클래스 추가 (`_DomainBridge`) + 한 블록 forward 수정 — 단일 mutation.
- **§17.3 binary success**: Δ ≥ +0.5pt val_AUC vs E_baseline_organizer. 미달 → REFUTED.
- **§17.7 falsification-first**: predictions.md 에 명시.
- **UB-3 (bridge가 새 정보 읽음)**: 도메인간 cross-domain 정보 = 현 baseline 어디서도 안 흐르는 새 channel. 충족.
- **UB-4 (softmax 라우팅 collapse 차단)**: bridge는 sigmoid gate scalar — softmax 아님. ≤ 12 bridge per block. 안전.
