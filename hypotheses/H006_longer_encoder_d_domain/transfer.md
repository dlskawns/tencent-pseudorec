# H006 — Method Transfer

## ① Source

LongerEncoder (organizer-supplied class in `model.py:616`). 디자인 영감 paper 들:
- **SIM** (Search-based Interest Modeling, Pi et al. CIKM 2020) — long sequence retrieval via candidate-aware top-K.
- **ETA** (End-to-End Target Attention with Long History, Chen et al. arXiv:2108.04468) — top-K via SimHash projection.
- **TWIN** (Two-stage Interest Network for User Behavior Modeling, Chang et al. KDD 2023) — two-stage retrieval-then-attention.
- **HSTU** (Hierarchical Sequential Transduction Unit, Meta 2024) — long-context backbone with hierarchical pruning.

LongerEncoder 의 구체 구현은 paper 1:1 reproduce 가 아니라 organizer 의 **top-K self-attention compression** — paper 들의 핵심 메커니즘 (long-tail 보존 + compute bound) 을 borrow.

## ② Original mechanism (LongerEncoder, organizer code)

기본 TransformerEncoder (`model.py:544`) 는 layer 당:
1. RoPE multi-head self-attention (Q · K^T softmax, V).
2. SwiGLU FFN.
3. Pre-LN residual.
- Compute: O(L² · d) per layer.

LongerEncoder (`model.py:616`) 는 layer 당:
1. RoPE multi-head self-attention (위와 동일) **단 K = top-K (default 50)** 만 보존.
2. 매 layer 시작 시 attention probability mass 계산 → top-K positions 선택 → 나머지 mask 처리.
3. SwiGLU FFN.
4. Pre-LN residual.
- Compute: O(L · log K · d) (top-K selection 포함).
- Input length 자유 — sequence 가 1100 events 든 100 events 든 같은 K=50 token 만 처리.

## ③ What we adopt

- LongerEncoder 그대로 (organizer 코드).
- `--seq_encoder_type longer` CLI flag (train.py 의 argparse 이미 지원, choice 에 `'longer'` 포함).
- `--seq_top_k 50` default (organizer 가 baseline 으로 검증한 값, 변경 없음).
- 4 도메인 모두 동일 encoder type 적용 (CLI 가 global). A/B/C 는 seq_max_lens 64/64/128 ≤ K=50 미만이므로 effective behavior 가 transformer 와 거의 동일 — 마지막 layer 의 sub-K 시퀀스에서 차이 무시.
- 그 외 모든 hyperparameter (lr, sparse_lr, batch_size, seed, dropout, num_epochs, num_hyformer_blocks, d_model, NS-tokens, BCE loss 등) original_baseline 과 byte-identical.

## ④ What we modify (NOT a clone)

- **paper claim 의 candidate-aware retrieval 미적용**: SIM/TWIN paper 들은 candidate item 의 representation 으로 query 만들어 history 에 cross-attention 으로 top-K retrieval. 우리 LongerEncoder 는 self-attention probability mass 기반 → less targeted. **이유**: organizer-supplied 로 이미 구현된 메커니즘 그대로 사용 (코드 변경 0). candidate-aware retrieval 은 target_attention 카테고리 별도 H.
- **seq_max_lens 확장 안 함**: paper 들은 1000+ events input 사용. 우리 smoke envelope 은 seq_d=128. **즉 LongerEncoder 의 top-K=50 이 D 의 128 token 중 50 만 보존** — 이론상 input 자체가 짧아서 이득 작음. 단 organizer 가 production data 에서 D=1100 events 에 적용 시 lift 본 의도라 그 신호 측정 가능. seq_max_lens 확장은 별도 H (compute cost 큼).
- **per-domain encoder type 미분리**: D 만 longer, A/B/C 는 transformer 인 ideal 비교 안 함. CLI flag 가 global 이라 코드 수정 없으면 4 도메인 전부 swap. 별도 H 로 carry-forward.
- **§17.2 one-mutation 엄격 적용**: encoder type 만 변경. seq_top_k tuning 별도 H, seq_max_lens 변경 별도 H, candidate-aware retrieval 별도 H.

## ⑤ UNI-REC alignment

- **Sequential reference**: SIM / ETA / TWIN / HSTU. 모두 long-tail sequence 처리 메서드. 본 H 의 LongerEncoder 가 그 영역의 organizer-supplied 변형.
- **Interaction reference**: 변경 없음. RankMixerNSTokenizer (model.py:1070) 그대로.
- **Bridging mechanism**: 변경 없음. MultiSeqHyFormerBlock (model.py:850) 의 token fusion 단계 그대로.
- **primary_category**: `long_seq_retrieval` (§17.4 rotation 첫 충족 추가).
- **Innovation axis**: 모델 capacity 가 아니라 **데이터 속성 (D 도메인 long-tail) 의 정보 보존**. 즉 동일 model size + 동일 hyperparameter 에서 attention 계산 방식만 바꿔서 더 많은 D 행동 데이터를 효과적으로 사용.

## ⑥ Sample-scale viability (Rule UB-1, §10.6)

- 추가 trainable params: **0** 또는 매우 작음 (LongerEncoder 와 TransformerEncoder 가 동일 backbone 형태, 다른 점은 top-K masking logic — gradient 흐르는 weights 동일). organizer 코드 직접 확인 시 no extra Linear/projection.
- Total params: PCVRHyFormer ~198M (embedding tables dominant) 그대로.
- Sample-scale (5%-data, 47k rows): top-K=50 selection 이 1-epoch 으로 학습 가능 여부 — paper-검증 부재. 단 selection 은 attention probability 기반 (이미 학습된 attention 의 부산물), explicit train signal 필요 없음. anchor envelope 동일이라 detect floor 안에서 measurement.
- §10.6 cap 안 (anchor 와 동일).

## ⑦ Carry-forward rules to honor

- **§10.5 LayerNorm on x0**: 변경 없음. PCVRHyFormer Pre-LN 그대로 충족.
- **§10.6 sample budget cap**: anchor envelope 동일.
- **§10.7 카테고리 rotation**: H006 = 첫 long_seq_retrieval. 추가 충족.
- **§10.9 OneTrans softmax-attention entropy abort**: 본 H 는 OneTrans 미사용 → 미적용. (단 LongerEncoder 의 self-attention 도 softmax 기반 → 만약 K=50 영역에서 uniform collapse 하면 비슷한 abort 룰 trigger 가능. carry-forward: H006 결과 attn_entropy 측정 옵션, 별도 H 로 instrumentation 추가.)
- **§17.2 one-mutation**: encoder type 만 변경. ✓
- **§17.3 binary success**: Δ ≥ +0.5pt vs original_baseline-anchor.
- **§17.5 sample-scale = code-path verification only**: smoke val_AUC 는 mutation 효과 측정용. full-data 결과는 별도 row.
- **§17.7 falsification-first**: predictions.md 에 negative-result interpretation 명시.
- **§17.8 cloud handoff discipline**: training_request.md + flat upload + git_sha pin.
- **§18 inference 인프라 룰**: original_baseline 패키지에서 그대로 inherit (§18.1–§18.5 모두 적용된 infer.py + dataset.py + make_schema.py).
