# H009 — Method Transfer

## ① Source

- **H007**: candidate-aware cross-attention pooling (DIN/CAN/SIM/TWIN/HSTU/OneTrans family idea, modern multi-head xattn implementation). PASS marginal at Platform 0.8352.
- **H008**: DCN-V2 explicit polynomial cross fusion swap (Wang et al. WWW 2021). PASS at Platform 0.8387.
- 본 H 는 **새 paper source 도입 안 함**. 두 단독 PASS mutation stacking.

## ② Original mechanism (H007 + H008 결합)

**H007 component (sequence axis)**:
- `CandidateSummaryToken` per-domain — Pre-LN multi-head cross-attention. Q=candidate (1 token), K=V=seq_tokens. padding mask + all-pad guard.
- candidate token = (item_ns + item_dense_tok) mean pool.
- per-domain (4 domains) 독립 ModuleDict.
- 통합: per-domain seq encoder 출력 → CandidateSummaryToken 호출 → seq 시작 prepend (L → L+1).

**H008 component (interaction axis)**:
- `DCNV2CrossBlock` — token-wise polynomial cross. `x_{l+1} = x_0 ⊙ (V_l(U_l(x_l))) + x_l`.
- Stack 2 cross layers → polynomial degree 3.
- Low-rank W = U V^T (rank=8 = D/8 saving).
- Pre-LN on x_0 (CLAUDE.md §10.5 MANDATORY 직접 충족).
- 통합: `MultiSeqHyFormerBlock` step 3 fusion 의 dispatch — `fusion_type='dcn_v2'`.

## ③ What we adopt

- **두 mechanism 동시 활성**:
  - `--use_candidate_summary_token` (H007).
  - `--fusion_type dcn_v2 --dcn_v2_num_layers 2 --dcn_v2_rank 8` (H008).
- 모든 hyperparameter (lr, dropout, batch_size, NS-tokens, num_queries 등) 안커 와 byte-identical.
- §18 인프라 룰 inherit (batch=256 default + PYTORCH_CUDA_ALLOC_CONF + universal handler + 진단 로그).
- envelope: extended (10 epoch × 30%, patience=3 — H008 verdict F-4 carry-forward).
- 속도 최적화 (선택): `--batch_size 512 --num_workers 4 --reinit_sparse_after_epoch 999` 적용 시 wall ~30~40% 단축.

## ④ What we modify (NOT a clone)

- **새 mechanism 도입 없음**: 본 H 는 stacking 이지 mechanism 변경 아님.
- **lr scaling 보류**: batch_size 256 그대로 유지 (또는 옵션 A 적용 시 512 — 작은 변경이라 lr 1e-4 그대로). batch_size 변경은 secondary 변수, single mutation 정신 위배 위험 → 본 H 는 batch_size 256 으로 H007/H008 paired 비교 유지.
- **patience=3** for early stop (H008 verdict F-4) — 본 H 의 단일 변경. 단 H006/H007 은 patience=5 였음 — fair 비교 위해 patience 도 envelope 일부로 카운트.
- **lr 변경 없음** — H007/H008 모두 lr=1e-4. 본 H 도 1e-4. lr scaling 별도 H.

## ⑤ UNI-REC alignment

- **Sequential reference**: H007 의 candidate-aware xattn (DIN/CAN/SIM/TWIN/HSTU/OneTrans family idea). 본 H 그대로 활용.
- **Interaction reference**: H008 의 DCN-V2 cross. 본 H 그대로 활용.
- **Bridging mechanism**: 두 mechanism 모두 `MultiSeqHyFormerBlock` 안 작동:
  - candidate summary: step 1 직후 (seq encoder 출력 prepend).
  - DCN-V2 cross: step 3 fusion (decoded_q + NS tokens 의 polynomial cross).
  - **block-level gradient sharing 직접 충족** — seq encoder 결과 + candidate summary 가 query decoder 통과 후 NS tokens 와 같은 block 안 cross 에서 polynomial interaction. §0 P1 가장 강한 형태.
- **primary_category**: hybrid (target_attention + sparse_feature_cross). §17.4 정당화 = stacking, 새 mechanism 아님.
- **Innovation axis**: §0 두 축 동시 강화 first direct verification.

## ⑥ Sample-scale viability (Rule UB-1, §10.6)

- 추가 trainable params:
  - H007 candidate summary: ~50K (per-domain × 4 modules × 12.5K/module).
  - H008 DCN-V2 cross: ~4K (2 layers × 2,176/layer × 2 hyformer blocks).
  - **Total**: ~54K.
- Total params: ~198M (변화 +0.027% 무시).
- Sample-scale (extended envelope 5.1M rows × 10 epoch = 51M sample steps): 충분.
- §10.6 cap 면제 (anchor envelope).

## ⑦ Carry-forward rules to honor

- **§10.5 LayerNorm on x_0 MANDATORY**: H008 의 DCN-V2 cross 가 Pre-LN on x_0 직접 충족.
- **§10.6 sample budget cap**: anchor envelope 동일.
- **§10.7 카테고리 rotation**: §17.4 hybrid 정당화 (stacking, 새 mechanism 아님).
- **§10.9 OneTrans softmax-attention entropy abort**: H007 candidate summary 의 cross-attention 도 softmax. uniform collapse 시 lift 약화 신호 — instrumentation 별도 sub-H.
- **§17.2 one-mutation**: 엄밀히는 stacking 이라 2-mutation. 단 §17.2 anchor exemption 정신과 유사 — H007/H008 단독 검증 후 stacking sub-H 는 합법. challengers.md §재진입정당화 인용.
- **§17.3 binary success**: Δ ≥ +0.5pt vs anchor. 추가 additivity 검증 (P2 sub-criterion).
- **§17.4 카테고리 rotation 정당화**.
- **§17.5 sample-scale**: extended envelope 결과는 mechanism 효과 measurement.
- **§17.6 cost cap**: extended ~3-4시간, patience=3 으로 cap 안.
- **§17.7 falsification-first**: predictions.md 에 sub-additive vs additive vs super-additive interpretation.
- **§17.8 cloud handoff discipline**: training_request.md + flat upload + git_sha pin.
- **§18 inference 인프라 룰**: H008 패키지 inherit + H007 의 candidate_summary cfg.get 추가.
