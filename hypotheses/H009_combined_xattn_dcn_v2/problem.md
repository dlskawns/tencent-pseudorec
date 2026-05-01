# H009 — Combined H007 candidate xattn + H008 DCN-V2 block fusion

## What we're trying to explain

H007 (target_attention sequence axis) PASS marginal at Platform 0.8352 (Δ +0.0035pt vs anchor). H008 (sparse_feature_cross interaction axis, DCN-V2 fusion swap) PASS at Platform 0.8387 (Δ +0.0035pt vs H007, ~+0.004~0.009pt vs anchor). 두 mutation 모두 단독으로 measurable lift. **§0 north star** 의 두 축 — sequence × interaction — 모두 작동 confirmed.

본 H = **두 mutation 동시 적용 (stacking)**. additivity 가정 검증 + §0 두 축 동시 강화 직접 측정. paired Δ 예상:
- additive: +0.007~0.008pt → Platform ~0.842~0.846.
- interference (sub-additive): 한 쪽 mechanism 우세, 다른 쪽 marginal contribution.
- super-additive: 두 mechanism 시너지 (paper-grade 발견).

본 H 의 codepath:
- H007 의 `CandidateSummaryToken` per-domain 적용 → seq 시작에 prepend (L → L+1).
- H008 의 `MultiSeqHyFormerBlock` step 3 fusion = `RankMixerBlock` → `DCNV2CrossBlock` swap.
- 두 mechanism 모두 PCVRHyFormer 의 `MultiSeqHyFormerBlock` 안에서 작동 — block-level gradient 공유. §0 P1 직접 충족.

## Why now

- **H007 + H008 모두 단독 PASS**: combined H 합법 (각 단독 mechanism 검증 끝남).
- **§0 north star 두 축 동시 강화 직접 검증**: sequence axis (candidate xattn) + interaction axis (DCN-V2 cross). 우리 프로젝트의 핵심 가설.
- **anchor 후보 등록 가능**: combined PASS + Δ 충분히 크면 H009 가 새 anchor → 미래 H 의 paired Δ 더 정확.
- **§17.4 카테고리 hybrid**: target_attention + sparse_feature_cross 두 개 동시. 정당화 = "두 PASS 단독 mutation 의 additivity 검증, 새 mechanism 도입 아님".
- **코드 통합 경량**: H007 의 CandidateSummaryToken + H008 의 DCNV2CrossBlock 모두 이미 작성됨. merge 만 (model.py + train.py + infer.py + run.sh).

## Scope
- In:
  - **H007 mechanism 통합**: `CandidateSummaryToken` class (model.py 확장 ~80줄) + per-domain ModuleDict + `_build_token_streams` 안에서 candidate token (item_ns + item_dense_tok mean pool) → per-domain summary → seq 시작 prepend.
  - **H008 mechanism 통합**: `DCNV2CrossBlock` class (model.py 확장 ~80줄) + `MultiSeqHyFormerBlock` 의 fusion dispatch (rankmixer | dcn_v2).
  - 두 mutation flag 동시 활성: `--use_candidate_summary_token --fusion_type dcn_v2 --dcn_v2_num_layers 2 --dcn_v2_rank 8`.
  - 그 외 모든 config: anchor 와 byte-identical envelope.
  - §18 인프라 룰 모두 inherit.
  - **속도 최적화 (선택 적용)**: `--batch_size 512 --patience 3 --num_workers 4 --reinit_sparse_after_epoch 999` (이전 turn 옵션 A). wall ~30~40% 단축 추정.
- Out:
  - candidate xattn 내부 변형 (CAN co-action, multi-layer) — H007 sub-H.
  - DCN-V2 layer/rank tuning — H008 sub-H.
  - fp16 autocast — 별도 turn 코드 작업.
  - lr scaling for batch_size 변경 — 1e-4 → 1.5e-4 (sqrt rule) 보수적 시도 또는 1e-4 그대로 (작은 batch 변경).

## UNI-REC axes
- **Sequential axis**: H007 의 candidate-aware xattn — per-domain seq encoder 출력에 candidate-attended summary 추가.
- **Interaction axis**: H008 의 DCN-V2 explicit polynomial cross — block-level fusion swap, x_0 residual 기반.
- **Bridging mechanism**: 두 mechanism 모두 `MultiSeqHyFormerBlock` 안 작동.
  - candidate summary token: step 1 직후, step 2 직전 (seq encoder 출력 → query decoder 입력).
  - DCN-V2 cross fusion: step 3 (RankMixer 자리, decoded_q + NS tokens fusion).
  - **block-level gradient 공유** — seq + interaction 한 block 안 통합. §0 P1 직접 충족.
- **primary_category**: hybrid (target_attention + sparse_feature_cross). §17.4 정당화 = "두 PASS 단독 의 additivity stacking, new mechanism 도입 아님".
- **Innovation axis**: §0 두 축 동시 강화 first direct verification.

## Success / Failure conditions

**§17.3 binary lift 임계 적용**:
- **Success (additive)**: Δ vs anchor (original_baseline) **platform AUC** ≥ **+0.5pt** AND combined Δ ≥ max(H007 Δ, H008 Δ) (= ≥ +0.0035pt vs anchor 어느 한 mutation 이라도 실패하지 않았다는 것).
- **추가 게이트 (additivity 정량)**:
  - additive 상한: H007 Δ + H008 Δ ≈ +0.007pt
  - 측정 Δ ∈ [+0.005, +0.012pt] → additive 검증 (paper-grade 발견 가능).
  - 측정 Δ ∈ [+0.0035, +0.005pt] → sub-additive (interference). 두 mechanism 중 dominant 한쪽.
  - 측정 Δ < +0.0035pt → 한 쪽 mechanism 망가짐 (더 약한 baseline). 디버그 필요.
- **부수 게이트** (1–4 모두):
  1. Train NaN-free 완주.
  2. §18 인프라 통과.
  3. `metrics.json` 에 두 mechanism flag (`use_candidate_summary_token=true`, `fusion_type=dcn_v2`) 기록.
  4. infer.py 가 두 cfg 모두 read-back → strict load 통과.

## Frozen facts referenced
- Anchor (original_baseline) Platform AUC: ~0.83X.
- H007 PASS marginal: Platform 0.8352, +0.0035pt vs anchor.
- H008 PASS: Platform 0.8387, +0.004~0.009pt vs anchor, +0.0035pt vs H007.
- §0 P1 anti-pattern (concat-late) 회피 룰 — 두 mechanism 모두 block 안 통합.
- §10.5 LayerNorm on x_0 MANDATORY — H008 의 DCN-V2 cross 자동 충족.
- §18 인프라 룰 — H006/H007/H008 검증 완료.

## Inheritance from prior H
- **H007 verdict F-1**: target_attention mechanism PASS → 본 H sequence axis 부분.
- **H008 verdict F-1**: sparse_feature_cross mechanism PASS → 본 H interaction axis 부분.
- **H008 verdict F-3**: additivity 검증 필요 → **본 H 의 핵심 측정**.
- **H008 verdict F-4**: patience=3 + early stop aggressive 추천 → 본 H 적용.
- **H008 verdict F-5**: OOF cohort 갭 좁아지는 패턴.
- **§18 인프라 룰** carry-forward.
