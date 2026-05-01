# H008 — DCN-V2 explicit polynomial cross as block-level fusion swap

## What we're trying to explain

H007 verdict F-1: candidate-as-attention-query mechanism class (target_attention 카테고리) 가 우리 데이터에서 +0.5pt 임계 도달 — sequential axis 강화 first confirmed PASS. 다음 자연스러운 step = **interaction axis 강화**, §0 north star 의 두 축 중 다른 한 쪽 검증.

CLAUDE.md §0 P1 진입 조건: "**시퀀스 인코더와 explicit interaction cross가 같은 블록에서 gradient 공유** (concat-late는 P0까지만)". Anti-pattern: "두 축을 concat 만으로 '통합' 주장".

PCVRHyFormer baseline 의 `MultiSeqHyFormerBlock` 현재 구조:
1. Per-domain seq encoder (transformer/swiglu/longer)
2. Per-domain query decoder (cross-attention from Nq queries to seq)
3. **Token fusion = `RankMixerBlock`** (decoded_q + NS tokens 의 token-mixing via Linear)
4. Split back per-domain

Step 3 의 RankMixerBlock 이 현재 baseline 의 "feature interaction" 자리 — 단 token-mixing 형태로 explicit polynomial cross 아님. 본 H = step 3 swap: `RankMixerBlock` → **`DCNV2CrossBlock`** (Wang et al. WWW 2021 의 explicit polynomial cross with x₀ residual).

같은 위치, 같은 역할 (token fusion), 다른 mechanism (token-mixing → explicit polynomial cross). **block-level integration 보존** — seq decoded queries + NS interaction tokens 가 한 block 안 cross 에서 gradient 공유.

## Why now

- **H007 F-1 직접 후속**: target_attention (sequence axis) PASS marginal → orthogonal axis (interaction axis = explicit feature cross) 의 lift 검증.
- **§17.4 카테고리 rotation**: H007 = target_attention. **H008 = sparse_feature_cross 첫 적용** → rotation 추가 충족.
- **§17.2 single mutation 깔끔**: 한 클래스 swap (`RankMixerBlock` → `DCNV2CrossBlock`).
- **§0 anti-pattern 회피**: classifier head 직전 concat-late 가 아니라 **block-level fusion swap**. seq 와 int 가 같은 block 안 cross 에서 gradient 공유 → P1 조건 직접 충족.
- **§10.5 LayerNorm on x₀ MANDATORY** 룰이 정확히 DCN-V2 같은 cross stack 위해 만들어짐 — DCN-V2 블록에 Pre-LN x₀ 자동 충족.
- **paper grade**: DCN-V2 (Wang et al. WWW 2021) production CTR 표준 lever, low-rank cross 로 params 절약.
- **비용 cheap**: smoke envelope (1 ep × 5%) 가능, ~5분 wall. extended 는 PASS confirmed 후.

## Scope
- In:
  - 신규 클래스 `DCNV2CrossBlock` (model.py 확장, ~80줄):
    - Input: `(B, T, D)` — same as RankMixerBlock signature.
    - Output: `(B, T, D)`.
    - Mechanism: `x₀ ← Pre-LN(input)`, then stack `num_cross_layers` (default 2) of:
      `xₗ₊₁ = x₀ ⊙ (Uₗ Vₗᵀ xₗ + bₗ) + xₗ` (low-rank, rank=8 default).
  - `MultiSeqHyFormerBlock` 통합:
    - Constructor 인자: `fusion_type: str = 'rankmixer'` (or `'dcn_v2'`).
    - Step 3 fusion 모듈을 `RankMixerBlock` 또는 `DCNV2CrossBlock` 으로 dispatch.
  - PCVRHyFormer constructor 인자 추가: `fusion_type`, `dcn_v2_num_layers`, `dcn_v2_rank`.
  - CLI flags: `--fusion_type {rankmixer, dcn_v2}`, `--dcn_v2_num_layers 2`, `--dcn_v2_rank 8`.
  - infer.py: `cfg.get` read-back for new keys.
  - 그 외 모든 config: anchor 와 byte-identical envelope.
  - §18 인프라 룰 모두 inherit from original_baseline.
- Out:
  - DCN-V2 cross 의 layer 수, rank tuning — sub-H.
  - DCN-V2 외 다른 explicit cross (FwFM, AutoDis) — 별도 H.
  - 다른 위치 (encoder 직후, classifier 직전 등) 통합 — concat-late anti-pattern.
  - H007 의 candidate xattn 와 동시 적용 — 별도 combined H.

## UNI-REC axes
- **Sequential axis**: 변경 없음 — TransformerEncoder + 기존 query decoder 그대로.
- **Interaction axis**: RankMixer token-mixing → DCN-V2 explicit polynomial cross (degree 2-3). 본 H 의 mutation.
- **Bridging mechanism**: block-level fusion 단계에서 (decoded_q × S domains + NS tokens) 가 같은 cross block 통과 → seq 결과 + interaction tokens 한 block gradient 공유. **§0 P1 조건 직접 충족** (concat-late 아님).
- **primary_category**: `sparse_feature_cross` (§17.4 rotation 추가 충족).
- **Innovation axis**: token-mixing → explicit polynomial cross. RankMixer 의 implicit fusion 대신 DCN-V2 의 explicit polynomial cross.

## Success / Failure conditions
**§17.3 binary lift 임계 적용**:

- **Success**: Δ vs anchor (original_baseline) **platform AUC** ≥ **+0.5 pt**. **+ 4 부수 게이트**:
  1. Train 1 epoch NaN-free 완주.
  2. Inference: §18 인프라 통과 (batch heartbeat + `[infer] OK` 로그, no fallback).
  3. `metrics.json` 에 `{seed, git_sha, config_sha256, host, best_val_AUC, best_oof_AUC, fusion_type, dcn_v2_num_layers, dcn_v2_rank}` 모두 채워짐.
  4. infer.py 가 새 cfg key read-back → strict load 통과.
- **Failure**: Δ < +0.5pt → REFUTED. sparse_feature_cross 카테고리 일시 archive.

## Frozen facts referenced
- Anchor (original_baseline) Platform AUC: ~0.83X.
- H007 PASS marginal: Platform 0.8352. mechanism class 작동 confirmed.
- H006 verdict F-3: paired Δ는 platform AUC 으로만.
- H007 verdict F-2: val ↔ platform 정합.
- §10.5 LayerNorm on x₀ MANDATORY.
- §18 인프라 룰 (CLAUDE.md 신설 2026-04-28).

## Inheritance from prior H

- **H007 F-1**: target_attention mechanism PASS → orthogonal axis (interaction) 검증 차례.
- **H007 F-2**: val ↔ platform 정합 → 본 H 도 같은 패턴 expected.
- **H007 F-4**: extended envelope cost 압박 → 본 H smoke 우선.
- **H007 F-5**: §18 인프라 룰 inherit.
- **§10.5 룰 carry-forward**: DCN-V2 cross stack 의 직접 적용 영역, Pre-LN on x₀ 필수.
