# DLRM — Naumov et al. 2019

- **Title**: Deep Learning Recommendation Model for Personalization and
  Recommendation Systems.
- **Authors**: Maxim Naumov, Dheevatsa Mudigere, Hao-Jun Michael Shi, et al.
  (Meta / Facebook AI).
- **Venue**: arXiv preprint 2019, arXiv:1906.00091. Production system at
  Meta.
- **Read at**: 2026-04-30 (carry-forward from H011 scaffold).

## Core mechanism (1단락)

DLRM 은 sparse categorical features 를 embedding lookup 으로 D-dim dense
vector 로 변환 (`E[id]`), continuous numerical features 를 bottom MLP 로
같은 D-dim 로 변환한 후, 두 source 의 **모든 pair** 를 dot-product (또는
element-wise) 로 cross 항 만들어서 top MLP 에 입력. interaction layer 가
명시적으로 sparse-dense 상호작용을 학습. 핵심: **dense numerical feature 가
sparse embedding 과 같은 interaction stage 에서 first-class** — implicit
학습 부담 제거.

## Why this matters for TAAC 2026 UNI-REC

TAAC 2026 데이터: `user_int_feats` (46 sparse, 35 scalar + 11 array) 와
`user_dense_feats` (10 dense list<float>). aligned pair `{61–66, 89–91}` 는
같은 entity 의 ID 와 weight. DLRM 패턴 적용 = aligned fid 의 input embedding
lookup 과 dense weight 를 element-wise 결합 → interaction layer (DCN-V2,
NS xattn) 의 입력에 explicit alignment 제공.

현재 PCVRHyFormer baseline 은 user_int 를 RankMixerNSTokenizer 로,
user_dense 를 단일 Linear 로 분리 처리 (`model.py:1760-1788`) — DLRM 스타일
input fusion 미적용. **§4.8 mandate 위반 가능성**.

## Adoption notes for H011

- **Adopt**: aligned fid 의 input embedding × dense weight element-wise
  (DLRM 의 minimum viable form, parameter-free).
- **Modify**:
  - 모든 dense → MLP 가 아닌 aligned pair (9 fids) 만 weighted embedding.
  - dot-product cross 가 아닌 element-wise broadcast multiply (downstream
    NS xattn / DCN-V2 가 cross 처리).
  - bottom MLP 미적용 — raw weight 곱셈 후 NS tokenizer 로 직접.
- **Sample-scale viability**: parameter-free, params 추가 0. PASS.

## Reference key claims (paper figures)

- Figure 1: DLRM architecture. sparse embedding + dense MLP → interaction
  cross → top MLP.
- Section 3: interaction layer 의 explicit cross 가 implicit MLP 학습보다
  data-efficient (specifically for production-scale recommendation).

## Caveats

- DLRM 은 production scale (수백M users). sample-scale (1000 rows) 또는
  extended (30% × 10ep) 에서 lift 보장 안 함.
- DLRM 의 dense MLP 는 unbounded continuous features 처리. TAAC 2026 의
  user_dense_feats scale 분포 미검증 — H011 P1 (NaN check) 의 핵심 risk.
- DLRM 은 ID-weight 같은 같은 fid 안의 binding 이 아닌 field-pair cross.
  본 H 가 추출하는 부분 = "input stage 에서 dense 를 sparse embedding 과
  같은 단계에서 결합" 패턴만.
