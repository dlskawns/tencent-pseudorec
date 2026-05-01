# FwFM — Pan et al. WWW 2018

- **Title**: Field-weighted Factorization Machines for Click-Through Rate
  Prediction in Display Advertising.
- **Authors**: Junwei Pan, Jian Xu, Alfonso Lobos Ruiz, et al. (Yahoo /
  현재 Tencent AdsRec).
- **Venue**: WWW 2018, arXiv:1806.03514.
- **Read at**: 2026-04-30 (carry-forward from H011 scaffold).
- **Note**: Junwei Pan 은 TAAC 2026 organizer 중 한 명 (CLAUDE.md §0).
  본 paper 의 weight semantics 이 organizer 의 thinking 과 일치 가능.

## Core mechanism (1단락)

FwFM 은 standard FM 의 cross 항 `<v_i, v_j>` (feature i, feature j 의 latent
factor 내적) 에 **field-pair-specific scalar weight** `r_{F(i),F(j)}` 를
곱한다. F(·) 는 feature 의 field 매핑 (예: "user_age", "item_category",
"context_hour"). field-pair weight 는 학습 가능 scalar 로, field 간 importance
asymmetry 를 explicit 모델. 핵심: **field-level weight 가 cross 항의 중요도
스케일링** — 모든 field-pair 를 동일하게 다루는 standard FM 보다 expressive.

## Why this matters for TAAC 2026 UNI-REC

본 H (H011) 가 채택하는 부분 = **weight 가 cross 항의 scaling 으로 작동
하는 패턴**. 단 차이:
- FwFM: field i × field j (다른 field 간 cross) 에 weight.
- H011: 같은 fid 안 ID 와 weight 의 binding (intra-fid). 즉 fid k 의
  embedding 과 dense weight `w_k` 의 element-wise multiply.

FwFM 의 weight 는 학습 가능 scalar. H011 의 weight 는 데이터에 주어진
raw value (`user_dense_feats[k]`). 본 H 는 FwFM 의 단순 form — **데이터
가 weight 를 명시적 제공** 하므로 추가 학습 parameter 없이 적용 가능.

또한 Junwei Pan (FwFM 1저자, TAAC 2026 organizer) 의 thinking 이 데이터
schema 의 aligned `<id, weight>` 룰 (CLAUDE.md §3 / §4.8) 의 motivation
일 가능성 — **organizer 의 데이터 디자인 의도 자체가 본 H 의 mechanism**.

## Adoption notes for H011

- **Adopt**: weight 가 embedding cross 의 scaling 으로 작동하는 semantics.
- **Modify**:
  - inter-field cross 가 아닌 intra-fid ID-weight binding.
  - 학습 가능 scalar 가 아닌 raw data weight 사용.
  - FM 의 second-order cross 만 적용 안 함 — 본 H 는 input stage 에서 binding
    만, second-order 이상 cross 는 downstream (DCN-V2 H008) 처리.
- **Sample-scale viability**: parameter-free (FwFM 의 학습 가능 weight 미채택),
  params 추가 0. PASS.

## Reference key claims (paper figures)

- Section 4 (FwFM formulation): `y = w_0 + Σ x_i w_i + Σ_{i<j} r_{F(i),F(j)}
  <v_i, v_j> x_i x_j`. weight scalar 가 cross 항 앞에 곱해짐.
- Table 3: FwFM 이 standard FM, FFM 보다 production CTR 데이터에서 lift
  (Yahoo display ads).

## Caveats

- FwFM 은 학습 가능 weight 로 lift 의 일부가 capacity 효과. 본 H 의 raw
  data weight 는 capacity 추가 0 — lift 작을 가능성.
- field 정의 자체가 잘못되면 weight 잘못 학습. H011 은 fid-level grouping
  사용 (CLAUDE.md §3 의 aligned pair 후보) — audit (P0) 통과 필수.
- FwFM 의 second-order cross 자체는 본 H 의 영역 아님 (DCN-V2 H008 anchor 에서
  처리). 본 H 는 **input stage binding** 의 좁은 추출.
