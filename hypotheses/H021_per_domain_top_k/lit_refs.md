# H021 — Literature References

## Primary

- **Chang, J. et al. 2024** — "TWIN" RecSys 2024. (Tencent — 대회 organizer)
  - paper top-K = 128 (single user history L=10K+). per-domain K policy 는 paper-uncovered extension.
  - 본 H = 4 도메인 분할 환경의 K policy mutation.
- **H019 (twin_long_seq_retrieval)** — sub-H base. uniform top_k=64. cloud measurable PASS (Δ vs H010 corrected +0.00187pt).

## Secondary

- **Pi, Q. et al. 2020** — "SIM" KDD 2020. soft search + hard search 의 K 가 search type 별 다름 — per-search-policy K 의 reference.
- **§3.5 데이터 facts** (CLAUDE.md) — domain별 p90 seq length 분포 (a:1562 / b:1393 / c:887 / d:2215). 본 H 의 quantitative motivation 1차 source.
- **`tencent-cc/eda/out/semantics.json:per_feature_seq_length`** — 같은 1000-row flat snapshot 검증. 본 H 인용 chain.

## Carry-forward refs (from H019/H020)

- **Chen, Q. et al. 2021** — ETA. Hash-based GSU — H022′ 후보.
- **Zhai, J. et al. 2024** — HSTU. Trunk replacement — H020/H021 모두 noise 시 paradigm shift 후보.
- **Kang, W.C. & McAuley, J. 2018** — SASRec. dense form baseline.
- **Zhou, G. et al. 2018** — DIN. Historical behavior weighted by attention.

## H020 와의 직교성 reference

- H020 = scoring function quality axis (parameter-free → learnable projection).
- H021 = scoring quantity axis (uniform K → per-domain K).
- 두 axis 직교 → 동시 검증 + paired 비교 framework (predictions.md §H020 와의 paired 비교).

## OneTrans (대회 organizer paper)

- **OneTrans (Tencent, WWW 2026, arXiv:2510.26104)** — H019 별도 mechanism class. H020/H021 모두 noise 시 paradigm shift 후보.

## What's NOT a clone

- 본 H 는 **TWIN paper 의 1:1 재현 아님**:
  - paper single history. 본 H 4 도메인 분할 history.
  - paper top-K = 128. 본 H per-domain {a:64, b:64, c:64, d:96} — domain d 만 50% 확장 (uniform K=128 sweep flat 결과 의 75% conservative end).
  - paper K policy = uniform across single search type. 본 H K policy = domain length 분포 인지.
  - paper 100M+ user. 본 H demo_1000 sample-scale sanity + cloud full-data 측정.

## H010~H020 carry-forward refs

- **H010 (NS→S xattn) PASS additive** — anchor 0.837806 corrected.
- **H011~H013 REFUTED** — input/MoE/hyperparameter mutations on H010.
- **H014 REFUTED** (L4 dense form retire) — uniform seq expansion 한계.
- **H015 marginal +0.0002pt** — temporal_cohort first attempt.
- **H016 model REFUTED / infra PASS** — OOF redefine future_only framework.
- **H017 INVALID** — submission lost.
- **H018 SCAFFOLDED** — temporal_cohort 4th sibling.
- **H019 cloud measurable PASS** — TWIN paradigm shift first entry. retrieval class lever confirmed.
- **H020 SCAFFOLDED 2026-05-06** — TWIN sub-H scoring axis (learnable GSU). H021 과 직교 sub-H.

H021 는 H019 mechanism class 안 quantity axis 1단계 깊이 — H020 과 직교 axis, 동시 검증.
