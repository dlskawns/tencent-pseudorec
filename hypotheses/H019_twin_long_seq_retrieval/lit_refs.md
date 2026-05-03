# H019 — Literature References

## Primary

- **Chang, J. et al. 2024** — "TWIN: TWo-stage Interest Network for
  Long-term User Behavior Modeling." RecSys 2024. (Tencent — 대회
  organizer 와 같은 회사) GSU lightweight scoring + top-K + ESU full
  attention. 100M+ user / 1B+ event 환경 검증.
  → 본 H 의 1차 source. minimum viable form.

- **Pi, Q. et al. 2020** — "Search-based User Interest Modeling with
  Lifelong Sequential Behavior Data" (SIM). KDD 2020. TWIN 의 prior —
  hard search (category match) + soft search (embedding similarity).
  → fallback reference. TWIN 이 SIM 의 superset 이라 본 H 우선.

## Secondary

- **Zhai, J. et al. 2024** — "Actions Speak Louder than Words: Trillion-
  Parameter Sequential Transducers for Generative Recommendations" (HSTU).
  ICML 2024. (Meta) Trunk replacement form, generative recommender.
  → H020 후보 reference (H019 REFUTED 시 paradigm shift 다른 form).

- **Chen, Q. et al. 2021** — "End-to-End User Behavior Retrieval in
  Click-Through Rate Prediction Model" (ETA). Alibaba production. Hash
  table-based GSU.
  → TWIN 보다 더 lightweight, sub-H 후보.

- **Kang, W.C. & McAuley, J. 2018** — "Self-Attentive Sequential
  Recommendation" (SASRec). Sequential rec backbone, dense form.
  → 본 H 의 baseline (TWIN 이 dense 대체 형태).

- **Sun, F. et al. 2019** — "BERT4Rec: Sequential Recommendation with
  Bidirectional Encoder Representations from Transformer."
  → 다른 dense form variant.

- **Zhou, G. et al. 2018** — "Deep Interest Network for Click-Through
  Rate Prediction" (DIN). 사용자 historical behavior weighted by
  attention. TWIN ESU 의 짧은 history 형태.

- **Zhou, G. et al. 2019** — "Deep Interest Evolution Network" (DIEN).
  GRU + attention over history.

## OneTrans (대회 organizer paper)

- **OneTrans (Tencent, WWW 2026, arXiv:2510.26104)** — S-token + NS-token
  single-stream transformer + mixed causal mask + pyramid pruning.
  → CLAUDE.md §0 backbone reference. H019 와 별도 mechanism class
  (backbone replacement vs sequence axis retrieval). H019 결과 따라 H020
  후보 결정.

## What's NOT a clone

- 본 H 는 **TWIN paper 의 1:1 재현 아님**:
  - paper = 100M+ user / 1B+ event. 본 H = demo_1000 sample-scale sanity +
    cloud full-data 측정.
  - paper top-K=128 / L=10K. 본 H top-K=64 / cap=512 (12% ratio,
    conservative).
  - paper GSU = learnable scorer + 별도 embedding table. 본 H GSU = simple
    inner product (parameter-free, §10.6 friendly).
  - paper ESU = customized multi-head attention. 본 H ESU = standard
    MultiHeadAttention (encoder reuse).
  - paper = single long history. 본 H = 4 도메인 분할 history (per-domain
    GSU+ESU).
  - paper = e-commerce CTR. 본 H = TAAC 2026 UNI-REC (cross-domain user
    intent).

## H010~H018 carry-forward refs

- **H010 (NS→S xattn) PASS additive** — anchor 0.837806 corrected.
- **H011~H013 REFUTED** — input/MoE/hyperparameter mutations on H010.
- **H014 REFUTED** (L4 dense form retire) — retrieval form 별도 (본 H).
- **H015 marginal +0.0002pt** — temporal_cohort first attempt, recency loss.
- **H016 model REFUTED / infra PASS** — OOF redefine future_only framework.
- **H017 INVALID** — submission lost.
- **H018 SCAFFOLDED** — temporal_cohort 4th sibling, per-user recency.

H019 는 첫 paradigm shift entry — 위 carry-forward 들의 cohort drift
hard ceiling 가설 검증 의무.
