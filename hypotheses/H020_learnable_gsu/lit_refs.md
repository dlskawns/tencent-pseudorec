# H020 — Literature References

## Primary

- **Chang, J. et al. 2024** — "TWIN: TWo-stage Interest Network for Long-term User Behavior Modeling." RecSys 2024. (Tencent — 대회 organizer 와 같은 회사)
  - Paper 의 GSU = lightweight learnable scorer. Q/K projection + inner product.
  - paper 100M+ user / 1B+ event 환경 검증.
  - **본 H 의 1차 source — paper-faithful GSU 직접 검증** (H019 의 simplified parameter-free 와 차이).

- **H019 (twin_long_seq_retrieval)** — sub-H base. parameter-free inner product GSU 의 learnable 변환. cloud-ready (`bash run.sh --seed 42`).

## Secondary

- **Pi, Q. et al. 2020** — "Search-based User Interest Modeling with Lifelong Sequential Behavior Data" (SIM). KDD 2020. soft search = embedding similarity 형태 — H020 의 reference.
- **Vaswani, A. et al. 2017** — "Attention is All You Need". Multi-head attention 의 Q/K projection convention 의 origin.
- **Devlin, J. et al. 2019** — "BERT: Pre-training of Deep Bidirectional Transformers". Q/K projection dim 결정의 standard practice (d_model // num_heads).

## Carry-forward refs (from H019)

- **Chen, Q. et al. 2021** — ETA. Hash-based GSU — H020 PASS 후 H022′ 후보.
- **Zhai, J. et al. 2024** — HSTU. Trunk replacement — H020 noise 시 paradigm shift 후보.
- **Kang, W.C. & McAuley, J. 2018** — SASRec. dense form baseline.
- **Zhou, G. et al. 2018** — DIN. Historical behavior weighted by attention.

## OneTrans (대회 organizer paper)

- **OneTrans (Tencent, WWW 2026, arXiv:2510.26104)** — H019 별도 mechanism class. H020 noise 시 H022′ 후보.

## What's NOT a clone

- 본 H 는 **TWIN paper 의 1:1 재현 아님**:
  - paper d_proj = d_model 또는 d_model//2. 본 H d_proj = d_model//4 (16) — §10.6 sample budget 친화 + Frame B 우려 인지.
  - paper bias=Yes 가능성. 본 H bias=False — magnitude 는 embedding norm 으로 충분.
  - paper single long history. 본 H 4 도메인 분할 history (H019 carry).
  - paper 100M+ user. 본 H demo_1000 sample-scale sanity + cloud full-data 측정.

## H010~H019 carry-forward refs

- **H010 (NS→S xattn) PASS additive** — anchor 0.837806 corrected. H019 + H020 의 carry-forward base.
- **H011~H013 REFUTED** — input/MoE/hyperparameter mutations on H010.
- **H014 REFUTED** (L4 dense form retire) — retrieval form 의 paper-faithful 검증 필요성 강화.
- **H015 marginal +0.0002pt** — temporal_cohort first attempt.
- **H016 model REFUTED / infra PASS** — OOF redefine future_only framework.
- **H017 INVALID** — submission lost.
- **H018 SCAFFOLDED** — temporal_cohort 4th sibling.
- **H019 BUILT** — TWIN paradigm shift first entry. simplified GSU + ESU per-domain. cloud-ready, sweep saturation 확인 후 H020 sub-H 진입.

H020 는 H019 mechanism class 안 1단계 깊이 — 위 carry-forward 들의 retrieval scoring axis 검증.
