# H022 — Literature References

## Primary

- **Bouthillier, X. et al. 2021** — "Accounting for Variance in Machine
  Learning Benchmarks." MLSys 2021. ML benchmark 의 seed variance
  정량화 표준. 5-10 seeds 권장. **본 H 의 1차 source**.

- **Reimers, N. & Gurevych, I. 2017** — "Reporting Score Distributions
  Makes a Difference: Performance Study of LSTM-networks for Sequence
  Tagging." EMNLP 2017. NLP 의 single-seed reporting 문제 직접 지적.
  multiple runs 의 score distribution 표시 권장.

- **Madhyastha, P. & Jain, R. 2019** — "On Model Stability as a Function
  of Random Seed." CoNLL 2019. seed-level variance 가 model selection
  결정에 미치는 영향.

## Secondary

- **Efron, B. 1979** — "Bootstrap Methods: Another Look at the Jackknife."
  Annals of Statistics 7(1). paired bootstrap CI 이론. H022 sub-H 후보
  (prediction-level resample).

- **Henderson, P. et al. 2018** — "Deep Reinforcement Learning that
  Matters." AAAI 2018. RL 의 reproducibility crisis. 5-30 seed 권장.

- **Lucic, M. et al. 2018** — "Are GANs Created Equal? A Large-Scale
  Study." NeurIPS 2018. GAN benchmark 의 single-seed 가정 무효 입증.

- **Pham, H. et al. 2021** — "Combined Scaling for Open-Vocabulary Image
  Classification." 모델 stability 측정 표준 patterns.

## Statistics references

- **Welford, B.P. 1962** — "Note on a Method for Calculating Corrected Sums
  of Squares and Products." Technometrics. unbiased online variance
  estimator. n=3 의 sample stdev 산출법.

- **Student (Gosset, W.S.) 1908** — "The Probable Error of a Mean."
  Biometrika. t-distribution 기반 small-sample CI.

## H010 baseline (control)

- H010 verdict.md: Platform AUC 0.8408 (uncorrected) → 0.837806 (corrected
  eval data, 2026-05-02~03 organizer fix). NS-token bidirectional xattn,
  H008 anchor 위 stacking. attn entropy [0.81, 0.81].
- H010 envelope: 10 epochs × 30% train_ratio × patience=3.
- H010 train wall: 3시간 44분 54초 (seed 42, single run).
- H010 = 본 H 의 mechanism (byte-identical 3 회 학습).

## What's NOT a clone

- 본 H 는 **Bouthillier et al. 의 1:1 재현 아님**:
  - paper 5-10 seeds. 본 H 3 seeds (cost trade-off, minimum viable).
  - paper benchmark suite (multiple datasets). 본 H single dataset
    (TAAC 2026 demo_1000 + cloud full-data).
  - paper paired bootstrap + ANOVA. 본 H mean ± stdev 만 (sub-H 로
    bootstrap).
  - paper publication-grade rigor. 본 H = decision-grade minimum
    (variance threshold 결정 위해 σ 측정 만 충분).
