# H022 — Method Transfer

## ① Source

- **Bouthillier, X. et al. 2021** — "Accounting for Variance in Machine
  Learning Benchmarks." MLSys 2021. **본 H 의 1차 source**. ML
  benchmark 의 seed variance 정량화 표준. 5-10 seeds 권장 (본 H 는 cost
  trade-off 로 3 seeds minimum viable form).
- **Efron, B. 1979** — "Bootstrap Methods: Another Look at the
  Jackknife." Annals of Statistics. paired bootstrap CI 의 이론적 근거.
  본 H 는 multi-seed mean ± stdev 만 (sub-H 로 paired bootstrap).
- **Reimers, N. & Gurevych, I. 2017** — "Reporting Score Distributions
  Makes a Difference: Performance Study of LSTM-networks for Sequence
  Tagging." EMNLP. NLP 의 single-seed reporting 문제 직접 지적.
- **Madhyastha, P. & Jain, R. 2019** — "On Model Stability as a Function
  of Random Seed." CoNLL. seed-level variance 가 model selection 결정에
  영향.

## ② Original mechanism

**Multi-seed variance estimation** (1단락 재서술):

같은 model, 같은 data, 같은 envelope, **seed 만 변경** (e.g., 42 → 42, 43,
44) 으로 N 회 학습. Platform AUC 의 mean ± stdev 산출. stdev 가 paired Δ
threshold 의 lower bound 결정 (σ × 2 < |Δ| → signal, ≥ → noise band 안).

**우리 적용**:
- H010 (corrected anchor 0.837806) 의 mechanism + envelope byte-identical.
- seed 42 (already done) + seed 43, 44 추가 학습.
- 3 platform AUC values → mean (~0.838) ± stdev.
- variance threshold 결정 → 9 H 의 historical Δ 들 retroactive 재분류.

## ③ What we adopt

- **Mechanism class**: pure measurement, **NO model mutation**.
- **변경 내용 (1 file + run.sh)**:
  1. `train.py`: `--seed <int>` 가 이미 argparse 에 있음 (검증). 변경
     없음.
  2. `run.sh`: 3 회 launch (seed 43, 44 추가 — seed 42 는 H010 result
     재사용). 또는 wrapper script 생성.
- **CLI**: `python train.py --seed 43` / `--seed 44` (separate runs).

## ④ What we modify (NOT a clone of paper)

- **3 seeds (paper 5-10)**: cost-conservative. Frame A 우려 인정.
  PASS strong σ tight 시 3 sufficient confirmed. PASS moderate σ 시
  H022-sub = +2 seeds.
- **Mean ± stdev (not paired bootstrap CI)**: 본 H minimum viable form.
  paired bootstrap 은 sub-H (prediction-level resample, more code).
- **Same envelope (10ep × 30% × patience=3)**: H010 envelope 그대로. seed
  외 변동 0.
- **§17.2 exempt rationale**: H022 = measurement H, mutation 없음.
  mutation-vs-mutation paired Δ 비교 framework 결정용. §17.2 "한 component
  교체" rule 적용 안 됨.

## ⑤ UNI-REC alignment

- **Sequential reference**: H010 stack 그대로 (NS xattn + per-domain
  encoder).
- **Interaction reference**: H010 stack 그대로 (DCN-V2).
- **Bridging mechanism**: 변경 없음.
- **Training procedure**: seed 만 변경.
- **primary_category**: `measurement` (NEW first-touch).
- **Innovation axis**: measurement infrastructure 강화. 모든 paired Δ 의
  statistical foundation. UNI-REC 의 mechanism comparison framework 정확도
  증진.

## ⑥ Sample-scale viability (Rule UB-1, §10.6)

- 추가 trainable params: **0** (mechanism unchanged).
- §10.6 cap: 위반 없음.
- Sample-scale risk:
  - **3 seeds × sample-scale**: 1000-row sample 에서 seed 변동이
    full-data variance 와 다를 수 있음. **Cloud full-data 측정 의무**.
    sample-scale = code-path verification only (§17.5).
  - **Seed reproducibility**: torch.manual_seed + numpy seed + cuda
    deterministic 확인 의무 (H010 train.py 검증 후 carry-forward).

## ⑦ Carry-forward rules to honor

- **§10.5 LayerNorm on x₀**: 변경 없음 (H010 stack 그대로).
- **§10.6 sample budget cap**: 변경 없음.
- **§10.7 카테고리 rotation**: `measurement` first-touch. rotation
  auto-justified.
- **§10.9 OneTrans softmax-attention entropy**: H010 baseline [0.81, 0.81]
  3 seeds 전부 측정 → variance 기록. **추가 데이터: entropy variance**
  도 부산물.
- **§10.10 InterFormer bridge gating σ(−2)**: 미적용.
- **§17.1 anchor-first**: H010 = current corrected anchor. H022 가 anchor
  의 variance 정량화 — anchor 자체 검증.
- **§17.2 one-mutation**: **EXEMPT** — measurement H, no mutation.
  challengers.md ④ 정당화 명시.
- **§17.3 binary success**: σ thresholds tight/moderate/large per
  predictions.md.
- **§17.4 rotation**: auto-justified.
- **§17.5 sample-scale = code-path verification only**.
- **§17.6 cost cap**: 3 × T2.4 ~$5 = $15. campaign cap $100 안.
- **§18.6 dataset-inference-auditor**: H010 upload/ 3 회 재 launch — 이미
  검증된 package, auditor 재 invoke 선택 (mechanism unchanged).
- **§18.7 nullable to_numpy**: H010 already H015 patch carry-forward
  적용됐으면 OK. 미적용 시 patch 후 launch.
- **§18.8 emit_train_summary**: H010 train.py 에 §18.8 SUMMARY block
  emit 추가 의무 (3 seeds 모두 SUMMARY 회수). 이게 H010 patch 의 단일
  변경점.
