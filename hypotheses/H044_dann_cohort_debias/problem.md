# H044 — problem.md

## Trigger
H039 (vanilla baseline, mechanism-stripped) Platform 0.835426 vs H019 (full
stack) Platform 0.839674 = +0.004248pt mechanism contribution. **그러나
Val/OOF 는 ≈ tied** (Val Δ −0.00043, OOF Δ +0.00040). 14 H 동안 누적된
"F-A 패턴" (OOF over Platform) + H019/H038/H039 비교 → cohort drift 가
**platform transfer 실패의 근본 원인** 정량 confirm.

H038 (BCE+aux MSE on log1p(timestamp)) Platform 0.839071 vs H019 0.839674
= OOF +0.0012 → Platform −0.0006 (transfer fail). aux MSE 가 timestamp 를
*positive* 로 fit 하면 cohort overfit → platform transfer 실패. H044 =
opposite direction: **backbone 이 timestamp 를 *forget* 하도록 강제**.

## Data signal (CLAUDE.md §3.4 + §3.5 + F-A pattern)
- F-A 패턴 (9 H): OOF Platform 갭 1.88~2.59pt, mechanism class 무관 일관.
  cohort 분포가 OOF/Val 와 platform 사이 다름.
- timestamp 가 cohort label 의 가장 직접 proxy (label_time 기준 split).
- H038 의 aux MSE 가 backbone 에 timestamp signal 을 *학습시킴* → OOF cohort
  overfit. H044 = 역방향 (Ganin & Lempitsky 2015 DANN).

## Hypothesis
backbone 이 timestamp 를 예측 못 하도록 gradient reversal layer (GRL) 로
adversarial 학습 → cohort-invariant feature 강제 → platform transfer 향상.

## Mutation
- trainer.py 에 _GradReverse(autograd.Function) 추가.
- aux_timestamp branch 안 dann_lambda > 0 시 aux_pred 에 GRL 적용.
- run.sh: dann_lambda=0.5 (Ganin 권장 1.0 의 절반, sample-scale 안전).
- model.py / dataset.py / infer.py / utils.py byte-identical to H038.
- dann_lambda=0 시 H038 byte-identical (safe carrier).

## Falsifiable
Δ vs H038 Platform (0.839071) ≥ +0.001pt → DANN cohort debias 가 transfer 향상
lever. 더 결정적: Δ vs H019 Platform (0.839674) ≥ +0.001pt → cohort attack 이
H019 mechanism stack 위 추가 lift. 미달 시 timestamp 가 cohort proxy 로
부족 또는 GRL form 의 한계.
