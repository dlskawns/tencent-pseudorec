# H042 — problem.md

## Trigger
14 H 모두 sample-level loss (BCE/BPR/focal). H038 가 처음으로 output-distribution
level supervision (aux MSE on log1p(timestamp)) 시도 → OOF Δ vs H019 +0.00120pt
+ LogLoss −0.0013 = small but real lever in NEW axis. 사용자 §0.5 의 shift-scheduling
KLD pattern 을 본 데이터에 적용.

## Data signal (CLAUDE.md §3.4)
- Class prior 12.4% positive (label_type=2: 124/1000).
- 14 H 모두 BCE / focal (sample-level), prior 와 *output prob distribution*
  의 일치는 한 번도 explicit constraint 으로 안 들어감.

## Hypothesis
H038 의 aux MSE 가 *간접* output supervision (auxiliary regression) 였다면,
KLD prior matching 은 *직접* output prob distribution attack. per-sample
Bernoulli KL(σ(logit) ‖ Bern(0.124)) 가 overconfident prediction 을 prior
쪽으로 regularize → label smoothing-like 일반화 + H038 의 aux signal 와
super-additive 가능성.

## Mutation
- trainer.py 'aux_timestamp' branch 에 KLD term 추가:
  `loss += λ_kld · mean_i KL(Bern(σ(logit_i)) ‖ Bern(prior))`
- λ_kld = 0.01 (보수적, T0 KLD ≈ BCE 의 1% 기여).
- prior = 0.124 (§3.4).
- kld_lambda=0 시 H038 byte-identical (safe carrier).

## Falsifiable
Δ vs H038 OOF (0.8623) ≥ +0.001pt → output-distribution axis 가 main lever
(H038 단독 +0.0012 + KLD 추가 +0.001 = 누적 +0.002 vs H019). 미달 시 KLD
가 MSE 와 redundant 또는 λ 너무 작음 (sub-H λ sweep).
