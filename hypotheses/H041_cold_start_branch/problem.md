# H041 — problem.md

## Trigger
F-G ceiling 0.832~0.836 (12 H mechanism mutation invariant) + 14 H paper-transplant
antipattern 사용자 지적 (CLAUDE.md §0.5 추가). data-signal-driven 으로 axis 재발견.

## Data signal (CLAUDE.md §3.5)
- `target_item_in_domain_seq.any_domain = 0.4%` (4 / 1000 rows).
- 즉 99.6% prediction 이 user 가 본 적 없는 item 추천 = extrapolation regime.

## Hypothesis
14 H 모두 single classifier 로 두 regime (familiar 0.4% vs novel 99.6%) 동시 처리.
cold-start case (novel) 에 capacity 부족 → dual classifier (main + cold) +
per-sample learned gate 로 specialize 시 lift.

## Mutation
- `cold_clsfier`: nn.Sequential(Linear→LN→SiLU→Dropout→Linear) — main 과 동일 shape.
- `cold_gate`: nn.Linear(d_model, 1) — per-sample scalar gate, init bias=2.0.
- forward dispatch: `gate * main + (1-gate) * cold`. init gate ≈ sigmoid(2.0) ≈ 0.88.

## Falsifiable
Δ vs H019 (Val 0.8372 / OOF 0.8611 / Platform 0.839674) ≥ +0.002pt → cold-start
specialization 이 lever. 미달 시 R2 reframe 결함 (data signal mapping wrong).
