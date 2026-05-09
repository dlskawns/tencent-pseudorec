# H031 — Predictions

> Pre-registered before run.

## P0 — Audit gate (pre-train)

- **Quantity 1**: `item_int_feats_13` schema vs == 10 (확인: `data/schema.json` `item_int` array, fid=13 entry).
- **Quantity 2**: 5 user_int fids 각각 schema vs (1=6, 97=5, 58=4, 49=4, 95=5) — total cross matrix ≤ 24 cells.
- **Quantity 3**: `model.py` 의 `RankMixerNSTokenizer` 가 fid=13 / fid={1,97,58,49,95} 제외 후에도 input shape consistent.
- Predicted: PASS — schema 확인 + 코드 수정 후 sanity test.
- Falsification: schema vs ≠ 10 → fid 매핑 재확인. P0 fail → INVALID.

## P1 — Code-path success

- Quantity: T1 1-batch forward NaN-free + 5-step train loss decreasing + cloud train 완주 (`metrics.json` 생성).
- Predicted: PASS. 새 Embedding 5개 + Linear 1개 추가만 — H010 backbone 그대로.
- Falsification: NaN abort → cross_token init scale issue (Xavier 또는 LayerNorm 추가).

## P2 — Primary lift (§17.3 binary)

- **Quantity**: extended envelope **platform AUC** vs control (TBD: H023 corrected baseline 또는 H010 corrected anchor 0.837806).

| Δ vs control | classification | mechanism interpretation |
|---|---|---|
| ≥ +0.005pt | **strong PASS** | fid-level explicit cross 추출 작동, dilute 가설 confirm |
| [+0.001, +0.005pt] | **measurable** | 작동 marginal, sub-H (item_9/16 추가, user_int 확대) 정당화 |
| (−0.001, +0.001pt) | **noise** | NS-tokenizer 이미 implicit 학습, fid-level cross redundant. **Frame B** |
| < −0.001pt | **degraded** | cross-token 이 DCN-V2 noise 증폭. mechanism class retire 검토 |

## P3 — Mechanism check: cross-token activation magnitude

- Quantity: `cross_token.norm(dim=-1).mean()` — 학습 진행 중 logging.
- Predicted classifications:
  - **active** (norm > 0.5 × mean(NS-tokens norm)): cross 가 의미 있는 signal 학습.
  - **inactive** (norm < 0.1 × mean(NS-tokens norm)): cross 가 학습 못 함. P2 noise 와 일관 → mechanism dispatch fail.
- Falsification 아님 — diagnostic.

## P4 — §18 인프라 통과

- Quantity: inference 시 §18 rules 충족 (batch heartbeat + `[infer] OK: torch path produced N predictions`).
- Predicted: PASS (H010 패키지 inherit + extra cfg key 1-2개).
- Falsification: §18 회귀 → infer.py cfg.get 추가 부분 디버깅.

## P5 — val ↔ platform gap (보너스)

- Quantity: `best_val_AUC − platform_AUC`.
- Predicted: ∈ [−0.005, +0.005pt] (H010 ~−0.003pt 패턴).
- Falsification 아님.

## P6 — OOF (redefined future-only) ↔ platform gap

- Quantity: redefined OOF AUC − platform AUC.
- Predicted: ∈ [−0.005, +0.005pt] (H016 framework, redefined OOF 가 platform 분포에 align 검증됨).
- Falsification 아님.

## P7 — verify-claim §18.8 SUMMARY parser dry-run

- Quantity: train.py 끝 SUMMARY block 정규식 파싱 PASS.
- Predicted: PASS (H018 framework inherit).

## Reproducibility

- compute_tier: T2.4 extended (10ep × 30%, batch=1024 OOM safety, patience=3) ~3-4시간.
- seed: 42 (H028 결과 보고 multi-seed 결정 — 만약 σ_val > 0.005pt 면 H031 도 multi-seed 재측정).
- split: label_time + 10% OOF (anchor 동일).
- expected wall: H010 (3.7h) 동급 또는 약간 길어짐 (forward path 1 cross-token 추가).

## Negative-result interpretation (§17.7 falsification-first)

본 H REFUTED 시:

- **P2 noise** (Δ ∈ (−0.001, +0.001pt)): NS-tokenizer 가 이미 item_13 implicit 학습 충분. **Frame B 채택** → fid-level cross retire. carry-forward: H032 (timestamp features) 우선, item-side mechanism 미진입.
- **P2 degraded** (Δ < −0.001pt): cross-token 이 DCN-V2 input distribution 흔듦. retract — sub-H = LayerNorm 추가 또는 cross-token 별도 path (DCN-V2 입력 합류 X, prediction head 직접 합류).
- **P2 measurable** (+0.001~+0.005pt): partial PASS — sub-H (item_9 추가, user_int_54 등 high-card cross) 진행 가능. anchor 갱신 보류.
- **P3 inactive + P2 noise**: cross-token gradient flow 막힘. init scale 또는 LayerNorm 누락 검토 후 sub-H.

## Decision tree (post-result)

| Result | Next action |
|---|---|
| Δ ≥ +0.005pt + P0/P1/P3/P4 PASS | **strong PASS — anchor 갱신**: H031 = 새 baseline. H032 timestamp stack 또는 item_9/16 추가. |
| Δ ∈ [+0.001, +0.005pt] | measurable PASS. anchor 갱신 검토. sub-H = item_9 추가. |
| Δ ∈ (−0.001, +0.001pt) | noise. anchor 유지. **Frame B confirm** — fid-level cross retire. H032 우선. |
| Δ < −0.001pt | degraded. retract — sub-H (LayerNorm, separate prediction-head path). |
| P0 fail | INVALID. schema 재확인. |
| P3 inactive | mechanism dispatch issue. init scale / gradient flow 디버깅 후 sub-H. |
