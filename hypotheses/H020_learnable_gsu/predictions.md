# H020 — Predictions

> TWIN sub-H — GSU scoring function 의 parameter-free → learnable projection.
> Paired Δ primary vs H019 (champion). H019 cloud result 회수 후 H010 corrected 와 secondary 비교.

## Outcome distribution (post H019 sweep saturation analysis)

| Outcome | Δ vs H019 | probability | mechanism implication |
|---|---|---|---|
| strong | ≥ +0.003pt | ~25% | learnable scoring 이 retrieval policy lift, sub-H = H021 stack |
| measurable | [+0.001, +0.003pt] | ~30% | scoring axis 약 effect, A2 stack 가능성 검증 |
| noise | (−0.001, +0.001pt] | ~30% | scoring 이 inner product 와 차이 없음 (NOOP), retrieval selection axis retire |
| degraded | < −0.001pt | ~15% | projection rank reduction 또는 학습 instability, sub-H = dim 확장 |

(H019 보다 strong probability 낮게 잡음 — sub-H 깊이 들어가는 변경은 paradigm shift first entry 보다 marginal probability 높음.)

## P1 — Code-path success

- Risk: W_q / W_k forward NaN (init 직후 score 계산에서 magnitude collapse). projection dim d_model//4 가 작으면 score variance 작아져서 top_k filter 의 -inf masking 후 모든 score ≈ 0 가능성.
- Failure mode: top_k filter 의 첫 step 모든 score 0 → ESU MultiheadAttention NaN (H019 의 defensive guard `all_padded` 가 잡지 못하는 case).
- Mitigation: T0 sanity 에서 GSU score variance 측정 (per-domain mean / std / p10 / p90).

## P2 — Primary lift (vs H019, §17.3 binary)

| 비교 | classification | Δ 임계 |
|---|---|---|
| Δ vs H019 | strong | ≥ +0.003pt |
| Δ vs H019 | measurable | [+0.001, +0.003pt] |
| Δ vs H019 | noise | (−0.001, +0.001pt] |
| Δ vs H019 | degraded | < −0.001pt |

**§17.3 binary cut**: Δ ≥ +0.5pt = strong PASS (sample-scale relaxed +0.003pt — sub-H 깊이 들어가는 변경의 보수적 임계, paradigm shift first entry 의 +0.005pt 보다 낮음).

**Secondary**:
- Δ vs H010 corrected (0.837806) = H019 의 Δ + H020 의 Δ. carry-forward 합산 신호.
- Δ vs H019 ≥ +0.003pt + Δ vs H010 ≥ +0.005pt → retrieval mechanism class 영구 confirm + scoring axis lever 검증.

## P3 — Learnable GSU mechanism 작동 검증

- **GSU score distribution** (H019 와 비교):
  - mean / std / p10 / p50 / p90 print every 100 steps.
  - 만약 H019 와 거의 동일 → projection 이 identity 로 수렴 (Frame A confirmed, NOOP).
  - 만약 std 더 큼 (score 가 더 polarized) → scoring 이 더 selective.
  - 만약 std 더 작음 (score collapse) → projection rank reduction issue (Frame B).
- **W_q / W_k weight norm**:
  - 학습 후 frobenius norm 측정. ≈ 0 → identity collapse. ≫ 1 → over-amplification.
  - 적정 범위 추정: 0.5 ~ 2.0 (Xavier init 기준).
- **ESU attention entropy** (H019 와 비교):
  - threshold 0.95 × log(top-K=64) = 3.95 upper, 0.5 lower (H019 carry).
  - H019 과 entropy 거의 동일 → scoring 변화 없음 (Frame A).
  - entropy 증가 (uniform 방향) → ESU 가 더 dense attend → top-K 안 token 들이 비슷한 quality (scoring 효과 약함).
  - entropy 감소 (sharp 방향) → ESU 가 더 sparse attend → scoring 이 더 잘 select.
- **Top-K filter activity**: 평균 attend size = 64 expected. H019 과 동일.

## P4 — §18 인프라 PASS

- §18.5 (list type variants): seq_max_lens 변경 없음 → make_schema.py 그대로.
- §18.6 dataset-inference-auditor invocation: upload/ ready 직전 PASS. **dataset.py / infer.py / make_schema.py 변경 없음** → audit 범위는 model.py + train.py 만.
- §18.7 nullable to_numpy: H015 carry-forward (변경 없음).
- §18.8 emit_train_summary: H019 의 train.py 그대로 carry, exp_id 만 변경.

## P5 — val ↔ platform gap (H016 framework)

- F-A baseline: 4 H 누적 mean −0.003pt (val under platform).
- H020 expected: similar (−0.001 ~ −0.005pt).
- > +0.01pt 또는 < −0.01pt → val 신호 깨짐.

## P6 — OOF (redefined) ↔ Platform gap

- H016 redefined OOF gap baseline = −0.0036pt.
- H020 expected: similar (cohort handling 변경 없음).
- > +0.01pt → cohort drift 다시 벌어짐 (scoring 의 cohort 효과 부정적).

## P7 — Cost cap audit (§17.6)

- Pre-H020 누적: H019 ~3.5h + 이전 누적 ~50h = ~53.5h.
- H020 T2.4 ~3.5h × $5-7. cumulative cost cap 친화.
- **REFUTED 시 carry-forward decision**: noise → ESU axis (A3) 또는 cohort attack (C). degraded → projection dim 확장 또는 retire.

## Decision tree (post-result)

| Outcome | Δ vs H019 | Action |
|---|---|---|
| **strong** Δ ≥ +0.003pt | scoring lift confirm, mechanism class lever | H021 = per-domain top_k stack on H020 base. anchor = H020. retrieval scoring axis 영구 confirm. |
| **measurable** [+0.001, +0.003pt] | scoring 약 effect | H021 단독 (per-domain top_k on H019 base) → H020 vs H021 paired 비교 후 stack 결정. |
| **noise** (−0.001, +0.001pt] | scoring 무 effect (NOOP) | retrieval selection axis retire. H022′ = ESU 2-layer (A3) 또는 cohort attack (C). |
| **degraded** < −0.001pt | projection issue | sub-H = projection dim d_model//2 또는 W_k only (W_q identity). 또는 retire. |
| **P5 fail** | val 신호 깨짐 | 다음 H decision platform paid 검증 필수. |
| **P6 fail** | cohort drift 다시 벌어짐 | scoring 의 cohort 부정 효과. retire. |

## Falsification claim (반증 가능)

H020 의 mutation = **단 1개의 측정 가능한 claim**:
> "TWIN GSU 의 parameter-free inner product → learnable projection (W_q, W_k: nn.Linear(d_model, d_model//4)) 가 H019 anchor 위 Δ ≥ +0.003pt 추가 lift 만든다."

위 claim 이 거짓 → backbone embedding space 가 이미 retrieval scoring 에 충분 → retrieval class 의 selection policy axis retire. H022′ = ESU capacity (A3) 또는 cohort drift (C) pivot 결정.

## Cost cap escape valve (§17.6 mandatory)

- H020 launch 직전 누적 cost 측정 (H019 cloud cost actual + Subset A H028~H031 actual).
- 누적 + H020 estimate > $80 (per-campaign cap $100 80%) 면 launch 사용자 confirm 필수.
- H020 noise 시 H022′ (ESU 또는 cohort) cost ratio 평가 후 진행 결정.

## H019 cloud result (Frame C — RESOLVED)

- **H019 cloud measurable PASS CONFIRMED** (platform 0.839674, Δ vs H010 corrected +0.001868pt = §17.3 measurable band).
- Frame C REFUTED → paradigm shift family ceiling-breaker confirmed → H020 launch 정당.
- H020 upload BUILT 2026-05-06, cloud submit ready (`bash run.sh --seed 42`).
