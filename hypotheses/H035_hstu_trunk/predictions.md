# H035 — Predictions

> Paradigm shift sub-H — per-domain seq encoder transformer → HSTU.
> Paired Δ primary vs H019 (cloud actual 0.839674).
> §17.3 binary cut at paradigm shift first-class entry 임계 (+0.005pt strong).

## Outcome distribution (post data_ratio=1 ceiling signal + 12 H ceiling)

| Outcome | Δ vs H019 | probability | mechanism implication |
|---|---|---|---|
| strong | ≥ +0.005pt | ~20% | mechanism class 전환이 진짜 lever, anchor = H035, 0.85 도달 가능성 |
| measurable | [+0.001, +0.005pt] | ~25% | 약 effect, sub-H 후보 (HSTU full form, multi-layer) |
| noise | (-0.001, +0.001pt] | ~35% | mechanism class 도 못 풂, cohort drift 가 진짜 hard ceiling, H036 강제 |
| degraded | < -0.001pt | ~20% | HSTU 본 데이터 부적합 (sample-scale + sparse classification), Frame B confirmed |

(noise probability 가장 큼 — data_ratio=1 결과 + 12 H ceiling 의 cohort drift 가설이 강함. 단 paradigm shift first hedge 라 strong probability 도 무시 안 함.)

## P1 — Code-path success

- Risk: HSTU forward 의 silu(scores) 가 negative score 영역에서 안정적인지. softmax 와 달리 normalization 보장 안 됨 (length norm /L 만).
- Failure mode: silu(score) overflow (huge positive scores) → output norm explosion. 또는 padded key 의 0-fill 후 silu(0)=0 이지만 dropout × L division 후 numerical underflow.
- Mitigation: T0 sanity 에서 forward NaN-free 검증 완료. attention entropy diagnostic carry (softmax 가 아니라 정보용).

## P2 — Primary lift (vs H019, §17.3 binary)

| 비교 | classification | Δ 임계 |
|---|---|---|
| Δ vs H019 (0.839674) | strong | ≥ +0.005pt (paradigm shift first-class) |
| Δ vs H019 | measurable | [+0.001, +0.005pt] |
| Δ vs H019 | noise | (-0.001, +0.001pt] |
| Δ vs H019 | degraded | < -0.001pt |

**Secondary**:
- Δ vs H010 corrected (0.837806) = +0.001868 + H035 Δ. ≥ +0.007pt → 0.85 까지 +0.005pt 만 남음.
- 0.85 target gap = 0.85 - 0.839674 = +0.0103pt. H035 strong (+0.005pt) 만으로는 미달, H035 + ensemble + cohort 조합 필요.

## P3 — HSTU mechanism 작동 검증

- **Attention shape distribution** (silu 의 분포):
  - silu(score) 의 mean / std / 음수 비율 측정.
  - 음수 비율 > 50% → silu 가 sparse signal 못 capture (Frame B 신호).
  - mean ≈ 0 → length norm 후 effective contribution 작음.
- **U gate distribution** (gate 의 selectivity):
  - U projection output 의 magnitude 분포. 거의 0 → gate 가 information 차단 (mechanism 무력).
  - 거의 1 → gate 가 identity, gate 의미 없음.
  - 적정 (0.3 ~ 0.7) → multiplicative gate 가 token-level filtering 역할.
- **Per-layer training stability**:
  - epoch trajectory (best_val) 이 monotonic 인가. softmax 보다 unstable 가능성.
  - overfit_gap (best_val − last_val) > +0.005pt 면 Frame B 또는 C 신호.

## P4 — §18 인프라 PASS

- §18.5: schema 변경 없음.
- §18.6 dataset-inference-auditor: model.py 의 HSTUEncoder class 추가 + create_sequence_encoder dispatch 변경. **state dict key 변경** (transformer → hstu) → infer.py 가 cfg 기반 모델 재구성 (`seq_encoder_type` cfg 기록) → 정상 (H019 carry 패턴). audit 범위 = model.py + train.py + run.sh.
- §18.7/§18.8: H019 carry.

## P5 — val ↔ platform gap (H016 framework)

- F-A baseline: 4 H 누적 mean −0.003pt. H035 expected: similar.
- > +0.01pt 또는 < −0.01pt → val 신호 깨짐.

## P6 — OOF (redefined) ↔ Platform gap

- H016 redefined OOF gap baseline = −0.0036pt.
- H035 expected: similar (cohort handling 변경 안 함, mechanism class 만 변경).
- > +0.01pt → cohort drift 다시 벌어짐 (HSTU 도 cohort 못 풂 강한 신호).

## P7 — Cost cap audit (§17.6)

- Pre-H035 누적: H019/H020/H021/H033 cloud + 이전 ~50h ≈ ~64h.
- H035 T2.4 ~3.5h × $5-7. cumulative cost cap 친화.
- **REFUTED 시 carry-forward decision**:
  - noise → H036_cohort_embed 즉시 (cohort drift 가 last hope).
  - degraded → HSTU full form sub-H 또는 OneTrans full backbone.
- HSTU 가 unstable 해서 epoch 수 늘면 cost 증가 가능 — 사용자 confirm threshold 5h.

## Decision tree (post-result)

| Outcome | Δ vs H019 | Action |
|---|---|---|
| **strong** Δ ≥ +0.005pt | mechanism class lever 진짜 | anchor = H035. sub-H = HSTU full form (relative attention bias 추가) 또는 HSTU + per-domain top_k stack. 0.85 도달까지 +0.005pt 가 ensemble + cohort 로 가능. |
| **measurable** [+0.001, +0.005pt] | 약 effect | sub-H = HSTU multi-layer (현재 2-layer → 3-layer per hyformer block). 또는 HSTU + H020 stack. |
| **noise** (-0.001, +0.001pt] | mechanism class 도 dead | cohort drift (L2) 가 진짜 hard ceiling 강한 confirm. H036_cohort_embed 강제. retrieval / backbone class 모두 retire. |
| **degraded** < -0.001pt | HSTU 본 데이터 부적합 (Frame B) | sub-H = HSTU full form (RAB 추가) 또는 OneTrans full backbone. 또는 retire. |
| **P5 fail** | val 신호 깨짐 | platform paid 검증 mandatory. |
| **P6 fail** | cohort drift 다시 벌어짐 | HSTU 도 cohort 못 풂. retire. |

## Falsification claim (반증 가능)

H035 의 mutation = **단 1개의 측정 가능한 claim**:
> "per-domain seq encoder 를 transformer (softmax MHA + FFN) → HSTU (silu-attention + gated linear unit) 으로 변경하는 것이 H019 anchor 위 Δ ≥ +0.005pt 추가 lift 만든다."

거짓 → mechanism class 도 ceiling 못 풂 → cohort drift (L2) 가 진짜 hard ceiling 강한 confirm. backbone_replacement class 의 다른 paper-form (OneTrans full) 시도도 우선순위 낮춤.

## Cost cap escape valve (§17.6 mandatory)

- H035 launch 직전 누적 cost 측정 (Subset A actual + H019/H020/H021/H033 cloud actual).
- 누적 + H035 estimate > $80 (per-campaign cap $100 80%) 면 사용자 confirm.
- HSTU 가 unstable 해서 epoch 수 5+ 시 사용자 abort 결정 (early termination 권장).
