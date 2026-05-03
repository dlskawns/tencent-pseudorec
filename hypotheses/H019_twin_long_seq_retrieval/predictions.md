# H019 — Predictions

> Paradigm shift first entry — `retrieval_long_seq` category first-touch.
> Paired Δ primary vs H010 corrected anchor (0.837806).

## Outcome distribution (post Recent Findings F-A through F-F)

| Outcome | Δ vs H010 corrected | probability | mechanism implication |
|---|---|---|---|
| strong | ≥ +0.005pt | ~30% | retrieval form 작동, L4 retrieval branch open, anchor = H019 |
| measurable | [+0.001, +0.005pt] | ~30% | retrieval 약 effect, sub-H 후보 |
| noise | (−0.001, +0.001pt] | ~25% | retrieval 도 cohort hard ceiling 못 풂, paradigm shift 다른 form 도 위험 |
| degraded | < −0.001pt | ~15% | TWIN 구현 detail 또는 sample-scale viability 문제 |

(H018 보다 strong probability 높게 잡음 — paradigm shift 가 다른 axis 직접 attack 이라 새로운 lift signal 가능성.)

## P1 — Code-path success
- Risk: GSU score shape mismatch (candidate broadcast). ESU top-K filter
  의 dynamic shape (per-batch K 다를 수 있음) → padding 처리.
- Failure mode: top-K filter NaN (모든 score 0) → 전체 mask zero → loss=0 → NaN.

## P2 — Primary lift (vs H010 corrected, §17.3 binary)

| 비교 | classification | Δ 임계 |
|---|---|---|
| Δ vs H010 corrected (0.837806) | strong | ≥ +0.005pt |
| Δ vs H010 corrected | measurable | [+0.001, +0.005pt] |
| Δ vs H010 corrected | noise | (−0.001, +0.001pt] |
| Δ vs H010 corrected | degraded | < −0.001pt |

**§17.3 binary cut**: Δ ≥ +0.5pt = strong PASS (sample-scale relaxed +0.005pt).

**Secondary**: Δ vs H015 corrected (0.83805) — H015 가 직전 측정 H 라 비교
의미 있음. Δ vs H015 ≥ +0.001pt → recency mechanism class 위 retrieval 추가
lift 신호.

## P3 — TWIN mechanism 작동 검증

- **GSU score distribution**:
  - mean / std / p10 / p50 / p90 print every 100 steps.
  - 만약 모두 ≈ uniform → GSU 가 candidate-relevance 학습 못 함.
  - 만약 highly concentrated (top-K 이 항상 같은 token) → ESU 가 dense
    attention 과 다름 없음 (retrieval 무력).
- **ESU attention entropy**:
  - threshold 0.95 × log(top-K=64) = 3.95 upper. lower bound 0.5.
  - entropy < 0.5 → ESU 도 sparse routing (degenerate).
  - entropy > 3.95 → ESU 도 uniform (dense fallback).
- **Top-K filter activity**:
  - 평균 top-K size (실제 attend) = 64 expected. < 32 → padding 과다.

## P4 — §18 인프라 PASS

- §18.5 (list type variants): seq_max_lens 변경 → make_schema.py 재생성
  필수. is_list / is_large_list / is_fixed_size_list 모두 check.
- §18.6 dataset-inference-auditor invocation: upload/ ready 직전 PASS.
  특히 schema regeneration 검증.
- §18.7 nullable to_numpy: H015 carry-forward.
- §18.8 emit_train_summary: train.py 마지막 SUMMARY 블록.

## P5 — val ↔ platform gap (H016 framework)

- F-A baseline: 4 H 누적 mean −0.003pt (val under platform).
- H019 expected: similar (−0.001 ~ −0.005pt).
- > +0.01pt 또는 < −0.01pt → val 신호 깨짐.

## P6 — OOF (redefined) ↔ Platform gap

- H016 redefined OOF gap baseline = −0.0036pt.
- H019 expected: similar (cohort handling 변경 안 함).
- > +0.01pt → cohort drift 다시 벌어짐 (retrieval 의 cohort 효과 부정적).

## P7 — Cost cap audit (§17.6)

- Pre-H019 누적: ~46h Taiji + H018 추가 ~3.5h = ~50h.
- H019 T3 ~$15 (per-job cap) + ~6h training (TWIN 의 추가 module 의 wall
  증가).
- **REFUTED 시 carry-forward decision**: H020 paradigm shift class 추가
  시도 cost cap (per-campaign ≤ $100) 압박 큼. H019 REFUTED → H020 보류
  결정 가능.

## Decision tree (post-result)

| Outcome | Δ vs H010 corrected | Action |
|---|---|---|
| **strong** Δ ≥ +0.005pt | retrieval 검증, mechanism class confirm | H020 = TWIN sub-H (top-K sweep, learnable GSU). anchor = H019. |
| **measurable** [+0.001, +0.005pt] | retrieval 약 effect | H020 = sub-H (top-K=128 / 32, GSU learnable variant). |
| **noise** (−0.001, +0.001pt] | retrieval 도 ceiling 못 풂 | H020 보류 (cost cap), ensemble / multi-seed measurement H 또는 P3 phase 대기. |
| **degraded** < −0.001pt | TWIN 구현 issue | H020 = TWIN sub-H (top-K=128, ESU 단순화). 또는 paradigm shift class 자체 retire. |
| **P5 fail** | val 신호 깨짐 | 다음 H decision platform paid 검증 필수. |
| **P6 fail** | cohort drift 다시 벌어짐 | retrieval 의 cohort 부정 효과. retire. |

## Falsification claim (반증 가능)

H019 의 mutation = **단 1개의 측정 가능한 claim**:
> "TWIN GSU+ESU per-domain retrieval (top-K=64 from cap=512) 가 H010 corrected anchor 위 Δ ≥ +0.005pt 추가 lift 만든다."

위 claim 이 거짓 → cohort drift 가설 강한 confirm + paradigm shift class
의 retrieval form 도 ceiling 못 풂 신호. H020 paradigm shift 시도 보류
결정 정당.

## Cost cap escape valve (§17.6 mandatory)

- H019 launch 직전 누적 cost 측정 (Taiji 가격 사용자 확인).
- 누적 + H019 estimate > $100 (per-campaign cap) 면 launch **중단** 사용자
  confirm 필수.
- Frame C (paradigm shift cost ratio 부적절) confirmed 시 H019 REFUTED 후
  ensemble / multi-seed measurement H 로 pivot.
