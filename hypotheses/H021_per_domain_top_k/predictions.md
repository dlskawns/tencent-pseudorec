# H021 — Predictions

> TWIN sub-H — top_k policy 의 uniform → domain-aware (single-domain change: d 만 64→96).
> Paired Δ primary vs H019 (cloud actual 0.839674).
> H020 (scoring axis) 과 직교 sub-H — 동시 scaffold + cloud submit.

## Outcome distribution (post H019 sweep saturation + §3.5 분석)

| Outcome | Δ vs H019 | probability | mechanism implication |
|---|---|---|---|
| strong | ≥ +0.003pt | ~20% | per-domain K 가 진짜 lever, domain d under-collecting confirmed, H022 정량안 stack |
| measurable | [+0.001, +0.003pt] | ~30% | 약 effect, H020 결과와 paired stack 가능성 |
| noise | (−0.001, +0.001pt] | ~40% | uniform K=64 가 globally saturated, per-domain 도 NOOP, retrieval quantity axis dead |
| degraded | < −0.001pt | ~10% | K=96 noise injection 또는 inter-domain instability, sub-H 또는 retire |

(H020 보다 strong probability 약간 낮음 — uniform K sweep 에서 K↑ flat 결과가 직접 신호. per-domain 변경 으로 lift 만들려면 domain d 가 *특별히* under-served 라는 가정 필요.)

## P1 — Code-path success

- Risk: PCVRHyFormer 의 `twin_top_k` argument type 변경 (int | dict) — backward compat 깨짐 위험. argparse 의 string 파싱 정합성.
- Failure mode: comma-separated string 파싱 실패 (`twin_top_k_per_domain="64,64,64,96"` → 4-int list 변환). 길이 ≠ 4 시 assertion fail.
- Mitigation: T0 sanity 에서 4 도메인 TWINBlock instantiation 정상 (top_k 값 출력 확인) + domain d K=96 forward shape 정상.

## P2 — Primary lift (vs H019, §17.3 binary)

| 비교 | classification | Δ 임계 |
|---|---|---|
| Δ vs H019 (0.839674) | strong | ≥ +0.003pt |
| Δ vs H019 | measurable | [+0.001, +0.003pt] |
| Δ vs H019 | noise | (−0.001, +0.001pt] |
| Δ vs H019 | degraded | < −0.001pt |

**§17.3 binary cut**: H020 동일 임계 (sub-H 보수적 cut). paradigm shift first entry (H019 +0.005) 보다 낮음.

**Secondary**:
- Δ vs H010 corrected (0.837806) = H019 cloud Δ (+0.00187) + H021 의 Δ. carry-forward 합산 신호.
- Δ vs H010 corrected ≥ +0.005pt → retrieval mechanism class 영구 confirm + quantity axis lever 검증.
- **Triple-paired**: H019 vs H020 vs H021 — 3 H 의 Δ 를 비교해서 어느 axis 가 진짜 lever 인지 isolation.

## P3 — Per-domain K mechanism 작동 검증

- **Per-domain TWINBlock instantiation 확인**:
  - log line: `H021 TWIN per-domain K: a=64 b=64 c=64 d=96`.
  - 4 TWINBlock 각각의 `self.top_k` 값 print.
- **ESU attention entropy per-domain** (H019 와 비교):
  - threshold per-domain: 0.95 × log(K). K=64 → 3.95 / K=96 → 4.34.
  - domain d entropy 가 H019 의 d entropy 보다 증가 → 추가 retrieved token 들이 attention 흡수 (mechanism 작동).
  - domain d entropy 동일 → ESU 가 추가 token 무시 (Frame A NOOP).
- **GSU score distribution per-domain**:
  - domain d 의 top-K=96 score 분포 (mean / std / p10 / p90) 측정.
  - top-K=64 (H019) 와 비교: top 64-96 슬롯 의 score 가 top 0-64 보다 얼마나 낮은지 → noise injection 정도 정량.
- **Top-K filter activity per-domain**: 평균 attend size = K (domain별). domain d 의 평균 < 96 → padding 과다 (history length < 96 user 비율 측정).

## P4 — §18 인프라 PASS

- §18.5 (list type variants): seq_max_lens 변경 없음 → make_schema.py 그대로.
- §18.6 dataset-inference-auditor invocation: upload/ ready 직전 PASS. **dataset.py / infer.py / make_schema.py / TWINBlock class 변경 없음** → audit 범위 좁음 (PCVRHyFormer wiring + train.py + run.sh).
- §18.7 nullable to_numpy: H015 carry-forward (변경 없음).
- §18.8 emit_train_summary: H019 의 train.py 그대로 carry, exp_id 만 H021 로 변경.

## P5 — val ↔ platform gap (H016 framework)

- F-A baseline: 4 H 누적 mean −0.003pt (val under platform).
- H021 expected: similar (−0.001 ~ −0.005pt).
- > +0.01pt 또는 < −0.01pt → val 신호 깨짐.

## P6 — OOF (redefined) ↔ Platform gap

- H016 redefined OOF gap baseline = −0.0036pt.
- H021 expected: similar (cohort handling 변경 없음).
- > +0.01pt → cohort drift 다시 벌어짐 (per-domain K 의 cohort 효과 부정적).

## P7 — Cost cap audit (§17.6)

- Pre-H021 누적: H019 cloud 3.5h + H020 cloud 3.5h + 이전 ~50h = ~57h.
- H021 T2.4 ~3.5h × $5-7. cumulative cost cap 친화.
- **REFUTED 시 carry-forward decision**: noise → ESU axis (A3) 또는 cohort attack (C). degraded → K=80 (gentle) 또는 retire.

## Decision tree (post-result)

| Outcome | Δ vs H019 | Action |
|---|---|---|
| **strong** Δ ≥ +0.003pt | per-domain K lift 검증, mechanism class lever confirm | H022 = 정량안 (a:96 d:128) sub-H. anchor = H021. retrieval quantity axis 영구 confirm. H020 결과와 paired 비교 → 둘 다 PASS 면 H023 stack. |
| **measurable** [+0.001, +0.003pt] | 약 effect | H020 결과와 paired 비교. 둘 다 measurable+ 면 H022 = stack (H020 + H021). 한 쪽만 measurable+ 면 그쪽 anchor. |
| **noise** (−0.001, +0.001pt] | quantity axis NOOP | retrieval class 의 quantity axis dead (Frame A confirmed). uniform K + per-domain K 모두 saturation → ESU 또는 cohort pivot. |
| **degraded** < −0.001pt | K=96 issue | sub-H = K=80 (Frame B 보수안) 또는 다른 도메인 (a:96, b:64, c:64, d:64) test 또는 retire. |
| **P5 fail** | val 신호 깨짐 | 다음 H decision platform paid 검증 필수. |
| **P6 fail** | cohort drift 다시 벌어짐 | per-domain K 의 cohort 부정 효과. retire. |

## Falsification claim (반증 가능)

H021 의 mutation = **단 1개의 측정 가능한 claim**:
> "TWIN top_k policy 의 uniform=64 → per-domain {a:64, b:64, c:64, d:96} (domain d 만 50% 확장) 가 H019 anchor 위 Δ ≥ +0.003pt 추가 lift 만든다."

위 claim 이 거짓 → uniform K=64 가 모든 도메인에서 effective limit (Frame A 또는 B confirmed) → retrieval class 의 quantity axis dead. ESU capacity (A3) 또는 cohort drift (C) pivot 결정.

## Cost cap escape valve (§17.6 mandatory)

- H021 launch 직전 누적 cost 측정 (H019/H020 cloud actual + Subset A H028~H031 actual).
- 누적 + H021 estimate > $80 (per-campaign cap $100 80%) 면 launch 사용자 confirm 필수.
- H021 noise + H020 noise 동시 발생 시 retrieval class 전체 saturation confirmed → H022′ (ESU) 또는 H023′ (cohort) cost ratio 평가 후 결정.

## H020 와의 paired 비교 framework

- H020 (scoring axis) 와 H021 (quantity axis) 동시 cloud submit, 결과 회수 후 paired 비교:
  - 둘 다 PASS measurable+ → H022 = stack (H020 + H021 단일 H 안 두 axis 동시 mutation).
  - H020 only PASS → anchor = H020, H021 결과 = quantity axis NOOP signal.
  - H021 only PASS → anchor = H021, H020 결과 = scoring axis NOOP signal.
  - 둘 다 noise → retrieval class 전체 saturation, ESU 또는 cohort pivot.
