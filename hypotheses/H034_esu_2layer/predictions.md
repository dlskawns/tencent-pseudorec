# H034 — Predictions

> Sub-H — ESU 1-layer → 2-layer (capacity axis). control = H019.
> H020/H021 와 직교 axis. conditional cloud submit (H020/H021 결과 의존).

## Outcome distribution

| Outcome | Δ vs H019 | probability | mechanism implication |
|---|---|---|---|
| strong | ≥ +0.003pt | ~20% | capacity bottleneck confirmed, anchor = H034 |
| measurable | [+0.001, +0.003pt] | ~25% | 약 effect, sub-H 후보 |
| noise | (-0.001, +0.001pt] | ~40% | 1-layer ESU sufficient, capacity axis dead |
| degraded | < -0.001pt | ~15% | over-capacity (small data instability) |

(noise probability 가장 큼 — candidate Q=1 token 의 단순함 때문에 1-layer 가 충분일 가능성 큼.)

## P1 — Code-path success
- Risk: ModuleList iteration order, intermediate LayerNorm shape mismatch.
- Mitigation: T0 sanity (1/2/3-layer 모두 NaN-free 검증 완료).

## P2 — Primary lift (vs H019, §17.3 binary)

| 비교 | classification | Δ 임계 |
|---|---|---|
| Δ vs H019 (0.839674) | strong | ≥ +0.003pt |
| Δ vs H019 | measurable | [+0.001, +0.003pt] |
| Δ vs H019 | noise | (-0.001, +0.001pt] |
| Δ vs H019 | degraded | < -0.001pt |

**Secondary**:
- Δ vs H010 corrected (0.837806) = +0.001868 + H034 Δ.
- Triple-paired: H020 + H021 + H034 → 3 axis 직교성 검증.

## P3 — ESU multi-layer mechanism 검증

- **Per-layer attention entropy**: 두 layer 모두 measurement. threshold 0.95 × log(K=64) = 3.95 upper.
  - 두 layer 모두 entropy > 3.95 → ESU 가 dense fallback (mechanism 무력).
  - 첫 layer entropy 낮고 두 번째 high → 첫 layer 가 sparse pick, 두 번째 가 noise mix (degraded 신호).
  - 두 layer 모두 적정 → multi-layer routing 학습.
- **Layer-wise gradient norm**: 두 번째 layer grad 가 첫 layer 의 1/10 미만 → 학습 안 됨 (NOOP 신호).

## P4 — §18 인프라 PASS

- §18.5: schema 변경 없음.
- §18.6 dataset-inference-auditor: TWINBlock 구조 변경 (state dict key 추가) → infer.py 의 ckpt loading 영향. 단 infer.py 가 cfg 기반 모델 재구성 → `twin_esu_num_layers` cfg 기록 시 정상 (H019 carry 패턴).
- §18.7/§18.8: H019 carry.

## P5/P6 — val/OOF ↔ Platform gap
- F-A baseline: −0.003pt. H016 redefined OOF baseline: −0.0036pt.

## P7 — Cost cap audit
- T2.4 ~3.5h × $5-7. campaign cap $100 친화.
- **Conditional submit**: H020/H021 결과별 우선순위:
  - 둘 다 noise → H034 즉시 cloud submit (capacity 가 마지막 hedge).
  - 한 쪽 PASS → H034 우선순위 중간.
  - 둘 다 PASS → H033 우선.

## Decision tree (post-result)

| Outcome | 다음 액션 |
|---|---|
| strong | anchor = H034, sub-H = num_layers=3 또는 H020+H034 stacking |
| measurable | sub-H 후보, H020/H021 와 paired stack 검증 |
| noise | 1-layer ESU sufficient, retrieval class capacity 도 saturation. ESU axis retire. cohort/HSTU pivot |
| degraded | over-capacity. sub-H = dropout 강화 또는 retire |

## Falsification claim

> "TWIN ESU 의 1-layer → 2-layer (intermediate residual + LayerNorm) 가 H019 anchor 위 Δ ≥ +0.003pt 추가 lift 만든다."

거짓 → 1-layer ESU sufficient → retrieval class 의 capacity axis dead → cohort 또는 HSTU pivot 필수.

## Cost cap escape valve

H020/H021 cloud actual cost 회수 후 결정. 누적 + H034 estimate > $80 면 사용자 confirm.
