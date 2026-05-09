# H034 — Challengers

## §17.4 rotation 정당화 (4회 연속)

retrieval_long_seq 4회 연속 (H019/H020/H021/H034). H020/H021 의 4 사유 carry + **H034 추가 사유**:
- ESU capacity axis 가 retrieval 의 3번째 직교 axis (scoring quality + quantity + capacity). H020/H021 와 함께 mechanism class 의 모든 internal axis 검증.

---

## Frame A — "1-layer ESU 가 이미 sufficient (NOOP)"

**가설**: candidate Q (1 token) × top-K history KV (64 tokens) attention 에서 1-layer 가 충분. attention weight 이미 candidate-relevant token 에 집중 — 두 번째 layer 가 추가할 정보 없음.

**근거**:
- H010 NS→S xattn entropy 0.81 = highly selective routing → 1-layer attention 도 sparse 잘 작동.
- candidate 가 1 token 이라 query side complexity 매우 낮음 → multi-layer 의미 작음.

**Falsification 조건**: H034 Δ ∈ (-0.001, +0.001pt] (noise) → Frame A confirmed.

**Frame A confirmed 시 carry-forward**: retrieval class 의 모든 internal axis (scoring + capacity) saturated → cohort 또는 HSTU pivot 강제.

---

## Frame B — "2-layer 가 학습 instability (degraded)"

**가설**: small data (sample-scale + cohort drift ceiling) 에서 +64K params 가 over-capacity. 두 번째 layer 의 attention 이 noise 학습 → platform 일반화 악화.

**근거**: H013 (lr 8e-4) Keskar 패턴 — capacity 변경이 platform 악화. H011 (input-stage encoding) overfit signature 재발.

**Falsification 조건**: H034 Δ < -0.001pt (degraded) + overfit_gap > +0.005pt → Frame B confirmed.

**Frame B confirmed 시 carry-forward**: sub-H = ESU num_layers=2 + dropout 강화 (현재 dropout_rate=0.01 → 0.05) 또는 retire.

---

## Frame C — "Sub-H 진입이 시기상조 (H020/H021 결과 회수 전)"

**가설**: H020/H021 모두 PASS 시 H033 stacking 이 우선순위 → H034 cloud submit 보류 권장.

**근거**: H020/H021 PASS 시 scoring axis 가 lever 임 confirmed → capacity axis 검증 우선순위 낮음.

**Falsification 조건**: H020/H021 모두 noise → H034 우선순위 즉시 상승.

**Carry-forward**: H020/H021 결과별 H034 conditional submit 권장:
- 둘 다 noise → H034 즉시 cloud submit (capacity 가 마지막 hedge).
- 한 쪽 PASS → H034 우선순위 중간 (axis 독립성 검증 가치).
- 둘 다 PASS → H033 우선, H034 후순위.

---

## Counter-argument

1. **3번째 직교 axis 검증**: scoring quality (H020) + quantity (H021) + capacity (H034) = retrieval class 의 internal axis 종합.
2. **Pre-build cost-effective**: 학습 비용 conditional, build 비용 0.
3. **Frame A confirmed (NOOP) 도 가치 있음**: retrieval class 종료 결정의 final hedge.
