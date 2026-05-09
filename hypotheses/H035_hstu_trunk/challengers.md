# H035 — Challengers (≥ 2 reverse frames)

## §17.4 rotation 정당화 (NEW category first-touch)

**카테고리**: `backbone_replacement` — NEW first-touch (auto-justified).

**§10.7 audit**: retrieval_long_seq 4회 연속 (H019/H020/H021/H034) 후의 강제 rotation. 미경험 카테고리 (`backbone_replacement`, `debiasing`) 중 backbone 우선 — paradigm shift first-class entry 의 strongest hedge.

---

## Frame A — "HSTU 도 cohort drift 못 풂 (NOOP)"

**가설**: H019 data_ratio=1 → eval 0.837785 의 진짜 root cause 가 cohort drift (later data 의 distribution shift) 이면, mechanism class 변경 (transformer → HSTU) 도 cohort handle 못 함. capacity-axis 차이 일 뿐 distribution-axis 효과 없음.

**근거**:
- H014 F-2: OOF-Platform gap 2.59pt = 9 H 중 가장 큼 (cohort drift 강한 signal).
- 12 H 누적 ceiling 0.832~0.836 — mechanism 다양 (input / MoE / hyperparam / recency / DCN-V2 / OneTrans / per-user) 무관.
- HSTU paper 의 lift = scale × data 환경 (트릴리언 params + 1B+ events). 본 sample-scale 에서 paper-claim 일반화 불확실.

**Falsification 조건**: H035 Δ vs H019 ∈ (-0.001, +0.001pt] (noise) → Frame A confirmed.

**Frame A confirmed 시 carry-forward**: backbone_replacement class 도 ceiling 못 풂 → cohort drift (L2) 가 진짜 hard ceiling → H036_cohort_embed 강제 attack.

---

## Frame B — "HSTU 가 sample-scale 에서 작동 안 함 (degraded)"

**가설**: HSTU paper 의 핵심 lift = trillion-param scale + autoregressive generation task. 본 환경 = 161M params + binary classification. paper formulation 의 silu-attention 이 본 데이터의 sparse classification signal 에 못 fit → softmax (transformer) 보다 underperform.

**근거**:
- 본 데이터 conversion rate 12.4% — sparse positive class. softmax-attention 의 sparse routing 이 이런 sparse signal 에 잘 작동 (H010 F-3 entropy 0.81 = highly selective).
- HSTU 의 silu-attention = pointwise dense attention → sparse signal 흐릴 가능성.
- Params 절반 (HSTU 20K vs Transformer 54K per layer) → capacity 부족 가능.

**Falsification 조건**: H035 Δ vs H019 < -0.001pt (degraded) + best_val 큰 폭 하락 → Frame B confirmed.

**Frame B confirmed 시 carry-forward**: HSTU 본 데이터 부적합. sub-H = HSTU full form (relative attention bias 추가) 또는 retire. OneTrans full backbone 후순위 candidate.

---

## Frame C — "Trunk 만 변경 = unfaithful 재현 (interference 위험)"

**가설**: HSTU paper 는 *전체 trunk* 가 HSTU layer stack. 본 H = per-domain seq encoder 만 HSTU, NS xattn / DCN-V2 fusion 은 standard transformer. 두 다른 mechanism 이 같은 forward path 에 → H009 패턴 (interference) 위험.

**근거**:
- H009 (combined xattn + DCN-V2) REFUTED — combined < strongest single. block-level fusion 위치 충돌.
- HSTU output (silu-gated) → NS xattn (softmax) → DCN-V2 (interaction cross). 분포 mismatch 가능.
- 깔끔한 paradigm shift = 모든 transformer block 을 HSTU 로 swap (NS xattn 포함). 본 H = partial swap.

**Falsification 조건**: best_val 의 epoch trajectory 가 unstable + overfit_gap > +0.005pt → Frame C confirmed.

**Frame C confirmed 시 carry-forward**: H036_hstu_full = NS xattn + 모든 attention 을 HSTU 로 통일. 더 큰 mutation 이지만 paper-faithful.

---

## Counter-argument 종합

1. **Paradigm shift 의 first hedge**: data_ratio=1 ceiling 신호 + retrieval class 4 H 누적 = mechanism class 변경 필수. HSTU = paper-faithful, cost-effective (T2.4).
2. **Falsification value 대형**: 4 outcome 모두 결정적 신호 — strong → mechanism class lever 영구 confirm / noise → cohort drift 강제 attack / degraded → backbone family 일반화 한계.
3. **Cost-effective vs OneTrans full**: T2.4 ~$5-7 < OneTrans full backbone $15+. 첫 backbone test 로 합리.
4. **Single mutation 깔끔**: per-domain seq encoder 의 type 1 변경. 다른 모든 부분 byte-identical to H019.
