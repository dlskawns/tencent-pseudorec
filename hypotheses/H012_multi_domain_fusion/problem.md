# H012 — Problem Statement

## What we're trying to explain

현재 baseline (PCVRHyFormer + H010 NS xattn + H008 DCN-V2 anchor) 은 4 도메인
(a/b/c/d) 의 시퀀스를 **per-domain encoder → S tokens concat → 단일 NS xattn
+ DCN-V2 fusion** 으로 처리. **도메인별 expert specialization 없음** —
모든 도메인 signal 이 같은 fusion block 으로 균일 처리.

데이터 사실 (eda 측정 결과):
- **도메인별 vocab 거의 disjoint** (jaccard a_vs_c=0.007, a_vs_b=0.079,
  최대 a_vs_d=0.100): 4 도메인이 사실상 다른 universe 의 item 들.
- **도메인별 length 분포 큰 차이** (§3.5): p50 a=577 / b=405 / c=322 / d=1035,
  frac_empty a=0.5% / d=8% 까지.
- **target item 이 user seq 에 거의 안 등장** (target_item_in_domain_seq:
  any_domain=0.4%): cross-domain candidate-history matching 가설 약함.

**측정 가능한 gap**: 단일 fusion block 이 4 도메인의 다른 분포 (vocab,
length, empty ratio) 를 강제로 균일 처리 → expert routing (MMoE/PLE) 으로
도메인별 specialization 가능 시 lift 가능성.

## Why now

직전 H 들의 carry-forward:
- H010 (target_attention) PASS — NS xattn 이 dominant signal selective
  routing 학습 (entropy 0.81). 그러나 H010 의 routing 이 implicit 도메인
  routing 을 학습 중일 가능성 (4 도메인 S tokens concat → NS-tokens 가
  ~2 tokens 만 attend → 도메인 specific token 위주).
- H011 (feature_engineering) REFUTED — input-stage modify 가 H010 의
  selective routing 흡수, marginal 효과. **explicit routing (H012) 이
  implicit 한 H010 routing 보다 강할 수 있음**.
- §0 north star: sequence axis × interaction axis 통합. multi_domain_fusion
  은 sequence-side fusion 강화 — orthogonal axis (interaction encoding 아닌
  sequence routing).

§10.7 rotation: 직전 2 H = H010 target_attention + H011 feature_engineering.
H012 차단 = feature_engineering. **multi_domain_fusion = 신규 카테고리
first-touch (FREE)**.

## Scope

- **In**:
  - 4 도메인 (a/b/c/d) S tokens 의 explicit expert routing.
  - MMoE (Multi-gate Mixture-of-Experts) 또는 PLE (Progressive Layered
    Extraction) minimum viable form.
  - 통합 위치: H010 NS xattn 출력 직후, DCN-V2 fusion 전. NS xattn / DCN-V2
    입력 텐서 변경 안 함 (H011 F-1 carry-forward — input-stage modify 가
    cohort drift 악화시키는 패턴 회피).
  - expert per domain (= 4 experts), 도메인별 gate (per query/NS-token 또는
    domain-fixed).
- **Out**:
  - sample-scale 한계 — 다수의 expert (예: 8+) 는 §10.6 budget 초과 위험.
    minimum viable form = 4 experts (도메인 = expert 1:1 매핑) 또는 ≤ 2
    expert with shared trunk (PLE pattern).
  - cross-domain attention 변형 (별도 H — H007 sub-H 에 가까움).
  - hard routing (top-K) 변형 (PLE 의 progressive 이외 별도 sub-H).

## UNI-REC axes

- **Sequential**: 4 도메인 per-domain encoder + NS xattn (H010) 출력 → MMoE/PLE
  expert routing 추가 → enriched representation. sequence-side specialization.
- **Interaction**: DCN-V2 (H008 anchor) 그대로. expert routing 출력 이
  DCN-V2 입력으로 들어가 polynomial cross 처리 — explicit interaction
  axis 보존.
- **Bridging mechanism**: per-domain S tokens → NS xattn (cross-domain
  alignment) → MMoE/PLE expert (per-domain specialization 명시) → DCN-V2
  (interaction cross). expert routing 가 sequence-side fusion 의 새 layer
  → §0 P1 ("seq + interaction 같은 block gradient 공유") 의 sequence
  specialization 강화 form.

## Success / Failure conditions

- **Success (PASS)**:
  - §17.3 binary: Δ vs anchor (H010 0.8408) ≥ +0.001pt (relaxed measurable
    threshold at sample-scale) → Platform ≥ 0.8418.
  - 또는 strong: Δ ≥ +0.005pt → Platform ≥ 0.8458.
  - mechanism check (P3): expert utilization 이 균등하지 않음 (도메인별
    specialization 신호) — sample-scale 에서 entropy/utilization 측정 가능.
- **Failure (REFUTED)**:
  - degraded: Δ < −0.001pt → REFUTED. expert routing 이 H010 의 implicit
    routing 보다 못함.
  - noise: Δ ∈ (−0.001, +0.001pt] → noise. H010 가 이미 implicit 도메인
    routing 학습 중 가설 (Frame B 강화).
  - expert collapse: gate 가 1-2 expert 에만 routing → §10.9 룰 abort 신호.

## Frozen facts referenced

- CLAUDE.md §3.5 — domain seq length p50/p90/max/frac_empty per domain.
- CLAUDE.md §3.5 (this turn 갱신) — domain Jaccard overlap (a_vs_c=0.007 ~
  a_vs_d=0.100), target_item_in_domain_seq (any=0.4%).
- `eda/out/domain_facts.json` — 정량 출처.
- §10.6 sample budget cap (≤ 2146 trainable params for new mutation).
- §10.7 rotation — multi_domain_fusion FREE first-touch.
- §17.3 binary — Δ ≥ +0.001pt measurable threshold (sample-scale 한계 인정).
- H010 F-1: NS-only enrichment safe pattern (downstream byte-identical).
- H011 F-1: input-stage modify cohort drift 위험 → H012 는 NS xattn 출력
  단계 stacking, anchor 입력 byte-identical.
- H011 F-5: cohort drift hard ceiling 가능성 — H012 결과 cohort gap 모니터.

## Inheritance from prior H

- H010 anchor (Platform 0.8408) — control, byte-identical 유지 (NS xattn +
  DCN-V2).
- H011 F-2 (NS xattn entropy 거의 변화 없음) → H012 의 expert routing 도
  H010 selective routing 위에서 보완적 가설. expert utilization 측정 의무.
- H010 F-3 (selective routing 가 dominant signal 학습) → H012 expert 가
  도메인 specialization 으로 그 routing 보완 가능 가설.
