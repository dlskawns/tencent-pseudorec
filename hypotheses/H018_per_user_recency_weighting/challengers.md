# H018 — Challengers (≥ 2 reverse frames)

> §10.3 + §17.4 — H018 이 temporal_cohort 4번째 entry (H015/H016/H017
> 후속). 재진입 정당화 + 반대 frame 명시 의무.

---

## §17.4 rotation 재진입 정당화 (mandatory)

**상황**: 직전 3 H (H015/H016/H017) 모두 `temporal_cohort` 카테고리.
H018 도 same category 면 §17.4 + §10.7 트리거.

**§10.7 audit**: "어느 카테고리도 다른 모든 카테고리 1회 미경험인 상태에서
2회차 금지." 미경험 카테고리 list:
- `retrieval_long_seq` (TWIN/SIM/HSTU) — 미경험.
- `backbone_replacement` (OneTrans full / HSTU trunk / InterFormer 3-arch)
  — 미경험.
- `debiasing` (position bias, exposure bias) — 미경험.

→ 엄격 §10.7 reading: H018 은 retire 카테고리 외 unexplored 가야.

**재진입 정당화 (§17.4 cite)**:

1. **H014 verdict.md F-3 직접 인용**: "4-layer ceiling diagnosis 종료 —
   L1+L3+L4 모두 retire. **L2 (cohort drift) 만 남음**. paradigm shift
   inevitable." → L2 = temporal_cohort. 직접 attack 외 paradigm shift 만
   남음.
2. **H015 verdict (marginal +0.00024pt)**: 부호 positive 라 mechanism
   direction 작동 신호. 형식상 REFUTED 지만 magnitude 가 작은 게
   granularity 약점일 가능성 (per-batch coarse). **per-user fine
   granularity 시도 없이 mechanism class 전체 retire 는 premature**.
3. **H016 verdict (infra PASS)**: redefined OOF 가 platform 과 gap −0.004pt
   ≈ 0 → 측정 framework 생김. H018 의 PASS/REFUTED 판정이 처음으로
   reliable. H016 framework 사용해야 다음 H 들 모두 신뢰성 회복.
4. **§17.6 cost cap audit**: paradigm shift (TWIN/OneTrans) per-job ≥ $20
   추정 (model size 5–10×). 누적 cost ~40h 위에 추가 $40+. 1 회 시도 후
   재기 어려움. **H018 (per-user, T2.4 ~3-4h, ~$5)** 가 cost-effective
   last-attempt.
5. **single mutation 정당성 (§17.2)**: granularity change (per-batch →
   per-user) 단 1개. mechanism stack (NS xattn + DCN-V2 + recency class)
   byte-identical. 이 mutation 자체가 작아서 paradigm shift 보다 1 데이터
   포인트 비용 작음.

**최종 결정**: §10.7 strict 위반 인지 + §17.4 정당화로 진입. 결과에
따라:
- H018 PASS strong → temporal_cohort 4 entry 정당화 confirmed.
- H018 PASS measurable → 다음 H 강제 rotation (temporal_cohort retire,
  retrieval_long_seq 또는 backbone_replacement 진입).
- H018 REFUTED → temporal_cohort 영구 retire, H019 강제 rotation
  (mandatory).

---

## Frame A — "Per-batch coarse 가 진짜 약점이 아니다, per-user 도 noise"

**가설**: H015 의 marginal Δ +0.00024pt 는 진정한 noise (random variation)
임. per-batch coarse / per-user fine 가 무의미. recency mechanism class
자체가 platform AUC lift 못 만듦.

**근거**:
- 5 H 누적 (H011/H012/H013/H014/H015) 모두 OOF saturated 0.858~0.860,
  Platform 0.834–0.840 변동. mechanism class 와 무관하게 ceiling 같은
  영역 → mechanism granularity 가 진짜 lever 가 아님.
- per-user time-decay 는 production CTR 에서 standard 이지만 그건 1B+
  user / 1B+ event 규모. demo_1000 은 1000 row → per-user signal sparse,
  noise > signal.
- exp decay 의 tau 가 hyperparameter (14 일 가정). tau=14 가 잘못된 가정
  이면 weight noise 만 추가.

**Falsification 조건**: H018 Δ vs H015 ≥ +0.5pt → Frame A REFUTED. Δ ∈
[+0.1, +0.5pt] → Frame A 부분 confirmed (per-user 효과 있지만 saturated).

**Frame A confirmed (Δ ≤ 0) 시 carry-forward**: recency mechanism class
전체 retire. paradigm shift mandatory (TWIN / OneTrans). H019 강제
rotation.

---

## Frame B — "Cohort drift 가 hard ceiling, per-user 로도 못 풂"

**가설**: cohort drift 가 train↔platform distribution 의 진짜 hard
ceiling. recency weighting (어떤 granularity 든) 는 distribution shift
의 source 인 cohort 자체 (어느 user 가 platform test 에 등장하는지) 를
바꾸지 못함. weight 만 조정 vs distribution 자체.

**근거**:
- H016 (OOF redefine) 가 platform-aligned cohort 측정 framework 생성 →
  하지만 H016 platform AUC 자체는 H010 anchor 위 −0.0059pt → cohort
  redefinition 도 lift 못 만듦. **cohort 정의 변경만으로는 ceiling 안
  뚫림**.
- platform test cohort 는 organizer 가 random sample (or temporal future)
  로 결정. user-level recency weighting 으로 어떤 user 가 test 에
  포함될지 못 바꿈.
- 진짜 lift 는 model 이 unseen user / unseen item 에 generalize 잘 해야
  → representation learning / regularization / multi-task learning 의
  영역 → 다른 mechanism class 필요.

**Falsification 조건**: H018 Δ vs H015 ≥ +0.5pt → Frame B REFUTED
(cohort hard ceiling 가설 무효, recency mechanism 작동).

**Frame B confirmed (Δ ≤ 0.1pt) 시 carry-forward**: cohort drift 가 hard
ceiling 강한 confirmation. paradigm shift 의 형태도 cohort handling 위
mechanism (e.g., user representation learning) 이 아닌 model
generalization 강화 (e.g., dropout 강화 / regularization / multi-task
auxiliary loss) 가 더 큰 lever 일 수 있음. H019 후보 재선정.

---

## Frame C — "Per-user weighting 이 gradient noise 증폭, per-batch 가
실제로 stable optimum"

**가설**: per-user fine weighting 은 batch 내 weight variance 증폭. 같은
batch 의 sample 들 weight [0.1, 2.0] 같은 wide range → gradient direction
noise 증폭 → SGD convergence 악화 → final val_AUC 하락.

**근거**:
- H013 (large-batch lr scaling) 에서 lr 8e-4 + batch 2048 → OOF AUC 0.857
  = 8 H 중 처음 0.86 미달. **noise 가 학습 자체 disrupt 한 선례**.
- per-batch [0.5, 1.5] 는 weight variance 작음 (range 1.0). per-user exp
  decay (tau=14) 는 variance 큼 (gap 1일 vs 60일 → exp(-0.07) vs
  exp(-4.3) = 0.93 vs 0.014).
- normalize (batch mean=1.0) 해도 variance 자체는 못 줄임.

**Falsification 조건**: per-batch (H015) val_AUC 0.8358 < per-user (H018)
val_AUC → Frame C REFUTED.

**Frame C confirmed (Δ < 0) 시 carry-forward**: per-user → per-batch
intermediate granularity (e.g., per-user-bucket-by-active-level) sub-H.
또는 weight clip [0.5, 1.5] tight cap 으로 variance 제한 sub-H.
