# MMoE — Ma et al. KDD 2018

- **Title**: Modeling Task Relationships in Multi-task Learning with
  Multi-gate Mixture-of-Experts.
- **Authors**: Jiaqi Ma, Zhe Zhao, Xinyang Yi, Jilin Chen, Lichan Hong,
  Ed Chi (Google).
- **Venue**: KDD 2018, arXiv:1810.10739.
- **Read at**: 2026-05-01 (carry-forward from H012 scaffold).

## Core mechanism (1단락)

MMoE 는 N 개의 expert tower (각각 small MLP, 보통 2-layer FFN with bottleneck)
+ K 개의 task 마다 별도 gate network 로 구성. gate 는 input x 를 받아 N
expert 위 softmax routing weight 출력. task t 의 출력 = `Σ_i gate_t(x)_i ×
expert_i(x)`. expert 는 task 간 공유, gate 만 task-specific. specialization
은 학습으로 자동 emergent — task 가 비슷하면 같은 expert 활용, 다르면 다른
expert. shared-bottom MLP 보다 cross-task interference 감소.

## Why this matters for TAAC 2026 UNI-REC

TAAC 2026 데이터: 4 도메인 (a/b/c/d) 의 vocab 거의 disjoint (Jaccard ≤
0.10), length 분포 차이 큼 (§3.5). 단일 fusion block (PCVRHyFormer 의
HyFormer block) 이 4 도메인 균일 처리 → MMoE 패턴 적용 = **task → domain**
매핑.

본 H (H012):
- N_experts = 4 (= 도메인 수).
- gate per NS-token (input-conditioned, not fixed per task).
- expert specialization → 도메인별 vocab disjoint 활용.
- single-task setting (post-click conversion) — multi-task 가 아닌 multi-
  domain 적용.

## Adoption notes for H012

- **Adopt**: expert tower + softmax gate. shared expert 위 specialization.
- **Modify**:
  - Multi-task → multi-domain (single-task setting, gate per NS-token).
  - 4 experts (= 도메인 수, fixed).
  - ffn_hidden = 128 (= 2 × d_model=64) — sample-scale §10.6 budget 안.
  - residual: `output = ns + moe(ns)` (H010 F-1 안전 stacking).
- **Sample-scale viability**: ~33K params (4 × FFN + gate). anchor envelope
  면제 적용.

## Reference key claims (paper)

- Section 3.2 (MMoE formulation): `y_t = h_t(Σ_i g_t(x)_i × f_i(x))` where
  f_i = expert i, g_t = gate for task t, h_t = task-specific tower.
- Section 4 experiments (UCI Census, synthetic, Google production): MMoE
  > shared-bottom > task-specific independent on multi-task with low
  task-similarity.

## Caveats

- **Sample-scale collapse 위험**: 1000-row × 30% extended 환경에서 expert
  routing 이 1-2 expert 에만 집중하는 collapse 가능. §10.9 entropy 룰 적용
  필수 (threshold 0.5 × log(4) ≈ 0.69).
- **Multi-task 가 아닌 multi-domain 적용**: paper 의 task-specific output
  head 미적용 — single output head + 도메인은 input 의 implicit 정보로만
  존재. routing 효과가 task-specific output head 의 강제 분리 효과보다 약할
  수 있음.
- **Single layer**: PLE 의 progressive separation 미적용 (sub-H 후보).
