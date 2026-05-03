# PLE — Tang et al. RecSys 2020

- **Title**: Progressive Layered Extraction (PLE): A Novel Multi-Task
  Learning (MTL) Model for Personalized Recommendations.
- **Authors**: Hongyan Tang, Junning Liu, Ming Zhao, Xudong Gong (Tencent).
- **Venue**: RecSys 2020 (Best Paper).
- **Read at**: 2026-05-01 (carry-forward from H012 scaffold).

## Core mechanism (1단락)

PLE 는 MMoE 의 cross-task interference 문제 해결. expert 를 **shared expert**
와 **task-specific expert** 로 분리. multi-layer 구조에서 **progressively**
task-specific expert 가 자기 task 의 정보만 받도록 layer 마다 점차 분리.
첫 layer 는 모든 expert (shared + task-specific) 가 모든 task 에 routing
가능, 마지막 layer 는 task-specific expert 만 자기 task 에 routing. CGC
(Customized Gate Control) 가 PLE 의 single-layer 변형. multi-task 학습
seesaw 현상 (task A 향상 → task B 악화) 감소.

## Why this matters for TAAC 2026 UNI-REC

본 H (H012) 의 minimum viable form 은 MMoE 단일 layer. PLE 는 **future
sub-H 후보**:
- H012 가 PASS 하지만 expert specialization 이 제한적 (entropy 너무 높거나
  cross-domain leakage) → PLE progressive separation 으로 강화.
- multi-domain 의 case 에서 도메인-specific expert 와 shared expert 분리
  가 자연스러움 (도메인 a 와 d 가 vocab 거의 안 겹치지만 user behavior
  pattern 은 공유 가능).
- Tencent 자체 paper — TAAC organizer 와 같은 회사 (Junwei Pan = Tencent
  Ads). production thinking 일관성 있을 가능성.

## Adoption notes for H012 (sub-H 후보)

본 H 에선 미적용 (single-layer MMoE 우선). PLE 는 sub-H:
- **Adopt** (sub-H 시): shared expert (1-2 개) + domain-specific expert
  (4 개). 첫 layer routing 자유, 마지막 layer 는 도메인 specific 만.
- **Modify**: multi-layer 구조 → params 약 2× 증가, sample-scale §10.6
  cap 위반 위험. minimum form = 2-layer with single domain-specific layer.

## Reference key claims (paper)

- Section 3 (PLE formulation): shared expert + task-specific expert layered
  selection. Eq. (3): task k 의 layer l 에서 gate `g^l_k(x) = softmax(W^l_k
  x)` 가 layer l 의 shared + task-k-specific expert 위에 작동.
- Section 4 experiments (Tencent Video production + multi-task synthetic):
  PLE > MMoE > Cross-Stitch > Shared-Bottom 일관되게.
- Section 4.4: seesaw 현상 분석 — MMoE 도 cross-task interference 일부 남아
  있음, PLE 가 명시 분리로 해결.

## Caveats

- **Sample-scale 에서 cap 위반 위험**: 2-layer PLE = MMoE × 2 ~= 66K
  params. anchor envelope 면제 인정 후도 §10.6 budget 압박.
- **Multi-task 효과가 multi-domain 에서 그대로 transfer 불확실**: PLE 는
  task 별 output head + supervision 분리 가정. single-task multi-domain 에선
  효과 작을 수 있음.
- **H012 결과 우선 측정 후 sub-H 결정**: H012 (single-layer MMoE) PASS +
  expert specialization confirmed → PLE sub-H 가치 있음. H012 noise 또는
  collapse → PLE 도 같은 한계.
