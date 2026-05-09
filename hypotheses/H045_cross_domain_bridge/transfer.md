# H045 — transfer.md (UNI-REC alignment)

## §⑤ UNI-REC alignment
- **Sequential axis**: H019 의 TWIN GSU+ESU + per-domain encoder 그대로 carry.
- **Interaction axis (user/global)**: NS→S xattn + DCN-V2 stack 그대로 carry.
- **Cross-domain axis (NEW)**: 4 도메인 encoder output 사이 attention bridge —
  도메인 간 sequential semantic 의 explicit transfer. backbone output 에
  residual ADD → §0 P1 unified block 안 sequential × interaction × cross-domain
  세 측면 gradient 공유.

## Why this mechanism for THIS data (CLAUDE.md §0.5)
- Signal: §3.5 4 도메인 jaccard 0.7~10% — disjoint 같지만 부분 overlap. 도메인
  간 정보 transfer 가 학습 가능한 신호 실재.
- Diagnose: 14 H 동안 도메인 independence 만 modeling. TWIN aggregator (mean)
  와 per-domain encoder 가 정보를 *섞지* 않고 *모음*. cross-domain 정보 손실.
- Mechanism design: 4 도메인 토큰 사이 MultiheadAttention — 가장 단순하고
  표준적인 cross-domain transfer form. paper transplant 아님 — 본 데이터의
  jaccard signal 직접 motivation.
- Validation chain: T0 sanity 9 test PASS (forward + grad + masked + ablation
  diff + md5 verify).

## What's NOT a clone
- 외부 multi-domain recsys paper 의 hard top-K routing (H012 MoE) 와 다름 —
  H045 는 dense attention, gate routing 없음.
- 외부 cross-domain transfer learning (DANN domain alignment 등) 과 다름 —
  H045 는 4 도메인 모두 같은 task 의 한 user 의 history, domain adaptation 아님.
- 외부 paper transplant 0. mechanism 정당성은 §3.5 jaccard signal + 14 H
  도메인 independence 의 한계 누적.

## Carry-forward
- §17.2 single mutation: CrossDomainBridge 1개 추가, model graph 의
  post-backbone path 만 변경, user-side / TWIN / 다른 모든 부분 byte-identical
  to H019.
- §17.4 rotation: cross_domain_modeling axis NEW first-touch, AUTO_JUSTIFIED.
- §10.10 InterFormer bridge gating σ(-2)≈0.12 mandate 통과.
- §10.6 sample budget: trainable params +21K (sample-scale 안).
