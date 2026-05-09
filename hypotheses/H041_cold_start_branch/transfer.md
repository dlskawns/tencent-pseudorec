# H041 — transfer.md (UNI-REC alignment)

## §⑤ UNI-REC alignment
- **Sequential axis**: H019 의 TWIN GSU+ESU + per-domain encoder 그대로 carry.
- **Interaction axis**: NS→S xattn + DCN-V2 stack 그대로 carry.
- **Cold-start branch**: backbone output 에서 분기 — 둘 다 같은 PCVRHyFormer
  trunk 의 latent 사용 → seq + interaction 통합 후 specialization.
- **Gate signal source**: backbone output 자체 → 모델 자체적 cold/familiar 판단.

## Why this mechanism for THIS data (CLAUDE.md §0.5)
- Signal: §3.5 target_in_history=0.4% → extrapolation regime dominant.
- Diagnose: single classifier 가 두 regime 평균화 → novel 케이스 fit 약화 가설.
- Mechanism design: dual classifier + learned gate (paper transplant 아님,
  data signal 직접 attack).
- Validation chain: gate 가 학습 안 되면 init 0.88 → main 지배 → H019 등가.
  학습되면 specialize.

## What's NOT a clone
- 외부 paper 의 cold-start 처리 (예: meta-learning, dual tower contrastive)
  과 다름 — 본 mechanism = 단순 dual head + per-sample gate.
- Mixture-of-Experts 도 아님 — expert 가 K 개 아닌 fixed 2 개, gate 가 backbone
  conditioning 만 사용.

## Carry-forward
- §17.2 single mutation: classifier 단일 → dual + gated. 다른 모든 부분
  byte-identical to H019.
- §17.4 rotation: NEW first-touch (data-signal-driven specialization),
  AUTO_JUSTIFIED.
- §10.10 InterFormer bridge gating σ(−2)≈0.12 패턴 변형 — gate init = 0.88
  (main 지배), TWIN 의 gate init = 0.12 (TWIN 종속) 와 대칭.
