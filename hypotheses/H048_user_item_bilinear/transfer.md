# H048 — transfer.md (UNI-REC alignment)

## §⑤ UNI-REC alignment
- Sequential: H019 TWIN GSU+ESU + per-domain encoder carry.
- Interaction: NS→S xattn + DCN-V2 carry.
- **User × Item cross (NEW)**: backbone output 위 explicit bilinear cross.
  같은 unified backbone graph 안 main BCE + user×item cross gradient 공유.
  Sequential × Interaction × User-Item cross 세 측면 통합 강화.

## Why this mechanism for THIS data
- 14 H 의 PASS mechanism 패턴 (새 정보 흐름) 직접 추적.
- §3.5 target_in_history=0.4% (cold item) → user × item explicit interaction
  이 cold-item generalization 의 가능 lever.
- FM-style cross 의 top-level 적용, paper transplant 아님 — H019/H010 PASS
  pattern 의 자연 다음 form.

## What's NOT a clone
- 외부 FM/DeepFM paper 와 다름 — input-side cross 가 아닌 backbone output
  level cross.
- TWIN (item attends history, 한 방향) 과 다름 — user_repr × item_repr
  symmetric cross.
- DCN-V2 (within-side polynomial) 와 다름 — cross-side bilinear.

## Carry-forward
- §17.2 single mutation: UserItemBilinearCross 1개 추가.
- §17.4: user_item_explicit_cross axis NEW first-touch.
- §10.10: bridge gating σ(-2)≈0.12.
- §0.5: 14 H "새 pathway = lever" 패턴 직접 추적.
- **§18.1+ infer.py flag parity (memory feedback rule)**.
