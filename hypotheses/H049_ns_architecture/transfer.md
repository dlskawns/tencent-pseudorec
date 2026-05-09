# H049 — transfer.md (UNI-REC alignment)

## §⑤ UNI-REC alignment
- Sequential: H019 TWIN GSU+ESU + per-domain encoder carry.
- Interaction: NS→S xattn + DCN-V2 carry.
- **NS architecture (NEW input-level)**: item_ns expansion + type embedding 이
  NS-token 단계 부터 user/item 통합 표현 강화. backbone 이 type-aware 처리.

## Why this mechanism for THIS data
- 14 H 동안 NS structure 0회 변경 — 진짜 unexplored axis.
- 사용자 직접 지적 axis (S/NS).
- §0 P1 의 두 축 unified block 의 가장 fundamental input.

## What's NOT a clone
- 외부 token expansion paper (BERT의 [CLS] 추가 등) 와 다름 — 본 case 는
  domain-specific (user/item/dense) type 차별.
- H012 MoE NS-token (uniform routing REFUTED) 와 다름 — H049 는 routing
  안 함, type 명시적 embedding 만.
- 외부 paper transplant 0.

## Carry-forward
- §17.2 single mutation: NS architecture refinement.
- §17.4: ns_architecture axis NEW first-touch.
- §10.6: trainable params +16K.
- §0.5: NS-axis unexplored 직접 attack.
- **infer.py flag parity** verified.
