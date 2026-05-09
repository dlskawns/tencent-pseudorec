# H043 — transfer.md (UNI-REC alignment)

## §⑤ UNI-REC alignment
- **Sequential axis**: H019 의 TWIN GSU+ESU + per-domain encoder 그대로 carry.
- **Interaction axis (user-side)**: NS→S xattn + DCN-V2 user/global stack
  그대로 carry.
- **Interaction axis (item-side, NEW)**: item_ns 토큰 자체에 DCN-V2 cross —
  item feature interaction 가 backbone 으로 들어가기 전 explicit modeling.
- **두 축 동시**: per-domain seq encoder (sequential) × user-side DCN-V2
  (interaction-user) × item-side DCN-V2 (interaction-item, H043 신규) =
  **§0 P1 unified block 가 두 축 + 두 측면 (user/item)** 모두 같은 backbone
  안 gradient 공유.

## Why this mechanism for THIS data (CLAUDE.md §0.5)
- Signal: §3.5 4 도메인 disjoint vocab (max jaccard 10%) + target_in_history
  =0.4% — item 의 cross-domain semantic 이 모델링 안 되면 cold case 에 대한
  generalization 부족.
- Diagnose: 14 H 동안 user-side 만 modeling, item 은 nn.Embedding lookup
  뿐. F-A pattern (OOF over Platform) 의 일부 원인이 item representation
  generalization 불충분 가능.
- Mechanism design: user-side DCN-V2 (H008 PASS +0.0035pt) 의 same form 을
  item-side 에 applied. 외부 paper transplant 아님 — 본 데이터의 user-side
  성공 mechanism 의 symmetric 적용.
- Validation chain: T0 sanity 8 test PASS (forward + NaN-free + grad +
  ablation diff + md5 verify).

## What's NOT a clone
- 외부 paper 의 item embedding pretraining (item2vec 등) 과 다름 — joint
  end-to-end training, 별도 pretraining stage 없음.
- Item-side cross attention 과 다름 — item_ns 토큰 *내부* feature interaction
  만, sequence attention 없음.
- 외부 paper transplant 0. mechanism 정당성은 user-side DCN-V2 작동 +
  §3.5 disjoint vocab signal 두 데이터 facts 에서 옴.

## Carry-forward
- §17.2 single mutation: item-side cross block 추가, model graph 의 item
  branch 만 변경, user-side / TWIN / 다른 모든 부분 byte-identical to H019.
- §17.4 rotation: item_side_modeling axis NEW first-touch, AUTO_JUSTIFIED.
- §10.5 LayerNorm on x_0 mandate: DCNV2CrossBlock 의 Pre-LN 자동 통과.
- §10.6 sample budget: trainable params +1.28K, hard cap 200 의 6.4× 였지만
  본 cap 은 sample-scale (1000 rows) 용 — cloud full-data 시 무관.
- §0.5: §3.5 데이터 signal 직접 attack.
