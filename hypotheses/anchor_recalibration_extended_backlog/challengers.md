# H010 — Challengers

> CLAUDE.md §10.1 의무: 반대 frame ≥ 2.

## Frame 1 — "Anchor recalibration 은 측정 행위라 mechanism H 가 아니다.
> envelope 변경은 단일 mutation 이지만 lift 측정 목적 H 가 아니므로 이번 cloud
> budget 1 슬롯을 mechanism H 에 써야 한다."

**Take**: §17.6 cost cap 압박 (H006~H009 누적 ~14시간 + H010 ~4시간 = ~18시간).
mechanism class rotation 1순위 후보 (NS→S xattn, MMoE/PLE multi_domain_fusion,
aligned `<id, weight>` pair encoding) 1개 진행이 더 큰 lift 정보 가치.
recalibration 은 H011+ 결과 paired Δ 의 보정 factor 정도로 다뤄도 됨 — 정확한
anchor 절대값 없어도 paired Δ 부호와 ranking 은 의미 있음 (H008 > H009 >
H007 > anchor 라는 ranking 은 anchor 정확값에 무관하게 confirmed).

**Counter-take**: H009 verdict F-3 이 정량 노출 — anchor 정확값이 결론 분류
(additive vs sub-additive vs interference) 자체를 흔든다. 즉 paired ranking 만
으로는 부족, 절대값 차이 (Δ +0.001 vs +0.006) 이 중요. 예: H011 = NS→S xattn
이 +0.003pt lift 시 anchor=0.83 기준 marginal pass / anchor=0.835 기준 fail.
recalibration 안 하면 H011~ 도 같은 모호함 재발. 한 번 measurement 으로
최대 5+ H 의 paired Δ 정확화. **cost-effective**.

**Resolution**: anchor recalibration 우선 진행. mechanism class rotation 은
H011 부터 (recalibrated anchor 기준).

---

## Frame 2 — "Anchor recalibration extended 결과가 envelope effect 가 작아서
> 0.83~0.835 그대로면 정보 가치 0이고 cost 만 소모. 결과 분류가 'envelope 효과
> 작음' 으로 미리 알려져 있다면 측정 안 해도 된다."

**Take**: smoke (1ep × 5%) 와 extended (10ep × 30%) 사이의 envelope 효과는 보통
production 베이스라인 에서 plateau 이전이라 크게 안 나는 경우가 흔하다. PCVRHyFormer
가 organizer-tuned baseline 이면 1 epoch + 5% data 에서 이미 거의 generalize.
extended 에서 lift 가 작으면 (≤ 0.5pt) measurement 가치 작음.

**Counter-take**: H006 (extended longer encoder) 가 anchor smoke (~0.83) 대비
**0.82** 로 −1pt **감소** 했다. 즉 extended envelope 에서 baseline 변경이 lift
주는 게 아니라 손해 보는 패턴 — envelope 효과가 negligible 이 아닌 신호 (만약
envelope 효과가 +0.5pt 이상이었으면 H006 baseline 이 anchor 보다 위에 있어야).
또한 H007/H008 (extended baseline 위 mechanism 추가) 가 0.8352/0.8387 로
anchor 보다 위 → envelope + mechanism 합산 효과. anchor extended 측정 안 하면
이 합산을 분리 못 함. **결과가 0.83~0.835 그대로 라도 정보 가치 있음** (envelope
효과 ≈ 0 confirmed → mechanism 효과 dominant 결론). 결과 어느 쪽이든 정보 가치
충분.

**Resolution**: anchor recalibration 진행. 결과 시나리오 분기 (problem.md 의
A/B/C) 모두 의미 있는 정보.

---

## Frame 3 — "patience=3 + train_ratio=0.3 는 H006~H009 envelope 와 paired
> 비교 가능하지만, 만약 anchor extended 가 sample size 부족으로 underpowered
> 면 anchor 가 부당하게 낮게 측정될 수 있고 (H007/H008 mechanism이 추가
> capacity 보정) 'envelope 효과 작음' 잘못된 결론 가능."

**Take**: PCVRHyFormer 는 198M params (158M sparse + 2.5M dense). train_ratio=0.3
× 10 epoch 로 충분히 학습 안 될 수 있고, 추가 mechanism 의 capacity (DCN-V2
2K params, candidate xattn 50K params) 가 sparse embedding 의 reinit/skip 패턴
변경을 통해 학습 dynamics 자체를 바꿀 수 있음.

**Counter-take**: H007/H008 도 같은 train_ratio=0.3 × 10 epoch 에서 학습됨.
같은 envelope 에서 측정 = paired 비교 가능. 만약 anchor extended 가 underpowered
이면 H007/H008 도 같은 underpowered 영역에서 측정된 것 → ranking 은 여전히
유효, 절대값만 다소 낮을 가능성. 또한 patience=3 으로 plateau 도달 시 early
stop. F-1 (H007 verdict) 이 이미 "extended envelope 임에도 plateau 일찍 옴 (3
epoch 에서 best)" 명시 — sample size 부족 신호 약함.

**Resolution**: patience=3 + train_ratio=0.3 envelope 그대로 유지. 결과 underpowered
의심 시 sub-H (train_ratio=1.0 또는 num_epochs=20) 별도 추가.

---

## Frame 4 — "anchor recalibration 은 H 라기보다 'measurement re-run' 이라
> hypothesis 패키지로 다루는 것 자체가 overhead. card.yaml + run.sh + INDEX
> entry 만으로 충분."

**Take**: §5 workflow cycle 의 step 1-7 (problem → challengers → lit → transfer
→ predictions → card → verdict) 이 mechanism H 용. anchor recalibration 은
literature scout 도 transfer 도 의미 적음 (재사용 코드, 새 paper 없음). 가벼운
re-run entry 로 처리.

**Counter-take**: §5 workflow 가 H 모든 종류에 적용 (anchor 자격 측정 H001/H004
도 같은 cycle). hypothesis package 가 의사결정 traceability 보장. 특히 결과
시나리오 (A/B/C) 가 후속 H 우선순위 변경하는 큰 효과 있음 → predictions.md
의 사전 등록이 후속 결정 자기-편향 방지. lit_refs.md 는 minimal (논문 인용
없음, prior H verdict 인용만) 로 처리.

**Resolution**: 표준 6 파일 hypothesis package 로 진행. lit_refs.md 는 prior
H verdicts 인용만으로 minimal.

---

## §재진입정당화 카테고리

본 H 는 mechanism category 미부여 (envelope mutation). §10.7 카테고리 rotation
2회 연속 차단 룰 적용 안 됨 (n/a). H011+ 부터 mechanism category 다시 적용.
