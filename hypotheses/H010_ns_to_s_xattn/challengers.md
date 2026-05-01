# H010 — Challengers

> CLAUDE.md §10.1 의무: 반대 frame ≥ 2.

## Frame 1 — "H007 single-token candidate query 가 marginal pass 였는데, NS 7-token query 일반화는 attention pattern uniform collapse 위험. paper-grade 라도 sample-scale 에서 작동 안 할 수 있다."

**Take**: H007 candidate token 1개가 384 S tokens (4 도메인 concat) 에 cross-attend → attention prob 가 candidate 의 query 의미에 집중 가능. NS 7 tokens 모두 query 로 작동 시 query 가 7배 → softmax attention 이 uniform 으로 spread 될 위험. §10.9 룰의 정확히 활성 영역. 또한 NS tokens 가 user/item feature embedding (categorical pool) 이라 candidate 처럼 명확한 query semantic 없음 → attention 학습 어려움.

**Counter-take**: §10.9 룰 적용 (threshold 0.95 × log(384) ≈ 5.65) 으로 abort 자동. 또한 OneTrans paper 자체가 100M user 규모에서 NS→S bidirectional 작동 confirmed. 우리 sample scale (extended 30% × 10 epoch) 이 실패 sufficient size 인지 측정 가치. 만약 attention uniform collapse 면 그 자체가 정보 (sample-scale 한계 정량화) — REFUTED 명확히 interpretable. 그리고 NS tokens 가 RankMixerNSTokenizer 로 already learned representation 이라 query semantic 약하다는 가정은 단정. NS tokens 도 user/item features 의 chunked embedding → 각자의 information content 있음.

**Resolution**: §10.9 attn entropy 룰 mandatory 적용. instrumentation 의무. PASS 시 NS-token query semantic 도 작동 confirmed, REFUTED 시 sample-scale 한계 또는 mechanism 한계 분류.

---

## Frame 2 — "H008 anchor 위 stacking 은 H009 와 같은 단일 mutation 위반 / interference risk. anchor (original_baseline) 위 NS xattn 단독 측정이 깔끔한 single mutation."

**Take**: §17.2 one-mutation-per-experiment 엄격 적용 시 anchor 위 단일 변경. H008 anchor 위 stacking 은 사실상 2-mutation (DCN-V2 + NS xattn). H009 의 interference 패턴 재발 risk. 또한 anchor 위 단독 측정이 NS xattn 의 isolated effect 정량 가능.

**Counter-take**: H010 의 통합 위치 (fusion 이전, NS dimension 변경 없음) 가 H009 와 다름 — H009 는 candidate token prepend 가 seq encoder 출력 변경 → DCN-V2 입력 변경 (위치 충돌). H010 NS xattn 은 NS tokens 만 enriched (S tokens, query decoder, DCN-V2 입력 변경 없음) → 위치 충돌 회피 by 설계. §17.2 anchor exemption 정신 (challengers Frame "stacking sub-H 합법") — H007 단독 검증 + H008 단독 검증 후 H010 = NS xattn (H007 일반화) on H008 anchor (champion) 은 stacking sub-H 합법. 또한 isolated effect 측정 위해 anchor 위 단독 측정 시 H011 = NS xattn + DCN-V2 stacking 추가 H 필요 (총 2 H, cost 2x). H010 stacking 으로 한 번에 측정 + paired vs H008 sub-criterion 으로 isolated effect 추정 가능 (super-additive vs additive vs interference 분류).

**Resolution**: H008 anchor 위 stacking 채택. 단 predictions.md 의 paired vs H008 sub-criterion 으로 isolated effect 추정 의무. H009 위치 충돌 회피 분석 (transfer.md) 명시. interference 시 mechanism class 한계 분류.

---

## Frame 3 — "OneTrans paper claim 의 +0.4-1.2pt AUC 는 100M user 규모. extended envelope 30% × 10 epoch 우리 데이터 (~10-15M sample steps) 에서 NS→S bidirectional 의 paper claim transfer 안 될 수 있다. anchor recalibration 으로 envelope 효과 isolation 먼저."

**Take**: H010 PASS 가 NS xattn mechanism 효과인지 envelope 효과 인지 분리 안 됨 (anchor smoke 만 있음). 또한 OneTrans paper claim 의 100M user 규모 의존성 가정 시 sample-scale 에서 lift 약화 — H004 verdict F-2 ("underpowered training vs architectural fit 구별 불가") 패턴 재발 risk.

**Counter-take**: H006/H007/H008/H009 모두 같은 extended envelope (paired comparison 가능). H010 도 같은 envelope → paired Δ 신뢰 가능. envelope 효과 separate 는 anchor recalibration H 의 별도 가치 (이번 turn 에서 H010 이전 안 함, backlog 로 미룸 — 사용자 합의). 또한 NS→S bidirectional 의 sample-scale viability 는 H004 backbone replacement (full OneTrans) 와 다름 — H010 은 mechanism injection (PCVRHyFormer baseline + 1 cross-attention layer 추가). param count 추가 ~12-50K (OneTrans full 195K dense + softmax attention 4 layers 와 다름). sample-scale risk 작음.

**Resolution**: H010 진행. anchor recalibration 은 backlog 유지 (사용자 가치 align). envelope 효과 의문 시 추후 anchor recalibration H 별도 측정.

---

## Frame 4 — "H010 통합 위치 (per-domain seq encoder 출력 concat → NS xattn → query decoder) 에서 도메인 간 cross-attention 이 도메인 ID 정보 제공 안 함. NS tokens 가 어떤 도메인 history 에 attention 하는지 모름 → effective routing 약화."

**Take**: 4 도메인 (a/b/c/d) S tokens 을 concat (L_total=384) 후 NS tokens cross-attend 시 도메인 ID 정보가 implicit (position 만). PCVRHyFormer 의 query decoder 는 per-domain 분리 처리 → 도메인 routing 명확. NS xattn 이 도메인 ID 정보 추가 안 하면 attention 이 도메인 random spread 가능.

**Counter-take**: 도메인 ID embedding 추가는 별도 mutation (§17.2 위배). H010 minimum viable form 으로 시작 — concat + position implicit 으로 도메인 정보 표현. attention 이 implicit position 으로 도메인 학습 가능 (transformer 자연 능력). PASS 시 confirmed, REFUTED 시 도메인 ID embedding 추가 sub-H 후속. 또한 H004 OneTrans 가 도메인 ID embedding 추가했지만 attn entropy 측정에서 layer 1 0.614 / layer 2 0.689 로 sparse pattern (uniform 아님) — 도메인 routing 작동 가능성. H010 도 같은 패턴 expected.

**Resolution**: H010 minimum viable form 채택 (concat + position implicit). 도메인 ID embedding 은 별도 sub-H 후보.

---

## §재진입정당화 카테고리 (target_attention)

H007 = target_attention (첫 적용, PASS marginal). H008 = sparse_feature_cross. H009 = hybrid stacking (재진입정당화). H010 = target_attention 재진입.

§10.7 룰 ("같은 papers/{cat}/ 에서 2회 연속 실험 금지 (재진입정당화 없으면)") 적용:
- 2회 연속 아님 (H007 다음 H008, H009 사이).
- 재진입정당화: H007 의 1-token candidate query 를 N_NS-token 으로 일반화. paper-grade mechanism (OneTrans NS→S bidirectional 직접 구현). H007 단독 검증된 mechanism class 의 강화 버전. 같은 paper 1:1 재현 아님.
- transfer.md §⑤ 재진입정당화 명시 의무.
