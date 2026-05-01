# H009 — Challengers

> §10.1 — 모든 H 시작 전 반대 프레임 ≥ 2.

## §재진입 정당화 (§17.4)

H007 = target_attention, H008 = sparse_feature_cross. H009 = 두 카테고리 hybrid. §17.4 의 "직전 2개 H 같은 카테고리 차단" 룰 — **다른 카테고리 두 개 stacking 이라 직접 적용 안 됨**. 본 H 정당화:

- 본 H 는 **새 mechanism 도입 아닌 stacking** — H007 + H008 단독 PASS 후 additivity 검증.
- §0 north star 의 "두 축 동시 강화" 가 정확히 stacking 으로 검증됨.
- carry-forward: H007 verdict F-1 ("sequence axis mechanism PASS"), H008 verdict F-3 ("additivity 검증 필요") 직접 인용.

## Frame 1 — "Sub-additivity (interference): H007 효과 가 H008 의 effective gradient flow 에 방해될 수도"

H007 의 candidate summary token 이 seq 시작에 prepend 되어 seq length 가 L → L+1. 다운스트림 query decoder + RankMixer (이제 DCN-V2 cross) 가 그 추가 token 처리. DCN-V2 cross 의 polynomial cross 에 candidate summary 가 noise 로 작용해 cross weights 학습 방해 가능성.

**구체적 risk**:
- DCN-V2 cross 의 W = U V^T (low-rank rank=8) 가 candidate summary token 의 representation 을 cross 에 효과적으로 통합 못 할 수도.
- candidate summary 가 일종의 "context anchor" 역할 위해 prepend 됐는데 RankMixer (token-mixing) 는 그걸 잘 활용했으나 DCN-V2 cross (per-token) 는 통합력 약할 가능성.
- 두 mechanism 의 학습 dynamics 가 single 보다 더 엉킬 수 있음 (lr 같음, 1 epoch 학습).

**mitigation**:
- §17.3 binary 임계 +0.5pt 가 falsifier — sub-additive 도 +0.5pt 넘으면 PASS (additive 만 못해도 lift 자체는 의미).
- 측정 Δ < max(H007 Δ, H008 Δ) → mechanism 간 interference 신호 → carry-forward 로 통합 위치 재고려 (별도 H).
- patience=3 + early stop 으로 cost 절약, 두 mechanism 의 fitting balance 측정.

**falsifier**: combined Δ < H008 단독 Δ → H007 의 candidate summary 가 DCN-V2 cross 와 호환 안 됨 → H007 mechanism 통합 위치 변경 (예: prepend 대신 separate token stream) sub-H.

## Frame 2 — "Super-additivity: 두 mechanism 시너지 가 paper-grade 발견 일 수도"

candidate summary token 이 candidate-relevant history events 정보 를 1 token 으로 압축. DCN-V2 cross 가 token-wise polynomial — candidate summary token 자체 의 internal representation 도 polynomial 로 enrich. 즉 (sequence-axis output) × (interaction-axis cross) 가 multiplicative interaction 가능. paper-grade 발견.

**구체적 risk** (positive):
- combined Δ > +0.007pt → super-additive. paper 보고 가능.
- 단 single experiment 라 confound 가능성 — multi-seed bootstrap 필요할 수도.

**mitigation**:
- 측정 Δ ∈ [+0.005, +0.012pt] → additive ~ super-additive 영역. P5 (paper-grade 발견) 검증 위해 multi-seed × 3 sub-H 추가.
- super-additive 시 interpretation: candidate-aware seq context + explicit feature cross interaction 의 multiplicative 효과 분석.

**falsifier 아님**: super-additive 는 positive 발견. challengers Frame 으로 적합하지 않음. but noted.

## Frame 3 — "단독 PASS 둘이 stacking 됐으니 새 mechanism 도입이 본질이 아님 — exploration 가치 낮을 수도"

H009 가 새 mechanism 도입 안 함 — 단지 두 단독 PASS stacking. 정보 가치 = additivity 검증만. 다른 H (CAN co-action, multi_domain_fusion 등) 는 새 mechanism 도입 → 더 큰 정보 잠재력. H009 보다 우선순위 낮을 수도.

**구체적 risk**:
- combined PASS 든 sub-additive 든 super-additive 든 결과는 정성적 정보 (additivity 가정 검증). 정량적 lift 는 H007 + H008 단독 합 근처 → exploration 새 axis 보다 marginal.
- §17.6 cost cap: H009 ~3-4시간 wall + inference. 다른 H 도 비슷.

**mitigation**:
- H009 의 핵심 가치 = **새 anchor 후보 등록**. PASS 시 H009-anchor 가 미래 H 의 paired Δ baseline → 더 정확한 측정.
- §0 north star 두 축 동시 강화 직접 검증 = 프로젝트 핵심 가설 검증. 추가 mechanism 시도 전 sanity check.
- 완료 후 즉시 다음 axis (multi_domain_fusion, external_inspirations) 탐험.

**falsifier 아님**: 정보 가치 의문 frame. mitigation 으로 해결 가능.

---

§10.1 충족: 반대 프레임 3개. Frame 1 (sub-additivity interference) 가 진짜 risk, Frame 2/3 는 framing 검증.

§17.4 정당화: stacking 새 mechanism 아님, 두 단독 PASS 의 additivity 검증.

§10.4 external_inspirations 의무: 본 H 미충족. H010+ carry-forward (Switch Transformer load balance, etc).
