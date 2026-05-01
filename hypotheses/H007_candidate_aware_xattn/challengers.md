# H007 — Challengers

> §10.1 — 모든 H 시작 전 반대 프레임 ≥ 2.

## Frame 1 — "Single candidate-attended token prepend 가 마진 lift 정도밖에 못 만들 수도"

본 H 는 mechanism class ("candidate as attention query") 의 minimum viable form 로 구현 — per-domain 1 token cross-attention → prepend. 다운스트림 (`MultiSeqQueryGenerator`) 가 그 1 token 을 추가 정보로 활용 가능하지만, query decoder 가 learnable Nq queries 자체를 candidate-aware 로 swap 하는 더 강한 변형 (예: CAN co-action, HSTU hierarchical multi-layer) 보다 약함.

**구체적 risk**:
- DIN paper 의 lift 보고 0.3–1.0pt 는 **dedicated DIN architecture** (target attention 이 main path) 기준. 우리 구현은 augment (prepend) 이라 효과 약화.
- 1 candidate token 만 prepend 하면 다운스트림 RankMixer 가 그걸 무시할 수도 (token mass 가 작음).
- candidate token 자체를 item_ns + item_dense mean pool 로 만든 구성이 충분히 informative 한지 paper-검증 부재.

**mitigation**:
- §17.3 binary 임계 +0.5pt 가 falsifier — 미달 → REFUTED + carry-forward 로 더 강한 변형 (CAN co-action, multi-layer target attention) H008 후보.
- 본 H 는 **mechanism class 가 우리 데이터에 의미 있는지** 의 sanity check. 통과하면 구현 디테일 강화 H 후속.

**falsifier**: smoke + extended 양쪽 모두 Δ < +0.5pt → REFUTED + variant H 큐 정렬.

## Frame 2 — "Candidate token 구성이 부적절할 수도"

본 H 의 candidate token = `item_ns + item_dense_tok` mean pool. 이게 candidate item 의 "쓸 만한" representation 인지 paper-grade reference 부재. DIN paper 는 candidate item 의 raw embedding 직접 사용 (하나의 ID embedding). 우리는 organizer 가 만든 multi-token representation 을 mean pool — average over heterogeneous tokens (categorical + dense).

**구체적 risk**:
- mean pool 이 information mass 손실. weighted pool (learnable) 또는 first token 사용이 더 나을 수 있음.
- item_dense_tok 와 item_ns 가 representation space 가 다를 수 있음 (item_ns 는 RankMixer chunked, item_dense 는 SiLU(linear) projected) → mean pool 시 noise.
- DIN paper 는 item ID 직접 embedding 1개. 우리는 7개 token 평균 → information dilution.

**mitigation**:
- 본 H 는 **mean pool 1차 시도**. 효과 marginal 시 candidate token 구성 alternative (first token, learnable weighted, item_id 직접 embedding) sub-H 후보.
- 디자인 선택을 transfer.md §④ 에 명시 — paper-1:1 reproduce 아님 노출.

**falsifier**: REFUTED 시 candidate token 구성 ablation H 우선순위 (별도 H).

## Frame 3 — "Smoke envelope 의 noise floor 안에 묻힐 수도"

H001–H006 의 모든 single mutation 이 smoke envelope (1 epoch × 5%) 에서 noise floor (±0.01~±0.02pt platform) 안에 묻힘. 본 H 의 candidate-aware mechanism 도 같은 envelope 에서 detectable lift 못 만들 가능성.

**구체적 risk**:
- 5%-data 47k rows 로 cross-attention layer (~50K 신규 params) 학습 어려움.
- 1 epoch 학습으로 attention pattern 이 candidate-relevant 하게 fitted 못 함.
- noise floor 안에서 REFUTED → 본 H 의 진가 미측정 (paper claim true vs envelope underpowered 구별 불가).

**mitigation**:
- 본 H smoke 1차, REFUTED 또는 marginal 시 extended envelope (train_ratio=0.3, num_epochs=3 이상) retry — 단 H006 의 4시간 cost 학습. budget 확인 후.
- §17.5 sample-scale 룰: smoke 결과는 mutation 효과 measurement 용도, anchor 결정 용도 아님.
- 본 H 결과 + extended retry 양쪽 패턴으로 mechanism 가치 결정.

**falsifier**: smoke 또는 extended 어느 한 쪽에서 +0.5pt 이상 → mechanism 작동. 양쪽 모두 noise → candidate-aware 자체가 우리 데이터에 안 맞음.

---

§10.1 충족: 반대 프레임 3개 (margin lift 약함, candidate token 구성, smoke noise floor). 모두 falsifier + mitigation 명시.

§17.4 카테고리 rotation: 첫 `target_attention` 적용. 정당화 불필요.

§10.4 external_inspirations 의무: 본 H 미충족 (target_attention 별도 카테고리). H008+ carry-forward (Switch Transformer load balance loss 등).
