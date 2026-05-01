# H008 — Challengers

> §10.1 — 모든 H 시작 전 반대 프레임 ≥ 2.

## Frame 1 — "RankMixer 가 이미 token-mixing 으로 feature interaction 표현 충분할 수도"

PCVRHyFormer 의 `RankMixerBlock` 이 token-mixing MLP — Linear projections 로 token 축 mix. 이건 **implicit 한 feature interaction** 의 한 형태. paper (RankMixer / MLP-Mixer 계열) 에선 충분히 표현력 있다고 주장. DCN-V2 의 explicit polynomial cross 가 추가 lift 안 줄 수도.

**구체적 risk**:
- RankMixer 가 Linear 와 활성화 함수 (SiLU) 를 통해 우리 데이터의 feature interaction 을 이미 충분히 capture.
- DCN-V2 의 polynomial cross degree 2-3 이 actually 더 expressive 한 게 아니라 다른 형태의 interaction 일 뿐 — 우리 데이터에 fit 안 할 가능성.
- low-rank approximation (rank=8) 이 d_model=64 의 1/8 → expressive capacity 한계.

**mitigation**:
- §17.3 binary 임계 +0.5pt 가 falsifier — REFUTED 시 sparse_feature_cross 카테고리 일시 archive.
- DCN-V2 layer 수 (2 → 4) + rank (8 → 16) tuning sub-H 후보.
- 다른 explicit cross (FwFM, AutoDis) 후보로 carry-forward.

**falsifier**: smoke val_AUC + Platform AUC 모두 anchor 대비 < +0.5pt → REFUTED.

## Frame 2 — "Single mutation 의 lift 가 H007 처럼 marginal 일 수도"

H001–H006 모두 noise floor 안 묻혔고 H007 만 marginal PASS. 본 H 도 같은 패턴 — single mutation 으로 +0.5pt 가 우리 envelope 의 detection floor 안에서 측정 가능한 최대치. extended envelope 에서만 lift 발현 가능성.

**구체적 risk**:
- smoke envelope (1 epoch × 5%) 의 noise floor (±0.005pt) 안에 lift 묻힘.
- DCN-V2 cross stack 학습이 1 epoch 으로 충분치 못함 (low-rank weights 수렴 미흡).
- 본 H 가 marginal 또는 noise → mechanism 자체 가치 측정 ambiguous.

**mitigation**:
- smoke 1차 → marginal/REFUTED 시 extended envelope (3 ep × 30%, H007 와 동일) retry. cost ~3시간.
- val ↔ platform 정합 (H007 F-2 carry-forward) 전제 — val 측정으로 mechanism 작동 여부 추정 가능.
- 다음 step 결정에 platform AUC 까지 받아본 후.

**falsifier**: smoke 또는 extended 어느 한 쪽에서 +0.5pt 이상 → mechanism 작동. 양쪽 모두 noise → DCN-V2 가 우리 데이터에 안 맞음.

## Frame 3 — "H007 + H008 combined 가 본질적인 next step 인데 단독 H 가 나뉘어 있음"

§0 north star 가 요구하는 건 **두 축 동시 강화**. H007 (target_attention sequence-axis) + H008 (DCN-V2 interaction-axis) 가 stack 된 combined H 가 진짜 P1 진입 가설. 본 H 단독으로 측정하는 건 §17.2 one-mutation 룰 충족이지만 §0 north star 직접 검증 부족.

**구체적 risk**:
- 본 H 단독 lift 가 H007 단독 lift 와 비슷 → combined H 에서 두 lift 가 stack 되는지 검증 안 됨.
- Sequential 진행 (H008 → combined H 후속) 으로 cost 증가 — 한 번에 combined 가는 게 더 효율.

**mitigation**:
- §17.2 one-mutation 룰 의 합리성: 두 mutation 동시 변경 시 어느 쪽이 lift 의 source 인지 확인 못 함. sequential 진행이 mechanism 분리 원칙.
- 본 H PASS 후 즉시 H009 = combined (H007 + H008) H 진행 — sub-H of H007/H008.
- 기대: H007 (+0.005pt) + H008 (+X pt) → combined = anchor + 0.005 + X. additive 가정 검증.

**falsifier**: H008 PASS 시 combined H 진행 → additive 검증. 만약 combined ≠ H007 + H008 단독합 → mechanism 간 interference 신호.

---

§10.1 충족: 반대 프레임 3개 (RankMixer 충분, single mutation marginal, combined 우선). 모두 falsifier + mitigation 명시.

§17.4 카테고리 rotation: 첫 `sparse_feature_cross` 적용. 정당화 불필요.

§10.4 external_inspirations 의무: 본 H 미충족 (sparse_feature_cross 별도 카테고리). H009+ carry-forward (Switch Transformer load balance loss).
