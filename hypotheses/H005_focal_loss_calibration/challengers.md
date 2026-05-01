# H005 — Challengers

> §10.1 + §17.4: 모든 H 시작 전 반대 프레임 ≥ 2.

## Frame 1 — "Imbalance 12% 는 focal 이 효과 보이는 영역이 아니다"

Focal loss 는 Lin et al. (ICCV 2017) 에서 **extreme imbalance** (object detection
1:1000+ foreground:background) 영역에서 +2–4 mAP lift 검증. 우리는 12% positive
(약 1:7.5 ratio) → moderate imbalance. focal 의 "hard sample 강화" 효과가 충분히
크지 않을 가능성. CTR 영역에서도 focal 의 lift 는 데이터마다 0.0–0.3 pt 분포 (안
오르거나 미미하게 오름).

**구체적 risk**:
- BCE 가 이미 imbalanced sigmoid loss 라 12% positive 환경에서 충분히 작동.
- focal 의 modulating factor `(1−p)^γ` 가 too-easy negatives 의 gradient 만 줄여
  실제로 학습된 representation 변화 작음.
- α=0.25 가 minority-class weight 를 추가로 늘려도 prior 0.124 가 이미 sigmoid
  cross-entropy 의 implicit balancing 으로 처리됨.

**mitigation**:
- §17.3 binary 임계 +0.5 pt 가 우리의 falsifier — Δ < +0.5 pt 면 명확히 REFUTED,
  focal 방향 retire. 측정 자체가 falsifier 다.
- α/γ 튜닝은 본 H 에서 미적용 (one-mutation rule). 만약 본 H REFUTED 후 α/γ 가
  의심되면 별도 H 로 분리 (loss_calibration 카테고리 재진입 정당화 필요).

**falsifier**: smoke val_AUC < 0.8301 → REFUTED.

## Frame 2 — "Smoke 1 epoch 비교는 H004 와 동일한 unfair-comparison 함정"

H004 verdict F-2: PCVRHyFormer 는 organizer-tuned baseline → 1 epoch + 5%-data
envelope 에서 빠르게 generalize. focal loss 는 BCE 와 다른 gradient 분포 → 1 epoch
으로 BCE-tuned starting point 를 능가하기 어려울 수 있음. 본 H 가 REFUTED 되어도
"sample-scale 한계" 핑계로 정보 가치 모호.

**구체적 risk**:
- BCE → focal 전환은 첫 epoch 에서 loss landscape 변화 → optimizer (Adagrad sparse
  + AdamW dense) 의 첫 step 패턴이 달라짐. 1 epoch 에선 focal 의 hard-positive
  강화 효과가 dense param 에 학습되기 전.
- Dual optimizer 의 sparse 부분 (Adagrac) 은 first-moment accumulation 을 epoch-level
  로 함 → focal 의 효과가 sparse param 까지 전파되려면 epoch ≥ 2 필요할 수 있음.

**mitigation**:
- predictions.md negative-result interpretation 에 명시: REFUTED 의 두 가능성
  (focal 자체 무효 vs 1-epoch underpowered) 을 분리. Δ ≈ 0 (±0.1 pt) 면
  underpowered 가능성 높음 → epoch ≥ 3 retry 후보로 carry-forward. Δ < −0.3 pt 면
  focal 본격 실패.
- 비교 baseline 은 BCE 환경의 PCVRHyFormer (val=0.8251) → BCE→focal 만이 변수.
  envelope 는 byte-identical → fair pairing 보장.
- §17.5 sample-scale 룰: 본 H 의 결과는 anchor 결정용 아님, mutation 효과 측정용.
  full-data 환경에서 결과 다를 수 있음을 verdict 에 명시.

**falsifier**: Δ < +0.5 pt + 별도 진단 (logloss 변화) 으로 underpowered vs 무효
구별. underpowered 면 retry, 무효면 retire.

## Frame 3 — "loss_calibration 카테고리 자체가 P1+ 진입 가설로 약하다"

§0 north star: 모든 가설은 두 축 동시 다뤄야 (seq + int). loss_calibration 은 axis
표현 자체를 안 건드리고 학습 신호만 보정. **§0 P1 진입 조건** ("seq + interaction
이 한 블록에서 gradient 공유") 충족은 anchor (PCVRHyFormer) 가 이미 함. 본 H 는
anchor 성능 위 calibration. P1 후보 중 약한 mutation.

**구체적 risk**:
- 같은 cost 로 더 강한 §0 axis-strengthening mutation (예: longer encoder for D,
  target attention) 이 가능. 본 H 가 PASS 해도 다음 H 들이 더 큰 lift 가능성.
- 두 축이 이미 anchor 에서 작동 중이라 calibration 효과 marginal.

**mitigation**:
- §17.4 rotation 의무 + §17.2 cheap mutation 우선 룰. 본 H 는 cheapest 가능한
  rotation step. 결과 패턴 (PASS/REFUTED) 으로 다음 H 후보 정렬:
  - PASS → loss-level mutation 이 우리 데이터에 효과 있음 → 카테고리 retain,
    H006 = long_seq_retrieval (D 도메인 1100 tail).
  - REFUTED → loss-level lever 작동 안 함 → loss_calibration retire, H006 =
    target_attention 또는 longer_encoder.
- 본 H 의 비용 (smoke 1회) 이 loss-level lever 의 sample-scale 효과 측정의 floor
  → 더 비싼 mutation 들 (full-data) 전에 cheap signal 확보.

**falsifier**: §17.3 binary 임계 (Δ ≥ +0.5pt). 통과/미통과 어느 쪽이든 다음 H
의사결정에 명확한 신호.

---

§10.1 충족: 반대 프레임 3개 (효과 영역 미스, 1-epoch underpowered, axis 강화 약함).
모두 falsifier + mitigation 명시. 본 H 의 success path 는 (a) NaN-free, (b) val_AUC
≥ 0.8301 (Δ ≥ +0.5pt), (c) submission round-trip PASS, (d) logloss 악화 < 10%.

§17.4 카테고리 rotation: 직전 H001/H002/H004 모두 unified_backbones → H005 =
loss_calibration **rotation 첫 충족**. 정당화 추가 불필요.

§10.4 external_inspirations 의무: 본 H 미충족 (loss_calibration 카테고리). H006 후보로
carry-forward (Switch Transformer load balance loss 등).
