# H033 — Challengers (≥ 2 reverse frames)

## §17.2 / §17.4 정당화 — strict reading 위반 인지

**§17.2 strict reading**: 단일 mutation. H033 = 2 mutation (learnable GSU + per-domain K) → 위반.

**Re-justification (3 사유)**:
1. **Stacking H 의 본질**: 직교 sub-H 모두 PASS 후의 *통합* H 는 single-mutation 룰 의 발견 단계 외부. paired 비교 framework 의 자연스러운 종착점.
2. **Pre-build = insurance**: H020/H021 결과 회수 전 patch 미리 build → 둘 다 PASS 확인 시 즉시 cloud submit. 학습 시간 압박 대응.
3. **Conditional cloud submit**: H020/H021 결과 미PASS 시 cloud submit 보류 가능 (사용자 결정). build 비용은 0 (코드만), training 비용은 conditional.

**§17.4 rotation**: retrieval_long_seq 4회 연속 (H019/H020/H021/H033). H020/H021 의 4 사유 carry + stacking 정보 가치 추가.

---

## Frame A — "Axis 간 간섭 (H009 패턴 재발)"

**가설**: H020 (scoring 학습) 과 H021 (per-domain K) 가 같은 ESU layer 의 input 에 영향. H020 의 learnable GSU 가 K=96 의 추가 token 들 (rank 65-96) 에 다른 weight 부여 → ESU attention 분포 distort. combined < strongest single (H009 의 OOF+/Platform− 패턴).

**근거**: H009 = combined H007 (xattn) + H008 (DCN-V2) 가 strongest single (H008) 보다 못함. block-level fusion 위치 충돌.

**Falsification 조건**: H033 Δ < max(H020, H021) → Frame A confirmed.

**Frame A confirmed 시 carry-forward**: sequential apply (H020 만 적용 후 학습된 모델 위 H021 fine-tune) sub-H 또는 retire.

---

## Frame B — "둘 다 NOOP — retrieval class 전체 saturation"

**가설**: H019 의 cloud measurable PASS (+0.001868pt) 가 retrieval class 의 *유일한* lift. internal axis (scoring/quantity/capacity) 모두 saturated. H020/H021/H033 모두 noise → retrieval class 종료 강한 evidence.

**근거**: 12 H ceiling (0.832~0.836) + H019 만 +0.001~0.002 위 = TWIN module 자체의 fixed lift, internal axis 변경 효과 0.

**Falsification 조건**: H020 또는 H021 PASS → Frame B partial REFUTED. H033 PASS → Frame B fully REFUTED.

**Frame B confirmed (H020+H021+H033 모두 noise) 시 carry-forward**: retrieval class 영구 retire. ESU axis (H034) 마지막 hedge → 그것도 NOOP 면 cohort/HSTU 강제 pivot.

---

## Frame C — "Stacking 의 over-capacity (small data 환경)"

**가설**: H033 = +8K params (H020 carry). sample-scale + ceiling 안에서 추가 capacity 가 학습 instability 증폭. small data 의 over-fitting.

**근거**: H013 (lr 8e-4) Keskar large-batch generalization gap 과 비슷한 패턴 — capacity 변경이 platform 일반화 악화.

**Falsification 조건**: H033 Δ < −0.001 (degraded). overfit_gap > +0.005pt (best_val − last_val 큼).

**Frame C confirmed 시 carry-forward**: sub-H = projection dim 축소 (d_model//8) 또는 H033 의 H020 부분 retire (H021 만 유지).

---

## Counter-argument 종합

1. **Pre-build cost-effective**: 학습 비용 0 (코드 build 만). cloud submit 은 conditional.
2. **결과 정보 가치 큼**: 4 outcome 모두 다음 H 결정 명확.
3. **Axis 직교성 검증 필요**: H020/H021 모두 PASS 시 stacking 효과 측정 없으면 통합 anchor 불가.
4. **H034 (capacity axis) 와 직교**: H033 + H034 = retrieval class 의 4 axis 중 3 axis (scoring quality + quantity + capacity) 동시 검증.
