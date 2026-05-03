# H014 — Challenger Frames

> CLAUDE.md §10.1, §10.7. **Envelope mutation** (input data shape 변경,
> mechanism 0). primary_category = no mechanism category 또는 신규
> `long_seq_retrieval/` (P2 phase entry 후보).

## Frame A (default — what we're proposing)

- **Claim**: 8 H 누적 ceiling 의 진짜 원인은 truncate 64-128 의 95%+ 정보
  손실. seq_max_lens 4× 확장 시 H010 anchor 위 ≥ +0.005pt (strong) 또는
  ≥ +0.001pt (measurable) 가능.
- **Mechanism**: H010 byte-identical, run.sh `--seq_max_lens` 만 변경.
- **Why this could be wrong**:
  1. **Long-seq retrieval 없는 단순 expansion 효과 작음**: dense self-
     attention 으로 긴 seq 처리 시 attention 이 분산되어 dominant signal
     희석 가능. TWIN/SIM/HSTU 의 retrieval/compression 없이는 lift 작음.
  2. **OOM**: batch 2048 + seq 512 메모리 위험. mitigation = sub-H seq
     256 uniform 또는 batch 줄임.
  3. **Long seq overfit**: 더 많은 정보 = 더 많은 noise. cohort drift 가
     dominant 이면 truncate 늘어 OOF AUC 일관성 깨짐 (8 H OOF 0.856~0.860
     일관 패턴 변경).

## Frame B (counter-frame — L4 도 ceiling 아님)

- **Claim**: cohort drift (L2) 가 진짜 hard ceiling. 8 H OOF-Platform gap
  1.88~2.42pt 일관 패턴 = mechanism / envelope 무관하게 발현. truncate
  4× 확장해도 OOF 만 향상하고 Platform 은 ceiling 그대로.
- **What evidence would prove this?**:
  - H014 결과 OOF AUC ≥ +0.005pt 향상 + Platform AUC ≤ +0.001pt → train
    cohort 더 fit 했지만 platform 분포 transfer 안 됨.
  - 또는 OOF-Platform gap > 2.5pt → cohort drift 더 벌어짐.
- **Distinguishing experiment**: H014 가 그 자체. Δ Platform vs Δ OOF 분리.

## Frame C (orthogonal frame — paradigm shift)

- **Claim**: PCVRHyFormer baseline 자체가 이 task 에 sub-optimal. 더 적합한
  backbone (OneTrans full single-stream / HSTU trunk / InterFormer 3-arch)
  으로 교체 필요.
- **Mechanism**: H010 anchor 폐기, 새 backbone class.
- **Cost vs A**: 가장 큰 paradigm shift. H014 가 cheap diagnostic 이라
  먼저 분리. H014 noise → paradigm shift 정당화.

## Decision

**A 선택** 이유:
1. **§3.5 데이터 motivation 가장 강함**: 95%+ 정보 손실 정량 명확.
2. **Cheap diagnostic**: 코드 변경 0, run.sh 1줄, ~3-5h 학습.
3. **§17.2 envelope mutation = "structural" 인정**: input data shape 변경,
   mechanism 0. parametric 아님.
4. **§10.3 challenger rule 정합**: H011/H012/H013 mutation REFUTED 후
   다른 axis (envelope) 시도 = challenger 사고 적용.
5. **§0 north star alignment**: sequential axis 강화 = UNI-REC 두 축 중
   하나 lever.

**B/C 미루기 조건 (즉, B/C 로 돌아갈 trigger)**:
- A 결과 Δ Platform ≤ +0.001pt + OOF 향상 → Frame B confirm. H015 = cohort
  H 우선 (recency-aware loss / temporal cohort embedding).
- A 결과 Δ Platform 음수 + OOM 또는 stability issue → Frame C confirm
  preliminary. paradigm shift (backbone replacement) H015.
- A 결과 PASS → P2 phase 본격 entry. retrieval/compression (TWIN/SIM/HSTU)
  H015.

## (조건부) Re-entry justification

해당 없음. envelope mutation 은 신규 axis (data preprocessing). 직전 H
mechanism category 와 무관.

§10.3 룰 trigger 는 정합:
- H011 / H012 / H013 = 3회 연속 H010 anchor 위 mechanism + parametric mutation
  REFUTED.
- H014 = **envelope mutation (다른 axis)** → challenger 사고 적용 형태.
- §10.3 의 의도 = "같은 계열 mutation 만 반복 시 강제 challenger" → H014
  자체가 challenger.
