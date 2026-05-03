# H014 — Problem Statement

## What we're trying to explain

8 H 누적 (H006~H013) Platform AUC ceiling 0.82~0.8408. 4-layer ceiling
diagnosis 의 마지막 unexplored axis = **L4 truncate 정보 손실**.

§3.5 데이터 사실 (verified, sibling cite):
- domain a: p50=577.5, p90=1562.1, max=1888.
- domain b: p50=405.0, p90=1393.0, max=1952.
- domain c: p50=322.0, p90=887.3, max=3894.
- domain d: p50=1035.5, p90=2215.3, max=3951, frac_empty=8%.

현재 baseline (H010 anchor + 모든 prior H): `seq_max_lens "seq_a:64, seq_b:64,
seq_c:128, seq_d:128"`.

→ 모든 도메인 p90 ≫ 100. **truncate ratio = 4-13%** (각 도메인 p90 대비).
즉 평균 user 의 history **95%+ 정보 손실**. 8 H 모두 이 truncate 위에서
측정 — truncate 자체의 영향 한 번도 분리 측정 안 됨.

**측정 가능한 gap**: H014 = `seq_max_lens 256-512` (4× expansion). H010
mechanism + envelope byte-identical 외 단일 mutation. Δ vs H010 결과로
L4 가설 검증.

## Why now

직전 H carry-forward:
- H013 verdict (Frame A REFUTED): hyperparameter L1 retire confirmed.
- H011/H012 Frame B (NS xattn 이 dominant signal sparse capture): NS-token
  level mechanism axis L3 retire.
- §10.3 룰 trigger: H011/H012/H013 = 3회 연속 H010 anchor 위 mutation REFUTED.
  paradigm 이동 의무.
- L4 가 §3.5 의 가장 강한 정량 motivation 보유 (95%+ 정보 손실).

§10.7 rotation: H013 (measurement, no category) 이전 H012 multi_domain_fusion
+ H011 feature_engineering. **H014 = envelope mutation (no mechanism category)
또는 신규 카테고리 `long_seq_retrieval/` (P2 phase entry 후보)**. category
충돌 없음.

§17.2 룰 분석: envelope mutation (data shape 변경, mechanism 0). "structural"
인정 가능 (input pipeline 변경). parametric 은 아님.

§0 north star alignment: **sequential axis 강화** (UNI-REC 의 두 축 중
sequence side). 95%+ 정보 손실 회복 = 가장 큰 sequence axis lever.

## Scope

- **In**:
  - `seq_max_lens "seq_a:64,seq_b:64,seq_c:128,seq_d:128" → "seq_a:256,
    seq_b:256,seq_c:512,seq_d:512"` (4× per domain).
  - 코드 변경 0 (model.py / train.py / infer.py / trainer.py / dataset.py
    byte-identical with H010).
  - run.sh + README.md 만 변경.
  - H010 mechanism (NS xattn) + H008 mechanism (DCN-V2 fusion) 그대로.
- **Out**:
  - 4× 보다 큰 expansion (예: 1024) — sub-H, OOM risk 높음.
  - retrieval/compression mechanism (TWIN/SIM/HSTU) — H014 가 PASS 하면
    H015 후보.
  - mechanism 추가 안 함 (single mutation 정신 유지).
  - batch_size/lr 변경 안 함 (H013 의 L1 axis 와 분리).

## UNI-REC axes

- **Sequential**: 강화 (sequence length 4×). per-domain encoder 가 더 많은
  history 보게 됨 → user representation richer.
- **Interaction**: H008 DCN-V2 fusion 그대로.
- **Bridging mechanism**: H010 NS xattn 그대로 (단 K=V 의 S concat 길이
  4× 증가, attention 에서 더 풍부한 candidate 후보).

## Success / Failure conditions

- **Success — L4 confirmed**:
  - Δ vs H010 anchor (0.8408) ≥ +0.005pt (strong) → Platform ≥ 0.8458.
    truncate 정보 손실이 진짜 ceiling. P2 phase (TWIN/SIM/HSTU) 정당.
  - 또는 measurable [+0.001, +0.005pt] → partial confirmed.
- **Failure (REFUTED)**:
  - noise (−0.001, +0.001pt] → L4 도 ceiling 아님. paradigm shift inevitable.
  - degraded < −0.001pt → 긴 seq noise 추가, 학습 더 어려움.
  - OOM → sub-H (256 uniform 또는 batch 256 복귀).

## Frozen facts referenced

- §3.5 sequence length 분포 (verified, sibling cite).
- §10.3 challenger rule trigger (3회 연속 mutation REFUTED).
- H010 anchor (Platform 0.8408).
- H011/H012/H013 verdicts → L1/L3 retire.
- run.sh + train.py 의 `--seq_max_lens` argparse (string parser).

## Inheritance from prior H

- H010 anchor + mechanism byte-identical.
- H013 F-1 (Frame A REFUTED): L1 retire → L4 가 더 priority 높아짐.
- H012 F-3 (4-layer ceiling diagnosis) → L4 가 마지막 unexplored axis
  명시.
- H011 F-5 / H012 F-3 (cohort drift hard ceiling 가설 L2) → H014 결과 후
  L2 까지 retire 시 paradigm shift inevitable.
