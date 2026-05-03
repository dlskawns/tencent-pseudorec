# H013 — Problem Statement

## What we're trying to explain

7개 H (H006~H012) 누적 Platform AUC ceiling **0.82~0.8408**. 같은 anchor
(H010 0.8408) 위 4가지 mechanism 시도:
- H010: NS→S xattn → PASS additive (champion).
- H011: input-stage aligned encoding → REFUTED degraded.
- H012: NS-token MoE 4-expert → REFUTED Frame B (uniform routing).

H012 F-2 가 새로 노출한 measurement bias:
- 사용자 `batch_size=2048` override (run.sh default 256 의 8×).
- `lr=1e-4` default 유지 → linear scaling rule (Goyal et al. 2017) 미적용.
- **Effective lr 1/8 underpowered**. 모든 H 같은 regime.

**측정 가능한 gap**: ceiling 0.82~0.8408 의 정체가
- (가설 A) **mechanism ceiling 진짜** — 다음 H 들이 different axis (long-
  seq, cohort drift) 가야 함.
- (가설 B) **hyperparameter artifact** — 모든 prior H 가 1/8 lr 로
  underpowered. 적정 lr 시 절대 lift 가 더 클 가능성.

H013 = 두 가설 분리 측정. H010 mechanism + envelope byte-identical, run.sh
의 hyperparameter 만 calibrated. Δ vs H010 anchor 결과로 ceiling 정체 결정.

## Why now

직전 H carry-forward:
- H012 F-2 (hyperparameter bias 노출) — H013 의 핵심 trigger.
- H012 F-5 (IO bound 신호) — wall 학습 −24%, inference −54%. data loading
  도 같이 calibrate.
- H010 F-1 (NS-only enrichment safe pattern) — 본 H 는 mechanism 변경 0,
  hyperparameter 만 → 위치 충돌 위험 0.
- H011 F-5 + H012 누적 → cohort drift hard ceiling 가설. 단 hyperparameter
  먼저 분리해야 cohort 한계 진짜인지 결정 가능.

§10.7 rotation: 직전 2 H = H011 feature_engineering + H012 multi_domain_fusion.
**H013 은 mechanism 변경 0 → primary_category 없음 (measurement H)**. rotation
룰 미발동.

§17.2 룰 위반 우려 (parametric mutation): justified by **measurement integrity
check**. 정당화는 challengers.md + transfer.md 에 명시.

## Scope

- **In**:
  - `--batch_size 2048` 명시 bake (사용자 override 흔적 metrics.json 에).
  - `--lr 1e-4 → 8e-4` (linear scaling for 8× batch).
  - `--num_workers 2 → 4` (IO bound 완화, Taiji deadlock 위험 8 미만).
  - `--buffer_batches 4 → 8` (큰 배치 IO 부하 완화).
  - H010 mechanism (NS→S xattn) + H008 mechanism (DCN-V2 fusion) byte-identical.
  - 코드 (model.py, train.py, infer.py, trainer.py) byte-identical — run.sh
    만 수정.
- **Out**:
  - sparse_lr 변경 안 함 (Adagrad 가 per-feature adaptive 라 linear scaling
    rule 직접 적용 안 됨, 0.05 유지).
  - num_workers 8 이상 변경 안 함 (Taiji historic deadlock 사례).
  - 다른 mechanism 추가 안 함 (single concern: training efficiency).
  - seed 변경 안 함 (paired Δ 정합성).

## UNI-REC axes

- **Sequential**: 변경 없음 (H010 NS xattn + per-domain encoder 그대로).
- **Interaction**: 변경 없음 (H008 DCN-V2 fusion 그대로).
- **Bridging mechanism**: 변경 없음. 본 H 는 mechanism이 아닌 measurement.

## Success / Failure conditions

- **Success — ceiling artifact confirmed**:
  - Δ vs H010 ≥ +0.005pt (strong) → ceiling 은 hyperparameter artifact.
    모든 prior H 의 paired Δ 재해석 의무. H010+ ranking 변경 가능.
  - Δ ∈ [+0.001, +0.005pt] (measurable) → 부분적 hyperparameter ceiling.
- **Success — mechanism ceiling confirmed**:
  - Δ vs H010 ∈ (−0.001, +0.001pt] (noise) → lr 적정, mechanism ceiling 진짜.
    Track B (long-seq P2) 또는 cohort drift 처리 진행 정당화.
- **Failure (REFUTED)**:
  - Δ < −0.001pt (degraded) → lr 8e-4 너무 큼 → divergence/instability. sub-H
    lr 4e-4.
  - NaN abort → lr 8e-4 너무 큼.
  - infrastructure regression (P4 fail).

## Frozen facts referenced

- CLAUDE.md §17.2 (one-mutation rule) — 본 H 의 정당화 논의.
- CLAUDE.md §17.3 (binary success threshold).
- H010 verdict.md — anchor 0.8408, NS xattn entropy 0.81.
- H011 verdict.md F-1, F-5 — input-stage 위험 + cohort drift 가설.
- H012 verdict.md F-1~F-3 — Frame B + 4-layer ceiling diagnosis + measurement bias.
- Linear scaling rule (Goyal et al. 2017, arXiv:1706.02677): batch K× → lr K×.

## Inheritance from prior H

- H010 anchor (Platform 0.8408) — control, **byte-identical 유지**.
- H012 F-2 (hyperparameter bias) — H013 의 핵심 trigger.
- H012 F-5 (IO bound 신호) — data loading 4/8 calibration.
- H010~H012 모두 batch=2048 + lr=1e-4 regime → paired Δ 비교 base 가 같음.
  H013 결과로 그 base 의 정합성 검증.
