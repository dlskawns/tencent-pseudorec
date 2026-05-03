# H013 — Method Transfer

> **Measurement H — no mechanism transfer**. 본 §는 H013 의 hyperparameter
> calibration 의 근거 문헌 + UNI-REC alignment degenerate case (mechanism
> 변경 없음) 명시.

## ① Source

- **Goyal et al. 2017** (arXiv:1706.02677) — "Accurate, Large Minibatch SGD:
  Training ImageNet in 1 Hour". **Linear scaling rule** 의 canonical
  reference.
- **Smith et al. 2018** (arXiv:1711.00489) — "Don't Decay the Learning
  Rate, Increase the Batch Size". batch-lr equivalence.
- **Production CTR system best practices** (Tencent / Meta / Google) —
  num_workers / prefetch_factor / pin_memory tuning for high-throughput
  training.
- 카테고리 family: 신규 카테고리 없음. **measurement H** — CLAUDE.md §X
  에 "calibration H" 분류 추가 후보 (H013 verdict 결과 따라).

## ② Original mechanism (linear scaling rule)

**Goyal et al. 2017** (1단락 재서술):

batch size B 에서 learning rate η 가 합리적으로 학습 수렴한다고 할 때,
batch 를 K× 늘려 KB 로 확장하면, learning rate 도 K× 늘려 Kη 로 동시
scaling 해야 같은 학습 dynamics 유지. 이는 SGD 의 stochastic gradient
noise 가 batch size 에 inversely 비례하기 때문 — batch K× = noise 1/K →
larger lr 이 같은 effective step size 만들기 위함. Warmup (초반 K-step 에
linear lr ramp-up) 권장 (initial instability 회피).

**우리 적용**:
- baseline batch=256, lr=1e-4 (PCVRHyFormer organizer default).
- 사용자 batch=2048 (K=8), lr 동일 → effective lr 1/8 (or noise 1/8).
- Linear scaling: lr=1e-4 × 8 = **8e-4**.
- Warmup 미적용 (현재 trainer.py 에 없음, sub-H 후보).

## ③ What we adopt

- **Mechanism class**: 없음 (model 변경 0).
- **변경 내용**: run.sh 4 hyperparameters:
  - `--batch_size 2048` (사용자 override 명시 bake — metrics.json 의
    config_sha256 sanity gate 정합성).
  - `--lr 8e-4` (linear scaling rule).
  - `--num_workers 4` (Taiji 안전 범위 — historical deadlock 위험 8 미만).
  - `--buffer_batches 8` (큰 배치 IO 부하 완화).
- **CLI flag**: 모두 train.py 의 기존 argparse 사용 (코드 변경 0).
- **PCVRHyFormer constructor**: 변경 없음.

## ④ What we modify (NOT a clone)

- **Goyal et al. 의 ImageNet 검증 → CTR 도메인 transfer**: Linear scaling
  rule 이 image classification 에서 검증됨. CTR 모델 (sparse embedding
  + dense MLP + transformer) 에서 같은 룰 적용 가능성 확인 안 됨. **risk**:
  sparse_lr (Adagrad) 의 per-feature adaptive 가 linear scaling 와 다른
  scaling rule 따를 수 있음 — 본 H 는 sparse_lr=0.05 유지 (변경 안 함),
  dense_lr 만 calibrate.
- **Warmup 미적용**: Goyal et al. 권장 warmup 안 함 — H013 단순화. NaN
  abort 시 sub-H (warmup 추가) 후보.
- **§17.2 단일 mutation 원칙 정밀 적용**: 4 hyperparameter 변경이지만 **single
  concern (training efficiency under batch 2048)**. Linear scaling rule
  의 mathematical relationship 으로 lr 와 batch 가 묶임 (한 변수만 변경
  하는 것이 incoherent). num_workers / buffer_batches 도 batch 2048 전제
  하 IO 완화 — single concern 일관성.

## ⑤ UNI-REC alignment (degenerate case)

- **Sequential reference**: H010 NS xattn (OneTrans NS→S) byte-identical 유지.
- **Interaction reference**: H008 DCN-V2 fusion byte-identical 유지.
- **Bridging mechanism**: 변경 없음 (mechanism mutation 0).
- **primary_category**: **없음** (measurement H, mechanism 미적용).
  CLAUDE.md §X 의 calibration H 카테고리 신규 검토 후보.
- **Innovation axis**: 없음. 본 H 의 가치는 mechanism 이 아닌 **measurement
  base 정합성 검증**.

## ⑥ Sample-scale viability (Rule UB-1, §10.6)

- 추가 trainable params: **0** (model byte-identical).
- §10.6 cap: anchor envelope 동일.
- Sample-scale risk:
  - **lr 8e-4 stability**: sample-scale 1000 rows 가 아닌 본 학습 (sub-sample
    30% × full data) 에서 batch 2048 + lr 8e-4 가 일반적인 stable 범위.
    NaN 위험 있으나 H010 같은 model 에서 historic precedent 없음.
  - **batch 2048 자체**: 큰 batch 가 generalization 면에서 sub-optimal 위험
    (Smith et al. 2018) — 단 본 H 는 batch 256 복귀 안 함 (사용자 override
    유지). batch 256 sub-H 별도.

## ⑦ Carry-forward rules to honor

- **§10.5 LayerNorm on x₀**: 변경 없음.
- **§10.6 sample budget cap**: 변경 없음 (params 추가 0).
- **§10.7 카테고리 rotation**: H013 primary_category 없음 → rotation 룰
  미발동. 다음 mechanism H (H014) 시 직전 2 mechanism H = H011
  feature_engineering + H012 multi_domain_fusion 차단.
- **§10.9 OneTrans softmax-attention entropy**: H010 NS xattn entropy
  threshold 0.95 × log(384) ≈ 5.65 그대로. 결과 metrics.json `attn_entropy_per_layer`
  비교 — H010 [0.81, 0.81] 과 다른가 측정 (proper lr 시 selective routing
  변화 가능).
- **§10.10 InterFormer bridge gating σ(−2)**: 미적용 (bridge/gate 추가 없음).
- **§17.2 one-mutation**: parametric mutation, **measurement integrity**
  로 정당화 (challengers.md §재진입정당화).
- **§17.3 binary success**: Δ ≥ +0.001pt (sample-scale relaxed) 또는
  +0.005pt (strong).
- **§17.4 카테고리 rotation 재진입 정당화**: 미발동.
- **§17.5 sample-scale = code-path verification only**: 본 H 는 mechanism
  변경 0 → sample-scale code-path 변경 없음. cloud full-data 결과로만 결정.
- **§17.6 cost cap**: extended ~3시간, T2 안. 누적 ~27시간 (H006~H013).
  cap 임박.
- **§17.7 falsification-first**: predictions.md 에 strong / measurable /
  noise / degraded 분기 명시.
- **§17.8 cloud handoff discipline**: training_request.md + flat upload.
- **§18 inference 인프라 룰**: 변경 없음.
- **H010 F-1**: NS-only enrichment safe pattern → 본 H 도 mechanism 변경
  0 → 안전.
- **H012 F-2 (hyperparameter bias)** → 본 H 의 핵심 trigger.
- **H012 F-5 (IO bound 신호)** → num_workers 2→4, buffer_batches 4→8 같이
  calibrate.
- **eda/out/* 검증값** — schema / fid 정의 모두 H010 패키지 inherit.
