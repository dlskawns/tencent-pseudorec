# H015 — Method Transfer

## ① Source

- **Sample weighting in domain adaptation** (Pan & Yang 2010 survey, "A
  Survey on Transfer Learning"): source-target distribution 차이 mitigation
  의 canonical 패턴.
- **Concept drift 처리** (Gama et al. 2014, "A Survey on Concept Drift
  Adaptation"): online learning 의 recency-based weighting.
- **Production CTR engineering** (Meta / Google / Tencent): time-decay
  loss weighting 이 standard practice (논문 부재, production know-how).
- **Importance weighting for distribution shift** (Sugiyama et al. 2007,
  "Covariate Shift Adaptation by Importance Weighted Cross Validation"):
  importance ratio 기반 weighting 의 이론적 근거.
- 카테고리 family (`temporal_cohort/`): recency weighting / temporal
  embedding / OOF 재정의 / domain adaptation. **신규 카테고리 first-touch**.

## ② Original mechanism

**Recency-aware loss weighting** (1단락 재서술):

학습 sample 마다 timestamp / label_time 기반 weight 부여. 더 recent sample
이 더 큰 weight → loss gradient 에 더 큰 영향 → 모델이 recent pattern 에
더 fit. 핵심 가정: train 과 eval 의 distribution 가 시간에 따라 drift,
recent train sample 이 eval distribution 에 closer. weight 함수: linear
decay / exponential decay / step function. 가장 단순 form: linear scaling
[w_min, w_max] by label_time percentile, mean weight = (w_min + w_max) / 2 (loss
scale 보존 가능).

**우리 적용**:
- Per-batch label_time min-max → percentile [0, 1].
- Linear weight = `recency_weight_min + (max - min) × percentile`.
- Default `[0.5, 1.5]` → mean = 1.0 (loss scale 보존, lr/optim 영향 없음).

## ③ What we adopt

- **Mechanism class**: per-batch linear recency loss weighting. minimum
  viable form.
- **변경 내용 (3 files + run.sh)**:
  1. `dataset.py`: `_convert_batch` 가 batch dict 에 `label_time` 노출
     (2 줄).
  2. `trainer.py.__init__`: 3 args (`use_recency_loss_weighting`,
     `recency_weight_min`, `recency_weight_max`).
  3. `trainer.py._train_step`: weighting branch (~20 줄). bce/focal loss
     reduction='none' + weighted mean.
  4. `train.py`: argparse 3 + Trainer 3 keys.
  5. `run.sh`: 3 H015 flags + 2 H010 default 명시 bake.
- **CLI**: `--use_recency_loss_weighting --recency_weight_min 0.5
  --recency_weight_max 1.5`.

## ④ What we modify (NOT a clone)

- **Per-batch (not per-dataset)**: production 표준은 per-dataset (전체 train
  min/max 미리 계산). 본 H 는 per-batch — implementation 단순. shuffle 영향
  없음. **단점**: batch 내 label_time spread 작으면 weight 거의 1.0 일관.
  PASS 시 sub-H 로 per-dataset.
- **Linear (not exponential decay)**: production 표준은 exp decay. 본 H 는
  linear — minimum viable form, hyperparameter 적음. PASS 시 sub-H 로 exp.
- **Mean weight = 1.0 보존**: linear [0.5, 1.5] symmetric around 1.0.
  loss scale 보존 → lr/optim 영향 0 → paired Δ 비교 confound 작음.
- **§17.2 single mutation**: loss weighting 만. mechanism stack (NS xattn
  + DCN-V2) byte-identical.

## ⑤ UNI-REC alignment (HARD)

- **Sequential reference**: 변경 없음 (H010 NS xattn + per-domain encoder
  그대로).
- **Interaction reference**: 변경 없음 (H008 DCN-V2 fusion 그대로).
- **Bridging mechanism**: 변경 없음.
- **Training procedure**: NEW axis (UNI-REC 안 새 layer). production CTR
  시스템의 cohort handling 표준 (deployment realism).
- **primary_category**: `temporal_cohort` (신규).
- **Innovation axis**: 9 H 누적 OOF stable / Platform 변동 패턴 직접 attack.
  paradigm 안 마지막 카테고리 시도 (4-layer L2 검증).

## ⑥ Sample-scale viability (Rule UB-1, §10.6)

- 추가 trainable params: **0** (loss 가중치만, model byte-identical).
- §10.6 cap: 위반 없음.
- Sample-scale risk:
  - **Loss scale 변동**: mean weight = 1.0 보존 → 영향 없음.
  - **Per-batch label_time spread 작을 위험**: batch 2048 + train_ratio
    0.3 = sample 많음, label_time 분포 충분히 넓음 (split label_time cutoff
    이전). spread 충분 가정.

## ⑦ Carry-forward rules to honor

- **§10.5 LayerNorm on x₀**: 변경 없음.
- **§10.6 sample budget cap**: 변경 없음 (params 추가 0).
- **§10.7 카테고리 rotation**: `temporal_cohort` 신규 first-touch.
- **§10.9 OneTrans softmax-attention entropy**: 변경 없음 (H010 NS xattn
  threshold 5.65 그대로).
- **§10.10 InterFormer bridge gating σ(−2)**: 미적용.
- **§17.2 one-mutation**: loss weighting only. mechanism byte-identical.
- **§17.3 binary success**: Δ ≥ +0.001pt (sample-scale relaxed) 또는
  +0.005pt (strong).
- **§17.4 카테고리 rotation 재진입 정당화**: 미발동.
- **§17.5 sample-scale = code-path verification only**.
- **§17.6 cost cap**: extended ~3-4h (H010 envelope 동일). 누적 ~36h.
  cap 위협.
- **§17.7 falsification-first**: predictions.md 에 strong / measurable /
  noise / degraded 분기 + Frame B (paradigm shift mandatory) trigger 명시.
- **§17.8 cloud handoff discipline**: training_request.md + flat upload.
- **§18 inference 인프라 룰**: 변경 없음 (infer.py byte-identical).
- **H010 F-1**: NS-only enrichment safe pattern → H015 도 mechanism 변경
  0 → 안전.
- **H011~H014 누적 carry-forward**: cohort drift hard ceiling 가설 → H015
  의 핵심 동기.
