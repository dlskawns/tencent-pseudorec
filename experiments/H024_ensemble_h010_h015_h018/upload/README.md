# H024 — ensemble_h010_h015_h018 (LOCAL ensemble, no cloud retrain)

> Measurement H — prediction-level ensemble of 3 trained models. NO model
> retrain. NO cloud training. LOCAL Python script (`ensemble.py`) takes 3
> predictions.json files (from H010 + H015 + H018 inference) and writes 1
> ensembled predictions.json. User uploads ensembled file to platform for
> scoring.
>
> Cost: $0 (local CPU < 1 minute) + 1 platform submission. The 3 source
> predictions come from cloud H010/H015/H018 runs that already ran (or will
> run independently as part of Subset B).

## What this is

H024 = **measurement H, no mutation, no retrain**. Average 3 trained
models' platform-eval predictions:
```
ensemble_prob[user] = (w_H010 * p_H010[user] + w_H015 * p_H015[user] + w_H018 * p_H018[user]) / (w_H010 + w_H015 + w_H018)
```

Default uniform weights (1.0 each). Optional weighted ensemble via
`--weights` flag.

## What you need (3 predictions.json files)

1. **H010 predictions.json** — from H010 corrected re-inference (the run
   that produced platform 0.837806). User should have this saved already.
2. **H015 predictions.json** — from H015 cloud run (platform 0.83805).
3. **H018 predictions.json** — from H018 cloud run (just launched, in
   parallel with H022/H023). After H018 done.

평소엔 platform 이 predictions.json 직접 user 에 안 노출. **Taiji 의 infer
output 회수 방법** 사용자 확인 필요. 아예 access 못 하면 H024 launch 불가
(이 경우 H024 retire, paradigm shift 다른 form 우선).

## Usage

```bash
python ensemble.py \
    --h010-preds /path/H010_predictions.json \
    --h015-preds /path/H015_predictions.json \
    --h018-preds /path/H018_predictions.json \
    --output     /path/H024_ensemble_predictions.json \
    --weights    1.0 1.0 1.0
```

Local Python (.venv-arm64 권장):
```bash
.venv-arm64/bin/python ensemble.py ...
```

Output: 1 ensembled predictions.json. Upload to platform → 채점 결과 회수.

## Sanity checks (script 가 자동)

1. **Row count match**: H010/H015/H018 의 N (user count) 동일.
2. **Key set match**: 3 dicts 의 user_id key set 동일.
3. **Prob range**: [0, 1] sanity (logit scale 면 WARN).
4. **Prob distribution**: ensembled mean/std/min/max print.

## Falsification (post-platform-score)

| platform AUC (ensemble) | classification | mechanism implication |
|---|---|---|
| ≥ max(H010, H015) + 0.005pt | strong | ensemble 가 single-model ceiling 위 lift, variance reduction signal 검증, H020+ 모든 H 의 ensemble candidate |
| ∈ [max + 0.001, max + 0.005pt] | measurable | 약 lift, ensemble 가치 marginal |
| ≈ max ± 0.001pt | noise | ensemble 무 effect (3 model 이 같은 cohort 구간 fit) |
| < max | degraded | weighted ensemble sub-H (H010 dominant or different weights) |

**현재 max(H010 corrected 0.837806, H015 corrected 0.83805) ≈ 0.8381**.
H024 ≥ 0.843 → strong (paradigm shift 대안). 0.838~0.843 → measurable.

## Decision tree (post-result)

- **strong**: anchor = H024 ensemble. H020+ 모든 H multi-model ensemble
  default. cost ~3× per H (3 models 학습).
- **measurable**: ensemble 가치 marginal — single-model attempt (H019
  TWIN 등) 우선.
- **noise/degraded**: ensemble retire — 3 model 이 같은 ceiling 영역 fit
  (cohort drift hard ceiling 추가 confirm).

## §17.2 / §17.4 / §17.6

- **§17.2 EXEMPT**: measurement H, no model mutation.
- **§17.4 rotation**: `measurement` re-entry (H022/H023 sibling, prediction-
  level form variant). 정당화: H023 가 single-model variance 측정,
  H024 가 multi-model ensemble — directly different mechanism class within
  measurement category.
- **§17.6 cost**: $0 (local) + 1 platform submission. 거의 free.

## What's NOT a clone

본 H 는 **production CTR ensemble 의 1:1 재현 아님**:
- production = N model (8~16) 의 GBDT 또는 stacking. 본 H = simple
  arithmetic mean over 3 model.
- production = learnable ensemble weights (validation set tuning). 본 H
  = uniform default + manual weighted sub-H.
- production = different architectures (NN + GBDT + linear). 본 H = same
  PCVRHyFormer 3 variant.

## Cloud package note

이 폴더는 cloud upload 가 아닌 **local 실행 도구**:
- `ensemble.py` — main script (CPU only, no GPU)
- `README.md` — 본 파일

평소 H 의 cloud upload package (run.sh + train.py + 등) 같은 구조 아님.
사용자가 local 에서 ensemble.py 실행 → output 을 platform 에 직접 upload.
