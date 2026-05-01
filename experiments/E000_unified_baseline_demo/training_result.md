# Training Result — E000_unified_baseline_demo

> 사용자가 cloud 학습 종료 후 채워서 paste. 어시스턴트가 `cloud-intake` skill로 INDEX/verdict/progress 자동 갱신.

## Run metadata (3 seeds)

### seed=42
- submitted_at: {YYYY-MM-DDThh:mm:ssZ}
- platform: {Tencent Cloud / Modal / Lambda / RunPod / Colab Pro}
- gpu: {A10G / A100-40GB / etc.}
- wall_time: {hh:mm:ss}
- cost_usd: {실측 또는 추정}

### seed=1337
- (동일 양식)

### seed=2026
- (동일 양식)

## metrics.json (verbatim paste, 3개 seed 모두)

### seed=42
```json
{paste metrics.json verbatim}
```

### seed=1337
```json
{paste metrics.json verbatim}
```

### seed=2026
```json
{paste metrics.json verbatim}
```

## Best ckpt path
- seed=42: {플랫폼 위 절대경로 또는 다운로드 링크}
- seed=1337: {...}
- seed=2026: {...}

## Last 30–50 lines of train.log (대표 1개 seed면 충분)
```
{여기에 paste}
```

## Falsification check
| P | 임계 | seed=42 | seed=1337 | seed=2026 | verdict |
|---|---|---|---|---|---|
| P1 | NaN-free, finite val_AUC | __ | __ | __ | __ |
| P2 | preds stdev ≥ 0.01 | __ | __ | __ | __ |
| P3 | overfit gap ≥ 0.05 | __ | __ | __ | __ |
| P4 | local_validate 5/5 | __ | __ | __ | __ |
| P5 | OOF ≥ 0.50 | __ | __ | __ | __ |
| P6 | seed×3 OOF stddev ≤ 0.05 | — | — | __ | __ |
| P7 | seed=42 OOF in local±0.10 | __ | — | — | __ |

## Notes / anomalies
- {플랫폼 특이사항}
- {OOM, NCCL, 재시도 발생 여부}
- {predictions 분포 (mean/std/min/max)}
- {본 사용자가 셋업하면서 발견한 packet 개선 제안}

## Files downloaded
- [ ] ckpt_seed42/metrics.json
- [ ] ckpt_seed1337/metrics.json
- [ ] ckpt_seed2026/metrics.json
- [ ] ckpt_seed42/global_step*.best_model/{model.pt, schema.json, train_config.json}
- [ ] logs/train.log (last 50 lines)
- [ ] work/_split_meta.json

## Optional: 사용자 추가 측정
- predictions stdev (test data 기준): ___
- max attention entropy per layer: ___
- (etc.)

## Next-action signals (intake가 자동 작성)
- treatment-vs-control paired Δ: TBD (E000은 anchor 자체 — control 없음)
- verdict 권장 status: TBD (intake 후 결정)
- 다음 H 후보: TBD (rotation: H001 = unified_backbones → H002는 다른 카테고리)
