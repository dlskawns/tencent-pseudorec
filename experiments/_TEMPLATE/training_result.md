# Training Result — {EXP_ID}

> 클라우드 학습 종료 후 사용자가 채우는 양식. 채운 내용 paste 하면 `cloud-intake` skill이 INDEX/verdict/progress 자동 갱신.

## Run metadata
- submitted_at: {YYYY-MM-DDThh:mm:ssZ UTC}
- platform: {Tencent Cloud Notebook / Colab Pro / Lambda / RunPod / Modal / Kaggle / 기타}
- gpu: {예: A100-40GB SXM, T4-16GB, ...}
- wall_time: {hh:mm:ss}
- cost_usd: {실측 또는 추정}
- queue_wait_min: {플랫폼 대기 큐 시간, 있으면}

## metrics.json (verbatim paste)
```json
{여기에 metrics.json 내용 그대로}
```

## Best ckpt path
{플랫폼 위 절대경로 또는 다운로드 링크}

## Last 30–50 lines of train.log
```
{여기에 paste}
```

## Falsification check (card.yaml 임계치 적용)
- P1 (NaN-free, finite best_val_AUC): pass / fail
- P2 (predictions stdev > 0.01): pass / fail / N/A
- P3 (overfit gap): {value} → pass / fail / inconclusive
- P4 (local_validate.py 5/5): pass / fail / N/A (다운로드 후 로컬 재검증)
- P5 (OOF threshold): {value} → pass / fail

## Notes / anomalies
- {OOM, NCCL crash, 재시도, 큐 대기 등}
- {predictions 분포가 이상하지 않은지: mean/std/min/max paste}
- {training cost가 예상 대비 어땠는지}

## Files downloaded (체크박스)
- [ ] metrics.json
- [ ] best_model/model.pt
- [ ] best_model/schema.json
- [ ] best_model/train_config.json
- [ ] train.log (last N lines)
- [ ] work/_split_meta.json (해당 시)

## Optional: 직접 측정한 추가 지표
- predictions stdev (test data 기준): ___
- max attention entropy (per layer): ___
- (etc.)

## Next-action signals (intake가 자동 작성, 사용자는 비워둠)
- treatment-vs-control paired Δ: TBD
- verdict 권장 status: TBD
- 다음 H 후보 (rotation 충족): TBD
