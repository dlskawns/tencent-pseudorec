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

## §18.8 TRAIN SUMMARY block (mandatory from H018+)

`train.py` 끝에서 emit 한 단일 SUMMARY 블록을 그대로 paste. 사용자가
복사한 stdout 의 마지막 ~15줄. 형식 변경 금지 (verify-claim 정규식
파싱 anchor).

```
==== TRAIN SUMMARY (HXXX_slug) ====
git=abc1234 cfg=def45678 seed=42 ckpt_exported=best
epoch | train_loss | val_auc | oof_auc
  1   |   0.4234   | 0.8124  | 0.8345
  2   |   0.3812   | 0.8312  | 0.8456
  ...
 12   |   0.2891   | 0.8378  | 0.8589
best=epoch7  val=0.8412  oof=0.8589
last=epoch12 val=0.8378  oof=0.8589
overfit=+0.0034 (best_val − last_val)
calib pred=0.124 label=0.124 ece=0.012
==== END SUMMARY ====
```

**Pre-H018 (legacy)**: `eval auc: 0.XXXXXX` 단일 라인 + per-epoch lines
받기. verify-claim 이 legacy 포맷도 파싱하지만 **WARN: §18.8 미준수**
표기.

## Metrics summary (computed from SUMMARY block)

| metric | value | 비고 |
|---|---|---|
| best_val_AUC | 0.XXXX | argmax(val_auc) over epoch_history |
| last_val_AUC | 0.XXXX | final epoch val_auc |
| overfit_gap | +0.XXXX | best_val − last_val (양수 = 마지막이 best 보다 안 좋음) |
| best_OOF_AUC (redefined) | 0.XXXX | H016 redefined OOF default |
| platform AUC | 0.XXXX | 채점 결과, 별도 입력 |
| **val ↔ platform gap** | +0.XXXX | best_val − platform (P5 핵심) |
| **OOF ↔ platform gap** | +0.XXXX | best_oof − platform (P6 핵심) |
| calibration ECE | 0.XXX | 10-bin ECE on val set |

## Falsification check (card.yaml 임계치 적용)
- P1 (NaN-free, finite best_val_AUC): pass / fail
- P2 (predictions stdev > 0.01): pass / fail / N/A
- P3 (overfit gap): {value} → pass / fail / inconclusive
  - 산출법: SUMMARY 블록의 `overfit=` 필드 그대로. 양수 = 마지막 epoch
    가 best 보다 안 좋음. 0.005pt 초과 시 inconclusive 검토.
- P4 (local_validate.py 5/5): pass / fail / N/A (다운로드 후 로컬 재검증)
- P5 (val ↔ platform gap): {value} → pass / fail
  - 산출법: `best_val_AUC − platform_AUC`. cut: ≤ 0.01pt = pass (val 이
    platform 잘 예측), > 0.01pt = fail (local 의사결정 신호 약함).
- P6 (OOF ↔ platform gap): {value} → pass / fail
  - 산출법: `best_OOF_AUC − platform_AUC` (redefined OOF, H016 default).
    cut: ≤ 0.005pt = pass (OOF 가 platform 가까이), > 0.005pt = WARN.

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
