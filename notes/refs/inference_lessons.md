# Inference Infrastructure Hard Lessons (former §18)

> **트리거**: `infer.py`, `dataset.py`, `make_schema.py` 작성·수정·리뷰 시 본 파일 먼저 읽기.
> 작성 배경: 2026-04-28, H001–H005 무한 fail 후 정리. Taiji eval container 에서 inference 가 silent fallback / shape mismatch / heuristic 으로 빠지는 빈번한 실수. 모든 H 패키지의 `infer.py / dataset.py / make_schema.py` 는 본 §18 룰 준수 의무.

---

## §18.1 — `infer.py` 의 dataset batch_size 는 **반드시 생성자 인자**로 전달

```python
# WRONG (silent buffer/batch shape mismatch — H001~H005 가 모두 이걸로 fail)
eval_ds = PCVRParquetDataset(..., batch_size=int(cfg.get("batch_size", 256)))
eval_ds.batch_size = infer_batch_size  # buffer 재할당 안 됨!

# CORRECT
infer_batch_size = int(os.environ.get("INFER_BATCH_SIZE", "1024"))
eval_ds = PCVRParquetDataset(..., batch_size=infer_batch_size)
# (override 절대 금지)
```

**Why**: `_buf_user_int / _buf_item_int / _buf_user_dense / _buf_item_dense` 가 `__init__` 에서 batch_size 만큼 한 번 할당. 사후 override 는 reader 의 batch 크기만 바꾸고 buffer 는 그대로. 파일 경계에서 reader 가 sub-batch (e.g., 621 rows) 를 produce 하면 buffer slice `[:621]` 이 256-row buffer 전체 (256, total_dim) 를 반환 → arr (621,) → user_int[:, offset] (256,) ValueError 발생.

## §18.2 — `dataset.py` 의 `dim==1` path 는 **type-agnostic universal handler**

```python
# WRONG (pyarrow type 검사로는 list_view / 신규 variant 못 잡음)
if pa.types.is_list(col.type) or pa.types.is_large_list(col.type):
    ...
else:
    arr = col.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)  # 잘못된 가정

# CORRECT
py = col.to_pylist()
arr = np.empty(B, dtype=np.int64)
for i, v in enumerate(py):
    if v is None:
        arr[i] = 0
    elif isinstance(v, list):
        arr[i] = int(v[0]) if (v and v[0] is not None) else 0
    else:
        arr[i] = int(v)
```

**Why**: pyarrow 가 column 을 어떤 list variant (list / large_list / fixed_size_list / list_view 등) 로 만들든, `to_pylist()` + Python `isinstance(v, list)` 가 가장 robust. Schema 가 `dim=1` 로 capture 했다면 모델은 row 당 1 값만 받게 학습됨 → 첫 값 추출이 정답.

## §18.3 — `infer.py` 진단 로그 의무 (silent fallback 차단)

매 infer 시 다음 로그가 **모두 stdout 에 print(flush=True)** 되어야 함:

```python
print(f"[infer] MODEL_OUTPUT_PATH={model_output} ckpt_dir={ckpt_dir}", flush=True)
# torch path try
try:
    preds = _try_torch_inference(ckpt_dir, files)
    if preds is None:
        print("[infer] WARNING: torch path returned None (sidecar missing or schema fail)", flush=True)
except Exception as e:
    import traceback
    print(f"[infer] WARNING: torch path raised {type(e).__name__}: {e}", flush=True)
    traceback.print_exc()
    preds = None
if preds is None:
    print("[infer] FALLBACK: using heuristic prior", flush=True)
    preds = _heuristic_predictions(files, prior)
else:
    print(f"[infer] OK: torch path produced {len(preds)} predictions", flush=True)
# ... write json ...
print(f"[infer] wrote {len(preds)} predictions to {out_path}", flush=True)
```

**Why**: H001~H005 에서 silent except → fallback → 모든 user 에 prior=0.124 → platform AUC=0.5 가 무엇 때문인지 모르고 며칠 잡아먹음. 명시 로그 있으면 즉시 분리.

## §18.4 — `infer.py` 기본 최적화 defaults

| Env var | Default | 이유 |
|---|---|---|
| `INFER_BATCH_SIZE` | **1024** | training batch (256) 너무 작음. Python/DataLoader overhead amortize. cfg.get fallback **금지** (training value 가 적합하지 않음). |
| `INFER_NUM_WORKERS` | **2** | 0 = direct, 8 = Taiji deadlock. 2 가 검증된 안전 영역. |
| `INFER_HEARTBEAT_EVERY_N_BATCHES` | 50 | 진행 가시화 |

**autocast (fp16) 는 default OFF**. 검증된 platform 에서만 추가. embedding lookup + pyarrow 변환과 충돌 가능.

## §18.5 — `make_schema.py` list-type 검출 모든 variant

```python
# WRONG (list_view / 신규 타입 누락)
if pa.types.is_list(col.type):

# CORRECT
if (pa.types.is_list(col.type)
        or pa.types.is_large_list(col.type)
        or pa.types.is_fixed_size_list(col.type)):
```

**Why**: pyarrow 에서 같은 의미의 list 컬럼이 여러 variant 로 표현 가능. 한 가지만 체크하면 schema 의 dim 필드가 잘못 capture (scalar 처리). dataset.py §18.2 fallback 이 잡아주긴 하지만 schema 자체 정확성도 보장해야 future H 의 inflated dim 추정 방지.

## §18.6 — 테스트 의무

신규 H 패키지의 `infer.py / dataset.py` 는 **§18.1–§18.7 룰 모두 준수** 후 cloud upload. 미준수 시 platform AUC 측정 의미 없음 (heuristic 0.5 로 헛 측정).

검증은 `.claude/agents/dataset-inference-auditor.md` 서브에이전트로 자동화. 새 H upload 패킷이 ready 상태가 되기 직전 (tar 직전, training_request.md 제출 직전) 반드시 invoke. PASS 받지 못하면 cloud upload 차단.

## §18.7 — Nullable 컬럼 `to_numpy()` 안전 변환 (CRITICAL — H015/H017 재발 방지)

```python
# WRONG (H015 trainer-time path 가 inference-time 에서 fail)
label_times = batch.column('label_time').to_numpy().astype(np.int64)
labels      = (batch.column('label_type').to_numpy().astype(np.int64) == 2).astype(np.int64)
# pyarrow.lib.ArrowInvalid: Needed to copy 1 chunks with 256 nulls,
# but zero_copy_only was True

# CORRECT
label_times = (batch.column('label_time').fill_null(0)
               .to_numpy(zero_copy_only=False).astype(np.int64))
labels = (batch.column('label_type').fill_null(0)
          .to_numpy(zero_copy_only=False).astype(np.int64) == 2).astype(np.int64)
```

**Why**:
- `to_numpy()` 의 default `zero_copy_only=True`. nullable column 에 null 이 있으면 zero-copy 불가능 → ArrowInvalid 즉시 발생.
- Train data (`demo_1000.parquet`) 의 `label_time` / `label_type` 은 nullable=False (§3). 그래서 training-time loop 에서는 통과.
- Inference data (Taiji eval 용 test parquet) 의 `label_time` / `label_type` 은 정의상 null (예측 대상). Training 에서 silent pass → inference 에서 instant fail.
- **반드시 두 가지를 함께**: `.fill_null(<sentinel>)` (null 제거) + `.to_numpy(zero_copy_only=False)` (copy 허용).

**컬럼별 nullability 빠른 표** (§3 + 운영 경험):

| 컬럼 | Train null? | Infer null? | `to_numpy()` 권장 |
|---|---|---|---|
| `user_id`, `item_id`, `timestamp` | No | No | `zero_copy_only=False` (안전 마진) |
| `label_type`, `label_time` | No | **Yes** | `.fill_null(0).to_numpy(zero_copy_only=False)` 의무 |
| `user_int_*` scalar | No | No | `zero_copy_only=False` |
| 모든 list/dense 컬럼 | element-level null 가능 | 동일 | offset/values 추출 후 처리 |

**언제 위반이 발생하는가** (lessons learned):
- 기존 H 코드는 `label_time` 을 split 단계에서만 (train data) 읽음 → 안전.
- 새 H 가 `label_time` / `label_type` 을 batch-level dataset.py 의 `__call__` 에 추가하면 (예: H015 recency loss weighting, H017 같은 base 카피본) inference path 가 즉시 깨짐.
- Local sample run 은 `data/demo_1000.parquet` (label 있는 train data) 만 사용 → 통과 → cloud 에서 첫 inference 에 fail.

**보호선**:
1. Code review — `dataset-inference-auditor` 서브에이전트가 `to_numpy()` 호출에서 위 표 우측 컬럼 검사.
2. Local sanity — `local_validate.py` 또는 sanity_check 가 **label 컬럼을 null 로 mock 한 row 가 적어도 1개** 포함된 inference batch 를 시뮬레이션.
3. 새 컬럼을 batch-level 에 노출 시 nullable 표 업데이트 의무.

## §18.8 — `train.py` end-of-train SUMMARY block (CRITICAL — 유일한 외부 신호 채널)

Taiji 플랫폼은 `metrics.json` / artifact 파일을 사용자에게 노출 안 함. **stdout 로그만** 사용자가 복사해서 가져올 수 있음. 그래서 `train.py` 마지막 epoch loop 이후 **고정 포맷의 SUMMARY 블록**을 한 번 print 해야 verify-claim 스킬이 파싱 가능.

**고정 포맷 (≤ 20줄, copy-paste 단위)**:

```
==== TRAIN SUMMARY (HXXX_slug) ====
git=<sha7> cfg=<sha8> seed=<int> ckpt_exported=<best|last>
epoch | train_loss | val_auc | oof_auc
  1   |   0.4234   | 0.8124  | 0.8345
  2   |   0.3812   | 0.8312  | 0.8456
  ...
 12   |   0.2891   | 0.8378  | 0.8589
best=epoch<N>  val=<float4>  oof=<float4>
last=epoch<N>  val=<float4>  oof=<float4>
overfit=<+/-float4> (best_val − last_val)
calib pred=<float3> label=<float3> ece=<float3>
==== END SUMMARY ====
```

**구현 스니펫 (`train.py` 끝, after final epoch loop)**:

```python
def emit_train_summary(exp_id, git_sha, cfg_sha, seed, ckpt_kind,
                       epoch_history, best_epoch, best_val, best_oof,
                       last_epoch, last_val, last_oof,
                       pred_mean, label_mean, ece):
    print(f"==== TRAIN SUMMARY ({exp_id}) ====", flush=True)
    print(f"git={git_sha[:7]} cfg={cfg_sha[:8]} seed={seed} "
          f"ckpt_exported={ckpt_kind}", flush=True)
    print("epoch | train_loss | val_auc | oof_auc", flush=True)
    for h in epoch_history:
        print(f" {h['epoch']:>3}  |   {h['train_loss']:.4f}   | "
              f"{h['val_auc']:.4f}  | {h['oof_auc']:.4f}", flush=True)
    print(f"best=epoch{best_epoch}  val={best_val:.4f}  oof={best_oof:.4f}",
          flush=True)
    print(f"last=epoch{last_epoch}  val={last_val:.4f}  oof={last_oof:.4f}",
          flush=True)
    print(f"overfit={best_val-last_val:+.4f} (best_val − last_val)",
          flush=True)
    print(f"calib pred={pred_mean:.3f} label={label_mean:.3f} "
          f"ece={ece:.3f}", flush=True)
    print("==== END SUMMARY ====", flush=True)
```

**Why**:
- Taiji = stdout 만 회수 가능. `metrics.json` 작성해도 사용자 손에 안 들어옴.
- 학습 중간 로그는 길어서 noise. **끝에 단일 블록**이 있어야 사용자가 한 번 긁어서 `verify-claim` 에 paste.
- 이전 7 H 모두 `last_epoch` 만 보고 → P5 (val↔platform gap) 영구 TBD. SUMMARY 블록 도입 시 **best vs last gap, overfit signal, calibration** 모두 한 번에 회수.
- `==== TRAIN SUMMARY` / `==== END SUMMARY` 마커는 정규식 파싱 anchor (변경 금지).
- `oof_auc` 산출 비용이 epoch 마다 큰 모델은 마지막 epoch 만 채우고 중간은 `-` 로 표시 (column 수 유지).

**필드 정의 (모호 금지)**:
- `git=<sha7>` — `git rev-parse --short=7 HEAD` 결과.
- `cfg=<sha8>` — `card.yaml` 의 `config_sha256` 앞 8자.
- `ckpt_exported=best|last` — `infer.py` 가 로드할 checkpoint. **best 권장**. `last` 인 경우 명시 (verify-claim 이 PASS 판정 시 best/last 차이만큼 보정).
- `best=epoch<N>` — `argmax(val_auc)` 의 epoch 번호.
- `overfit=` — best_val − last_val. 양수 = 마지막 epoch 가 best 보다 안 좋음 (overfit 의심). 음수 = 마지막이 best (이상적).
- `calib pred=` — 검증셋 예측 확률 평균. `label=` — 검증셋 양성 비율. `ece=` — Expected Calibration Error (10-bin).

**언제 위반인가**:
- SUMMARY 블록 없음 → 사용자가 복사할 게 없음 → 검증 자체 차단.
- 마커 (`==== TRAIN SUMMARY (`, `==== END SUMMARY ====`) 변형 → verify-claim 정규식 fail.
- epoch table 컬럼 변경 / column 수 변동 → parser 오작동.
