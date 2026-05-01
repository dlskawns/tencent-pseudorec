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

신규 H 패키지의 `infer.py / dataset.py` 는 **§18.1–§18.5 룰 모두 준수** 후 cloud upload. 미준수 시 platform AUC 측정 의미 없음 (heuristic 0.5 로 헛 측정).
