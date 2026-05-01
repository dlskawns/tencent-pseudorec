# Training Request — {EXP_ID}

> 사용자가 클라우드 플랫폼의 **"Training Code" 페이지**에 파일을 업로드하고 학습을 트리거. self-contained — 우리 로컬 컨텍스트 없이 그대로 따라하면 됨.

## TL;DR
- **Hypothesis**: {H???_slug}
- **Claim (1줄)**: {transfer.md ④}
- **Compute tier**: {T?.?}
- **Expected wall**: ~{N} min
- **Cost cap (CLAUDE.md §17.6)**: ${X} per-run / ${Y} per-campaign
- **config_sha256**: {16-hex from card.yaml}

## Upload bundle
모든 파일은 `experiments/{exp_id}/upload/` 에 있음. 한 번에 업로드 가능한 backup archive: `experiments/{exp_id}/upload.tar.gz` (크기 ≤ 200KB, 플랫폼 100MB cap 대비 여유).

| File | Size | Purpose |
|---|---|---|
| `dataset.py` | ~30K | PCVRParquetDataset + patched v2 split + OOF holdout |
| `model.py` | ~63K | PCVRHyFormer (organizer-supplied, unchanged) |
| `ns_groups.json` | 2K | NS feature grouping reference |
| `run.sh` | ~2K | Entry point — platform invokes `bash run.sh` |
| `train.py` | ~18K | Trainer driver + metrics.json output |
| `trainer.py` | ~21K | Pointwise BCE/Focal loop + tqdm fallback |
| `utils.py` | ~11K | Logger, EarlyStopping, focal loss |
| `make_schema.py` | ~5K | Auto-generates schema.json from parquet |
| `README.md` | ~5K | Technical report stub (Final Round 의무) |
| `requirements.txt` | ~0.3K | Pinned deps (informational) |

## Steps (사용자 → 플랫폼)

### 1. 파일 업로드
플랫폼 "Training Code" 페이지에서 **개별 업로드**:
- "Upload from Local" 클릭 → `experiments/{exp_id}/upload/` 안의 파일을 하나씩 (또는 다중 선택).
- 또는 백업 tar.gz 다운로드 후 로컬 추출 → 추출된 파일 재업로드.
- 업로드 후 파일 테이블이 위 manifest와 일치하는지 확인. Storage Size 합계 ≤ 200K.

### 2. 환경 변수 (플랫폼 측)
플랫폼이 자동 주입한다고 가정. 누락 시 콘솔에서 명시 설정:
- `TRAIN_DATA_PATH` — 학습 parquet 디렉토리 (필수, organizer 컨벤션).
- `TRAIN_CKPT_PATH` — checkpoint 저장 디렉토리 (필수).
- `TRAIN_LOG_PATH`, `TRAIN_TF_EVENTS_PATH` — 옵션, run.sh가 CKPT 하위로 자동 derive.

### 3. 학습 실행
플랫폼이 `bash run.sh` 호출 (또는 사용자가 manual trigger). run.sh의 1줄 args 추가는 다음과 같이 가능:
- `bash run.sh --num_epochs 50 --batch_size 256 --seed 42` (multi-seed 1번째)
- `bash run.sh --num_epochs 50 --batch_size 256 --seed 1337` (multi-seed 2번째)
- `bash run.sh --num_epochs 50 --batch_size 256 --seed 2026` (multi-seed 3번째)

각 run마다 `TRAIN_CKPT_PATH`를 다르게 설정 (예: `${{base}}/ckpt_seed42`, `_seed1337`, `_seed2026`) 해서 ckpt 충돌 회피.

### 4. (옵션) 30초 dry-run
본 학습 전 sanity:
```
bash run.sh --num_epochs 1 --batch_size 32
```

### 5. 결과 다운로드
- `${{TRAIN_CKPT_PATH}}/metrics.json` — **1차 보고 source** (수 KB).
- `${{TRAIN_CKPT_PATH}}/global_step*.best_model/` — model.pt + sidecars (~MB–GB).
- `${{TRAIN_LOG_PATH}}/train.log` — 마지막 50–200줄.
- `${{TRAIN_WORK_PATH}}/_split_meta.json` — train/valid/oof row count + label_time cutoff.

## Falsification thresholds (card.yaml에서 복사)
- P1: NaN-free + finite best_val_AUC.
- P2: predictions stdev ≥ 0.01 (mechanism check, infer.py 라운드트립으로 측정).
- P3: |train_AUC − val_AUC| ≥ 0.05 (overfit 검증).
- P4: `submission/local_validate.py` 5/5 PASS (다운로드한 ckpt로 로컬 재검증).
- P5: OOF AUC ≥ 0.50 (random보다 좋음).
- P6 (multi-seed): seed×3 OOF stddev ≤ 0.05.
- P7 (sanity): cloud seed=42 OOF가 로컬 결과 ± 0.10 안.

## Report back

`experiments/{exp_id}/training_result.md` 양식 채워 paste.
또는 최소 필드: `metrics.json` 본문 + log 마지막 30줄 + wall time + cost.

이후 어시스턴트에게 "{exp_id} 결과 왔어" 라고 하면 `cloud-intake` skill로 INDEX/verdict/progress 자동 갱신.
