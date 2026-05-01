# Training Request — E000_unified_baseline_demo

> 사용자 클라우드 플랫폼의 **"Training Code" 페이지** 업로드 → `bash run.sh` 실행 모델. self-contained.

## TL;DR
- **Hypothesis**: H001_unified_baseline_full (anchor)
- **Claim**: Organizer PCVRHyFormer + 결함 A/B/C/D 패치만 → anchor 측정 (zero hyperparameter mutation).
- **Compute tier**: 플랫폼 기본 GPU (A10/A100 가정).
- **Expected wall**: ~10–30 min (50 epoch, demo_1000 1000 rows).
- **Storage cap**: 100.0 MB. **이 bundle: 169.6 KB** (cap 대비 0.17%).
- **로컬 검증 완료**: M1 CPU에서 val_AUC=0.5165, OOF_AUC=0.7415 (1 epoch dry-run).

## Upload bundle — `experiments/E000_unified_baseline_demo/upload/` (flat 디렉토리, 10 files)

플랫폼 화면의 "Training Code" 테이블에 그대로 매칭되는 file manifest:

| File | Size | Purpose |
|---|---|---|
| `dataset.py` | 37.5K | PCVRParquetDataset + **patched v2 split** (label_time + 10% user OOF) |
| `model.py` | 63.6K | PCVRHyFormer (organizer-supplied, 1714 lines, 변경 없음) |
| `ns_groups.json` | 2.0K | NS feature grouping reference (현 run에선 미사용 — `--ns_groups_json ""`) |
| `run.sh` | 2.1K | **Entry point**. 플랫폼이 `bash run.sh` 호출. local-dev default 제거됨. |
| `train.py` | 22.5K | Trainer driver + `--use_label_time_split`, `--oof_user_ratio`, `metrics.json` 자동 dump |
| `trainer.py` | 21.8K | Pointwise BCE/Focal loop + tqdm graceful fallback (no-op when missing) |
| `utils.py` | 11.3K | Logger, EarlyStopping, focal loss helper |
| `make_schema.py` | 5.6K | **추가**. parquet → schema.json 자동 생성 (ts_fid heuristic 포함) |
| `README.md` | 2.9K | Technical report stub (Final Round 의무 준수) |
| `requirements.txt` | 0.1K | Pinned deps (informational — 플랫폼 prebuilt python image면 무시) |

**Backup archive**: `experiments/E000_unified_baseline_demo/upload.tar.gz` (43.4K) — 브라우저 업로드 중 파일 하나 누락 시 재추출용.

## Steps (사용자 → 플랫폼)

### 1. 파일 업로드
플랫폼 "Training Code" 페이지에서:
- "Upload from Local" 클릭 → 위 10개 파일 각각 업로드 (또는 다중 선택 지원 시 전체 선택).
- 업로드 후 파일 테이블이 **위 manifest와 정확히 일치**하는지 확인.
- Storage Size 표시: `169.6K / 100.0M` 정도 (1% 미만).

### 2. 환경 변수 (플랫폼 콘솔)
플랫폼이 자동 주입한다고 가정. UI에서 명시 설정 가능하면:

| Env | 의미 | 필수? |
|---|---|---|
| `TRAIN_DATA_PATH` | 학습 parquet 디렉토리 | ✅ |
| `TRAIN_CKPT_PATH` | ckpt + metrics.json 저장 디렉토리 | ✅ |
| `TRAIN_LOG_PATH` | train.log 디렉토리 | optional (CKPT/logs로 derive) |
| `TRAIN_TF_EVENTS_PATH` | tensorboard events | optional (CKPT/tf_events) |

run.sh가 미설정 시 즉시 abort 메시지 출력 (`TRAIN_DATA_PATH not set by platform`).

### 3. 학습 실행 — 3 seed 권장 (multi-seed anchor)

플랫폼이 **첫 호출**: `bash run.sh --num_epochs 50 --batch_size 256 --seed 42`
- TRAIN_CKPT_PATH = `${{base_ckpt}}/ckpt_seed42`

**두 번째**: `bash run.sh --num_epochs 50 --batch_size 256 --seed 1337`
- TRAIN_CKPT_PATH = `${{base_ckpt}}/ckpt_seed1337`

**세 번째**: `bash run.sh --num_epochs 50 --batch_size 256 --seed 2026`
- TRAIN_CKPT_PATH = `${{base_ckpt}}/ckpt_seed2026`

3개 run 모두 동일 코드 (파일 재업로드 불필요), env CKPT만 다르게.

### 4. (옵션) 30초 sanity dry-run
본 학습 트리거 전, 동일 코드로 1 epoch만:
```
bash run.sh --num_epochs 1 --batch_size 32
```
→ 약 30초 안에 metrics.json 생성. NaN 없으면 본 학습 진행.

### 5. 결과 다운로드

플랫폼 콘솔/파일 탐색기에서:
- `${{TRAIN_CKPT_PATH}}/metrics.json` — **1차 보고 source** (수 KB).
- `${{TRAIN_CKPT_PATH}}/global_step*.best_model/{model.pt, schema.json, train_config.json}` — best ckpt + sidecars (수십 MB, 한 seed만 받아도 됨).
- `${{TRAIN_LOG_PATH}}/train.log` — 마지막 50–200줄.
- `${{TRAIN_WORK_PATH}}/_split_meta.json` — train/valid/oof row count + label_time cutoff.

## Falsification thresholds (card.yaml에서)
- **P1**: NaN-free, finite best_val_AUC.
- **P2**: predictions stdev (test 기준) ≥ 0.01 — submission/infer.py로 측정.
- **P3**: |train_AUC − val_AUC| ≥ 0.05 — 현재 trainer는 train AUC 미측정 → inconclusive 가능.
- **P4**: `submission/local_validate.py` 5/5 PASS — 다운로드한 ckpt로 로컬 재검증.
- **P5**: OOF AUC ≥ 0.50.
- **P6 (multi-seed)**: 3 seed의 OOF AUC stddev ≤ 0.05.
- **P7 (sanity)**: cloud seed=42 OOF가 로컬 결과 (0.7055) ± 0.10 안.

## Report back

`experiments/E000_unified_baseline_demo/training_result.md` 양식 채워 paste:
- 3개 seed의 metrics.json 본문
- log 마지막 30줄
- 각 run wall time + cost
- 다운로드한 best_model 경로

또는 최소: 3개 metrics.json만 paste 해도 intake skill이 처리 가능.

이후 어시스턴트에게 "E000 결과 왔어" → `cloud-intake` skill 호출 → INDEX/verdict/progress 자동 갱신.

## 중요 주의

- **본 데이터셋 (1000 rows × 124 positives)** 는 신뢰구간 매우 큼. 이 anchor는 **자격 검증** 용도이지 leaderboard 점수 예측 아님 (`claim_scope=demo-only`).
- Full data 도착 시 별도 `E000_full_unified_baseline/` 실험 폴더로 새 packet 발급.
- run.sh의 `--ns_groups_json ""` 옵션은 의도된 것 (RankMixer NS tokenizer 기본 사용; group tokenizer로 전환 시 ns_groups.json 활용).
