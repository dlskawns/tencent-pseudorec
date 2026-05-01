# H006_longer_encoder_d_domain — Technical Report

> CLAUDE.md §17.2 one-mutation: original_baseline anchor 와 byte-identical envelope,
> `--seq_encoder_type` transformer → longer (top-K=50 self-attention compression).
> §17.4 rotation 추가 충족 (long_seq_retrieval, 첫 적용).
> §18 inference 인프라 룰 inherit from original_baseline.

## 1. Hypothesis & Claim
- Hypothesis: **H006_longer_encoder_d_domain**
- Claim: PCVRHyFormer baseline 의 TransformerEncoder 는 `O(L²)` self-attention.
  smoke envelope 의 `seq_max_lens=128` 에서도 D 도메인의 1100-event tail 의 약 88%
  데이터가 truncation 으로 손실. **LongerEncoder (organizer-supplied, top-K=50
  self-attention compression)** 로 교체하면 동일 compute 에서 더 긴 입력을
  effectively 처리. paper-grade reference: SIM/ETA/TWIN/HSTU.
- Compute tier: **T2.4 smoke (Taiji)**.
- Expected wall: ~5 분 (anchor 3분 + LongerEncoder top-K overhead 2분 추정).

## 2. What this code does
H001 anchor 인프라 (label_time-aware split, OOF holdout, path defaults, auto
schema.json, infer.py prior fallback) 그대로. PCVRHyFormer backbone 그대로.
**Encoder 만 transformer → longer** (CLI flag).

§18 inference 인프라 룰 모두 inherit from original_baseline:
- §18.1: `infer.py` 의 dataset batch_size 생성자 인자 (1024 default, override 금지)
- §18.2: dataset.py 의 dim==1 universal handler (`to_pylist` + `isinstance`)
- §18.3: 진단 로그 (MODEL_OUTPUT_PATH, ckpt_dir, WARNING/FALLBACK/OK)
- §18.4: INFER_BATCH_SIZE=1024, INFER_NUM_WORKERS=2, no autocast
- §18.5: make_schema.py 모든 list-type variant 검출

## 3. Files
| File | Purpose |
|---|---|
| `run.sh` | Entry point. Bakes `--seq_encoder_type longer` (encoder 만 변경, 나머지 anchor 동일). |
| `train.py` | CLI driver. label_time split + H004 backbone router (default hyformer). |
| `trainer.py` | tqdm 진행률 (epoch / step% / avg loss) 포함. |
| `model.py` | PCVRHyFormer + OneTrans router (dormant). LongerEncoder (line 616) factory 선택. |
| `dataset.py` | PCVRParquetDataset + universal dim==1 handler. |
| `infer.py` | §18 인프라 룰 적용된 inference. |
| `local_validate.py` | G1–G6 gate runner. |
| `make_schema.py` | Auto schema (모든 list variant 검출). |
| `utils.py` | Logger, EarlyStopping. |
| `ns_groups.json` | NS-token feature ref. |
| `requirements.txt` | torch 2.7.1+cu126. |

## 4. Platform contract
- `TRAIN_DATA_PATH` / `TRAIN_CKPT_PATH` 필수.
- `EVAL_DATA_PATH` / `EVAL_RESULT_PATH` (inference 시).

## 5. Outputs
- `metrics.json` — `best_val_AUC`, `best_oof_AUC`, `split_meta`, repro meta.
- best_model 디렉토리 — model.pt + schema.json + train_config.json sidecar.
- `train.log` — tqdm postfix + heartbeat 로그.

## 6. Sanity dry-run
```
bash run.sh --num_epochs 1 --batch_size 32
```

## 7. Method extensions over original_baseline
- **encoder swap**: `--seq_encoder_type transformer` → `longer`. 4 도메인 모두 적용 (CLI global). D 도메인 1100-event tail 이 최대 lift 영역.
- **paper grounding**: SIM (Pi et al. CIKM 2020), ETA (arXiv:2108.04468), TWIN (KDD 2023), HSTU (Meta 2024).
- **NOT a clone**: organizer LongerEncoder 는 self-attention probability mass 기반 top-K. paper 의 candidate-aware retrieval (target item embedding query) 은 별도 H (target_attention 카테고리).

## 8. Reproducibility
- All seeds fixed (42).
- `metrics.json` records git SHA + config SHA256.
- `train_config.json` sidecar at best_model dir; `infer.py` reads it to instantiate matching encoder type.
