# H004 — Cloud Training Request

> Generated per CLAUDE.md §17.8.1 (cloud-handoff). Hand this packet to the
> Taiji Training Code page, run, then return with the artifacts listed in §6.

## 1. Hypothesis & Claim
- Hypothesis: **H004_onetrans_anchor**
- Anchor (NOT one-mutation, §17.2 anchor exemption): PCVRHyFormer 의
  `MultiSeqHyFormerBlock` + `MultiSeqQueryGenerator` 를 OneTrans
  single-stream block + mixed-causal mask 로 통째로 교체. 결함 A/B/C/D/E
  패치 인프라 (dataset.py, train.py path defaults, make_schema.py, infer.py
  prior fallback) H001 그대로 재사용.
- Control reference: `E_baseline_organizer` (PCVRHyFormer, val_AUC = **0.8251**).
- Anchor 자격 게이트 (4 P1–P4):
  - P1: 1 epoch NaN-free 완주.
  - P2: val_AUC ≥ **0.70** hard / ≥ **0.80** soft (anchor 자격 + 비교 의미).
  - P3: 모든 layer attn entropy < 0.95·log(N_tokens) (§10.9 룰).
  - P4: G1–G6 5/5 PASS.

## 2. Compute tier
- `T2.4` smoke (Taiji platform, 사용자 manual launch).
- Expected wall: **3–6 min** (E_baseline_organizer 2:47 + similar attn cost).
- Cost cap (§17.6): per-job ≤ $5, per-day ≤ $20.
- Full-data (T3) 는 smoke anchor 자격 통과 후 별도 row.

## 3. Upload manifest (Taiji "Upload from Local", flat namespace)

경로: `experiments/H004_onetrans_anchor/upload/`
백업: `experiments/H004_onetrans_anchor/upload.tar.gz` (61,065 B)
총 용량: **248 KB** — 100 MB cap 여유 충분.

| File | Bytes | H001 대비 | Role |
|---|---:|---|---|
| `run.sh` | 2,400 | 변경 | Entry point. `--backbone onetrans --num_onetrans_layers 2 --mixed_causal_anchor timestamp --log_attn_entropy` baked. |
| `train.py` | 27,765 | +3 KB | NEW H004 CLI flags 5종, `attn_entropy_per_layer` 진단 dump. |
| `trainer.py` | 23,039 | 동일 | Train loop (BCE, AUC eval). |
| `model.py` | 77,926 | +15 KB | NEW: `build_onetrans_mask`, `OneTransAttention`, `OneTransBlock`, `OneTransBackbone` + `PCVRHyFormer` backbone router. |
| `dataset.py` | 49,154 | 동일 | PCVRParquetDataset + organizer split. |
| `infer.py` | 12,179 | +218 B | OneTrans cfg 5종 read-back (backbone, num_onetrans_layers, mixed_causal_anchor, domain_id_embedding). |
| `local_validate.py` | 6,949 | 동일 | G1–G6 gate runner. |
| `make_schema.py` | 10,049 | 동일 | Auto schema from parquet. |
| `utils.py` | 11,545 | 동일 | Logger, EarlyStopping, focal helper. |
| `ns_groups.json` | 2,092 | 동일 | NS-token feature group ref. |
| `requirements.txt` | 100 | 동일 | torch 2.7.1+cu126, etc. |
| `README.md` | 6,466 | 신규 | H004 anchor 정체성 (§17.8 Final-Round 의무). |

총 12 files.

## 4. Platform env vars (필수)
- `TRAIN_DATA_PATH` — training parquet 디렉토리. 미설정 시 `run.sh` abort.
- `TRAIN_CKPT_PATH` — writable, metrics.json + best_model 경로. 미설정 시 abort.
- `TRAIN_LOG_PATH`, `TRAIN_TF_EVENTS_PATH`, `TRAIN_WORK_PATH` — 미설정 시
  CKPT/{logs,tf_events,work} 자동 derive.

> §17.8.7: `run.sh` 는 local-dev fallback 제거됨. 두 필수 env 누락 = abort.

## 5. Run command
플랫폼이 자동으로:
```
bash run.sh
```
호출. `run.sh` 내부 baked args (변경 금지):
```
--num_epochs 1
--seed 42
--ns_tokenizer_type rankmixer
--user_ns_tokens 5         # H001 anchor와 동일 (paired 비교)
--item_ns_tokens 2         # H001 anchor와 동일
--num_queries 2            # hyformer 경로 호환용 (onetrans는 무시)
--ns_groups_json ""
--emb_skip_threshold 1000000
--num_workers 2
--buffer_batches 4
--train_ratio 0.05         # smoke (~94k rows)
--seq_max_lens seq_a:64,seq_b:64,seq_c:128,seq_d:128
--backbone onetrans        # H004 NEW (router → OneTransBackbone)
--num_onetrans_layers 2    # H001 num_hyformer_blocks=2 와 paired
--mixed_causal_anchor timestamp  # paper default; seq_index fallback 옵션 살아있음
--log_attn_entropy         # §10.9 룰 진단 dump
```

## 6. Bring-back artifacts (intake 시 paste 필요)
사용자가 학습 끝나고 가져와야 할 것 — `cloud-intake` skill 이 처리:

1. **`metrics.json`** (TRAIN_CKPT_PATH) — 핵심 필드:
   - `best_val_AUC` (P2 판정 baseline)
   - `best_oof_AUC` (organizer split이라 null 가능)
   - **H004 NEW**: `attn_entropy_per_layer` (list, 길이=num_onetrans_layers=2),
     `attn_entropy_threshold` (= 0.95·log(N_tokens)), `attn_entropy_violation` (bool, P3 판정)
   - `seed`, `git_sha`, `config_sha256`, `host`, `python`, `cuda`, `torch`
   - `split_meta`, `total_param_count`
2. **`train.log` 마지막 ~200 lines** — NaN/OOM, attention dim mismatch 검증.
   특히 `Backbone: onetrans, num_onetrans_layers=2, mixed_causal_anchor=timestamp,
   domain_id_embedding=True, log_attn_entropy=True` 줄 + attn_entropy_per_layer 로그
   확인.
3. **Submission round-trip 증빙** — best_model 으로 `submission/local_validate.py`
   5/5 PASS 로그 (P4).
4. **Wall time** — 실측 분.

## 7. Verdict update path (post-intake)
- `hypotheses/H004.../verdict.md` 의 P1–P5 TBD 채우기.
- `hypotheses/INDEX.md` 의 H004 status: `scaffold` → `pending` (이미) → `done`.
- `experiments/INDEX.md` 새 row: `EXP_ID = E002`, hypothesis_id, val_AUC,
  attn_entropy_per_layer, config_sha256, git_sha, status.
- `progress.txt` 1 블록 append.
- 결과에 따라 anchor 선택:
  - **P2 PASS (≥0.80) + P3 PASS** → OneTrans-anchor 등록. 미래 H 들이
    max(PCVRHyFormer 0.8251, OneTrans X) 위에서 mutation. 즉시 후속 H005
    promote (memory `project_h005_candidate.md` 참조 — NS-token 12로 expand).
  - **P2 soft warning (0.70–0.80) + P3 PASS** → 두 anchor 공존, 약한 anchor
    archive 룰 발동. PCVRHyFormer-anchor 단독 가능성 높음.
  - **P2 PASS + P3 FAIL (entropy violation)** → §10.9 룰 첫 active 검증
    성공. sample-scale 에서 OneTrans 사용 보류, full-data T3 도착 후 재검증.
  - **P2 hard fail (<0.70)** → OneTrans backbone 자체 retire. PCVRHyFormer-anchor
    단독 유지. H005 후보 폐기, PCVRHyFormer 위 mutation 큐 재구성.

## 8. Pre-flight checks (사용자 launch 전)
- [ ] `upload/` 12 파일 모두 Taiji 에 업로드 (subdir 없음, flat).
- [ ] `TRAIN_DATA_PATH`, `TRAIN_CKPT_PATH` 환경변수 설정.
- [ ] (선택) 30-sec sanity: `bash run.sh --num_epochs 1 --batch_size 32`.
- [ ] launch.

## 9. Repro pin
- git_sha (현 워킹트리): TBD — 사용자 launch 직전 `git rev-parse HEAD` 캡쳐.
- config_sha256: card.yaml `repro_meta.config_sha256` 미정 → run 후 metrics.json
  에서 확인 (train.py 가 `hashlib.sha256(json.dumps(vars(args)))[:16]` 으로 계산).
- model.py 신규 추가 코드: 309 lines (1714 → 2023). OneTransBackbone +
  block + attention + mask builder + PCVRHyFormer router 변경.
- 로컬 sanity 한계: M1 Pro 에 torch 미설치 → forward live test 불가.
  Static syntax check (`python3 -c "import ast; ast.parse(...)"`) 12 files
  모두 PASS. live forward 검증은 Taiji 에서 수행.

## 10. Build status: ✅ COMPLETE — ready to upload
