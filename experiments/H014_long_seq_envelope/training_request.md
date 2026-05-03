# H014 — Cloud Training Request

> Generated per CLAUDE.md §17.8.1 (cloud-handoff). 2026-05-02.

## 1. Hypothesis & Claim
- Hypothesis: **H014_long_seq_envelope**.
- **Single envelope mutation** — H010 mechanism + envelope byte-identical
  except `--seq_max_lens "seq_a:64,seq_b:64,seq_c:128,seq_d:128"` →
  `"seq_a:256,seq_b:256,seq_c:512,seq_d:512"` (4× per domain).
- Goal: 4-layer ceiling diagnosis 의 마지막 unexplored axis **L4 (truncate
  정보 손실)** 검증.
- Data motivation: §3.5 도메인 p90 1393~2215 vs 현재 truncate 64-128 →
  **95%+ 정보 손실**.
- Predicted classifications:
  - **strong** Δ ≥ +0.005pt → L4 confirmed (Platform ≥ 0.8458).
  - **measurable** Δ ∈ [+0.001, +0.005pt] → L4 partial.
  - **noise** Δ ≤ +0.001pt → dense expansion 효과 없음, retrieval mandatory.
  - **degraded** Δ < −0.001pt → 긴 seq noise.
- §17.2 정당화: envelope mutation = "structural" 인정 (input data shape
  변경, mechanism 0).

## 2. Compute tier
- `T2.4` extended (10 epoch × 30%, patience=3).
- Cost cap: per-job ≤ $5.
- **Expected wall: 6-15h** (attention O(L²), seq 4× → compute 16×).
- 누적 cost: H006~H013 ~27h + H014 ~10h = **~37h**. §17.6 cap 임박.

## 3. Upload manifest

경로: `experiments/H014_long_seq_envelope/upload/`
백업: `experiments/H014_long_seq_envelope/upload.tar.gz` (64,243 bytes).
총 12 files.

| File | H010/upload/ 대비 | Role |
|---|---|---|
| `run.sh` | 변경 (seq_max_lens 1줄) | Entry point |
| `README.md` | 변경 | H014 정체성 |
| `model.py` / `train.py` / `infer.py` / `trainer.py` / `dataset.py` / `local_validate.py` / `make_schema.py` / `utils.py` / `ns_groups.json` / `requirements.txt` | byte-identical | unchanged |

## 4. Run command
```
bash run.sh
```
internal baked args (only seq_max_lens changed vs H010):
```
--seq_max_lens "seq_a:256,seq_b:256,seq_c:512,seq_d:512"  # H014 NEW (4× expansion)
+ all other H010 flags 그대로 (lr 1e-4, batch 256 default, num_workers 2, buffer_batches 4, ...)
```

**중요**: 사용자 평소 `--batch_size 2048` override 사용 시 OOM risk. memory
부족 시 batch 줄이거나 sub-H seq 256 uniform.

## 5. Bring-back artifacts (intake 시 paste 필요)

1. **`metrics.json`** — `best_val_AUC`, `best_oof_AUC`, `seed`, `git_sha`,
   `config_sha256`, **seq_max_lens 명시 확인**, `attn_entropy_per_layer`
   (새 threshold ≈ 6.97).
2. **`train.log` 마지막 ~200 lines** — OOM check, NaN check, peak epoch,
   wall.
3. **Submission round-trip** — `[infer] OK` + batch heartbeat.
4. **Platform AUC** (eval auc) — 본 H 의 핵심 measurement.
5. **Wall time** — P7 efficiency (예상 6-15h).

## 6. Verdict update path (post-intake)
- `hypotheses/H014_long_seq_envelope/verdict.md` 의 P1–P7 채우기 + decision
  tree 분기.
- `hypotheses/INDEX.md` H014 status: `code_build_pending` → `done`.
- `experiments/INDEX.md` 새 row.
- 결과에 따라 (`card.yaml.decision_tree_post_result`):
  - **strong** → L4 confirmed, anchor = H014. H015 = TWIN/SIM/HSTU retrieval.
  - **measurable** → L4 partial. H015 = retrieval combo.
  - **noise** → retrieval mandatory. H015 = TWIN/SIM 또는 cohort H.
  - **degraded** → sub-H seq 256 uniform.
  - **OOM** → sub-H 1/2/3.
  - **P6 gap > 2.5pt** → Frame B confirm, cohort H 우선.

## 7. Pre-flight checks
- [x] H013 verdict.md REFUTED (Frame A 갱신, calibration 효과 없음).
- [x] H014 hypothesis docs (6 files) 완비.
- [x] H014 코드 패키지 빌드 (model.py byte-identical with H010 + run.sh seq_max_lens 변경).
- [x] ast.parse 5 .py files OK.
- [x] tar.gz 생성 (64,243 bytes).
- [ ] `upload/` 12 파일 모두 Taiji 에 업로드.
- [ ] `TRAIN_DATA_PATH`, `TRAIN_CKPT_PATH` 환경변수 설정.
- [ ] (선택) sanity dry-run: `bash run.sh --num_epochs 1 --train_ratio 0.05 --batch_size 256`
  (memory smoke test).
- [ ] launch.

## 8. Build status: ✅ BUILT (2026-05-02)
- Code overlay: H010/upload/ 10 파일 byte-identical 카피, 2 파일 변경
  (run.sh / README.md).
- ast.parse: 5 .py files OK.
- Mechanism unchanged: model.py byte-identical (sha256 검증 가능).
- run.sh diff: seq_max_lens 만 변경 (확인됨).
- tar.gz: 64,243 bytes.

## 9. Memory risk warning

batch 2048 + seq 512 + 2 hyformer blocks + 4 heads = significant activation
memory. Taiji GPU spec 미공개 (likely V100 16GB 또는 A100 40GB).

**Mitigation order if OOM**:
1. seq 256 uniform: `--seq_max_lens "seq_a:256,seq_b:256,seq_c:256,seq_d:256"`.
2. seq 2× conservative: `--seq_max_lens "seq_a:128,seq_b:128,seq_c:256,seq_d:256"`.
3. batch 256 복귀 (사용자 override 안 함).

## 10. Why long-seq envelope now (rotation 정당화)

- 직전 3 H (H011/H012/H013) 모두 H010 anchor 위 mechanism / parametric
  mutation REFUTED. **§10.3 challenger rule trigger**.
- H014 = **envelope mutation (data input shape)** — 다른 axis = challenger
  사고 적용 형태.
- §0 north star alignment: sequential axis 직접 강화 (UNI-REC 두 축 중 하나).
- 4-layer ceiling diagnosis 의 L4 = 마지막 unexplored axis. §3.5 정량
  motivation 가장 강함 (95%+ 정보 손실).
