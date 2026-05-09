# H032 — Verdict (REFUTED — noise band, 2026-05-05)

## Result (cloud submission)
- **eval AUC: 0.83328**
- vs control H023 baseline (0.8334): Δ ≈ **−0.00012pt** → **NOISE band** (§17.3 cut < +0.001pt)
- predictions.md decision tree → "noise" 분기 적용:
  > "seq 내부 time bucketing + per-batch optimizer 로 implicit 학습 충분.
  > input-axis temporal signal retire."

## F-1 (carry-forward)
**input-axis temporal signal REFUTED**. H015~H018 의 loss-axis 도 모두 marginal/REFUTED. **temporal axis 전반 retire** 강한 신호 (input + loss 양쪽 다 시도, 효과 없음). H_TIMEFEAT class 후속 sub-H 우선순위 낮음.

## F-G update — 12 H ceiling
H011/H012/H013/H015/H018/H022/H023/H025/H026/H030/**H031/H032** 모두 0.832~0.836 narrow band. **12 H 누적 mechanism mutation 효과 없음** confirm. paradigm shift (H019 TWIN) 만 남음.



> Status placeholder. Updated by `verify-claim` skill upon cloud
> training_result.md paste.

## Status
`pending` — H032 **upload package BUILT 2026-05-04** (`upload.tar.gz` 68KB).

**Build approach** (residual ADD 패턴, H031 과 동일):
- `TimeFeaturesBlock` 추가 (1,873 params = 0.0012% of 161M model)
- ModelInput NamedTuple 에 3 Optional fields 추가 (backward-compat)
- dataset.py `_convert_batch` 가 max_seq_ts 추적 + hour/dow/recency 추출
- model forward 에서 residual ADD: `output += sigmoid(time_gate) × time_state`
- gate init = sigmoid(-2.0) ≈ 0.1192 (CLAUDE.md §10.10)
- num_ns 변경 없음 → d_model % T 제약 안전
- §18.7 timestamp.fill_null(0) 적용 (inference null safety)

**Files modified** (6 files):
- model.py: +TimeFeaturesBlock class (~50 lines), +ModelInput 3 Optional fields, +PCVRHyFormer args (3) + instantiation + forward/predict residual ADD
- dataset.py: +max_seq_ts_global tracker init/update, +hour/dow/recency extraction before return, +timestamp.fill_null(0) defensive
- trainer.py: +3 ModelInput fields in `_make_model_input`
- infer.py: +3 ModelInput fields, +3 cfg.get entries
- train.py: +3 CLI args, +3 cfg dict entries
- run.sh: header rewrite + `--use_timestamp_features` + `--time_emb_dim 16` + `--time_gate_init -2.0`

**T0 sanity (local, python3 + torch 2.2.2 MPS)**:
1. ✅ TimeFeaturesBlock direct forward (4, 64), NaN-free, gate=0.1192
2. ✅ All 10 cross-block params have non-zero grad
3. ✅ Defensive clamp on OOB inputs (hour=24, dow=7, recency=8 → 모두 NaN-free)
4. ✅ Full PCVRHyFormer instantiation (160,934,866 total params, 1,873 time block)
5. ✅ 1-batch forward (B=4) shape (4, 1) finite
6. ✅ Ablation enabled vs disabled max abs diff = 0.0753 (cross 작동)
7. ✅ dataset.py end-to-end: hour/dow/recency 모두 vocab 범위 내 (demo_1000 의 15분 window 특성으로 hour=15 dow=0 uniform — full data 에서 다양한 값 측정 예상)

**Cloud submission**: ready (control=H023). 권장 호출:
```
bash run.sh --seed 42
```

**Stacking with H031**: 둘 다 residual ADD post-backbone, orthogonal axes (item-side fid cross vs temporal context). 둘 다 PASS 면 H033 = stacking 시도 가능.

## Source data
- TBD (post-cloud).

## P0 — Audit gate
- TBD (training timestamp non-null 검증 + max_seq_ts 추출 로직).

## P1 — Code-path success
- TBD.

## P2 — Primary lift (§17.3 binary)
- TBD. Δ vs control. Cut: ≥ +0.005pt strong / ≥ +0.001pt measurable / < +0.001pt noise/refuted.

## P3 — Mechanism check (time_token gradient norm)
- TBD.

## P4 — §18 인프라 통과
- TBD.

## P5 — val ↔ platform gap
- TBD.

## P6 — OOF (redefined) ↔ platform gap
- TBD.

## P7 — verify-claim §18.8 SUMMARY parser dry-run
- TBD.

## Findings (F-N carry-forward)
- TBD.

## Surprises
- TBD.

## Update to CLAUDE.md?
- TBD.

## Carry-forward to next H
- TBD per Decision tree (predictions.md).

## Decision applied (per predictions.md decision tree)
- TBD.
