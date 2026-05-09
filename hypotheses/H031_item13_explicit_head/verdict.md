# H031 — Verdict (PENDING — awaiting upload build + cloud submission)

> Status placeholder. Updated by `verify-claim` skill upon cloud
> training_result.md paste.

## Status
`pending` — H031 **upload package BUILT 2026-05-04, HOTFIX 2026-05-04** (`upload.tar.gz` 68KB).

### Hotfix log (2026-05-04)
- **Initial cloud launch (5ep × batch 1024) → SIGABRT** (`Aborted` from train.py).
- **Cause**: `Item13UserCrossBlock` 의 `item13_vs=10` 와 `user_specs vs=4~6` 가 demo_1000 schema 에서 hardcode. cloud 가 full-data 로 `make_schema.py` 재실행 → vs 가 훨씬 큼 (vocab-safety-mult 10x 적용 시 vs ≥ 100). 제 `nn.Embedding(11, 32)` 에 11+ index → CUDA assert → SIGABRT.
- **Fix**: train.py 에서 schema 로드 후 `pcvr_dataset.item_int_schema.entries` + `pcvr_dataset.item_int_vocab_sizes` 에서 동적 추출. PCVRHyFormer.__init__ 가 새 args (`item13_offset`, `item13_vs`, `item13_user_specs`) 받아 Item13UserCrossBlock 에 전달. infer.py 도 cfg 통해 동일 전달.
- **Hotfix T0 sanity**: vs=200 (cloud-realistic) 으로 재테스트, 모두 PASS. defensive ValueError 도 추가 (use=True without runtime args → raise).
- **Action**: 사용자 cloud 재upload 시 같은 명령으로 launch 가능 (run.sh 변경 없음).

**Build approach** (transfer.md 의 invasive 안 대신):
- 새 `Item13UserCrossBlock` (model.py 추가, 85,137 params = 0.053% of model)
- residual ADD post-backbone (output += sigmoid(gate) × cross_state)
- 기존 RankMixerNSTokenizer 변경 없음 → H010 anchor 입력 byte-identical 유지
- gate init = sigmoid(-2.0) ≈ 0.1192 (CLAUDE.md §10.10 InterFormer mandate)
- num_ns count 변경 없음 → d_model % T 제약 안전

**T0 sanity (local, python3 + torch 2.2.2 MPS)**:
1. ✅ Item13UserCrossBlock direct forward — shape (4, 64), no NaN
2. ✅ All 21 cross-block params have non-zero grad (with proper loss)
3. ✅ gate value = 0.1192 (sigmoid(-2) 정확)
4. ✅ Per-row variation (다른 i13 → 다른 cross_state, uniform input → 동일)
5. ✅ Full PCVRHyFormer instantiation (161M params total, +85K from cross block)
6. ✅ 1-batch forward (B=4) shape (4, 1) finite
7. ✅ Ablation: enabled vs disabled max abs diff 0.139 (cross 가 contribution)

**Cloud submission**: ready (control=H023). H028~H030 결과 wait 불필요 (사용자 결정).
**dataset-inference-auditor**: 본 H 는 dataset.py / make_schema.py 변경 없음 → §18.1~§18.7 회귀 risk 작음 (H023 inherit). cloud submission 후에 SUMMARY block check.

## Source data
- TBD (post-cloud).

## P0 — Audit gate
- TBD (schema.json fid=13 vs == 10 확인 필요).

## P1 — Code-path success
- TBD.

## P2 — Primary lift (§17.3 binary)
- TBD. Δ vs control. Cut: ≥ +0.005pt strong / ≥ +0.001pt measurable / < +0.001pt noise/refuted.

## P3 — Mechanism check (cross-token activation magnitude)
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
