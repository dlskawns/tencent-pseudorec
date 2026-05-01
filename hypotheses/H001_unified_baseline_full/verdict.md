# H001 — Verdict (SUPPORTED, 2026-04-26)

## Status
`supported` — anchor 자격 4/5 P-condition 충족 (P3 inconclusive).

## Run summary
- exp_id: E000
- compute_tier: T1.2 (M1 Pro CPU, sibling .venv-arm64 borrowed read-only)
- seed: 42, num_epochs: 2 (smoke), batch_size: 64
- split_meta: train=810, valid=90, oof=100, label_time_cutoff=1772725748, oof_users=100/1000
- total params: 160,944,833 (158M sparse Adagrad + 2.5M dense AdamW) — §10.6 sample budget 100,000x 초과 (anchor 자격으로 면제됨, claim_scope=demo-only)

## P1 — Code-path success
- Measured: 2 epoch 완주, NaN 0건, ckpt 저장 (`global_step26.layer=2.head=4.hidden=64.best_model/{model.pt, schema.json, train_config.json}`).
- Predicted: NaN-free + finite best_val_AUC.
- Verdict per P1: ✅ **SUPPORTED**.

## P2 — Mechanism check (model이 prior와 다른 분포)
- Measured: predictions std = 0.1006, range = [0.0876, 0.6845], mean = 0.1252.
- Predicted: std ≥ 0.01.
- Verdict per P2: ✅ **SUPPORTED** (10x threshold). PCVRHyFormer forward path가 실제 작동, mean이 prior 0.124와 거의 일치 — calibration이 우연히 잘 맞은 것.

## P3 — Negative control (overfit gap)
- Measured: train AUC 미측정 (trainer는 epoch별 train AUC 안 산출), val AUC = 0.5088, OOF AUC = 0.7055.
- Predicted: |train_AUC − valid_AUC| ≥ 0.05.
- Verdict per P3: ⚠️ **inconclusive** — train AUC 미측정. 그러나 val AUC가 0.51 (random에 가까움) 인 반면 OOF는 0.71 인 **역전 현상**이 관찰됨. 1000-row noise 또는 split 구조 차이 (val=label_time tail, OOF=random user) 둘 중 하나. carry-forward로 분리 측정 필요.

## P4 — Submission round-trip (§13 contract)
- Measured: G1 PASS / G2 PASS / G3+G4 PASS (n_users=1000) / G5 PASS (bit-identical 2회 run) / G6 PASS.
- Predicted: 5/5.
- Verdict per P4: ✅ **SUPPORTED**.

## P5 — OOF generalization
- Measured: OOF AUC = 0.7055 (10% user holdout, seed=42, 100 users, label_time-uniform sample).
- Predicted: ≥ 0.50.
- Verdict per P5: ✅ **SUPPORTED** (random보다 훨씬 좋음). 단, 1000-row 신뢰구간 매우 큼 — full-data 결과로 재검증 의무.

## Findings (F-N carry-forward)

### F-1 — val AUC vs OOF AUC 역전 (P3-related surprise)
val AUC (0.51) < OOF AUC (0.71) 인 역전이 관찰됨. 가능한 기제:
- (a) label_time cutoff val은 13분 윈도우 끝부분 90 rows만 봄 → 그 시점에 모델이 덜 학습됨.
- (b) OOF는 시간 분포가 train과 균일 → 학습 분포에 더 가까움.
- (c) 1000-row × 124 positive에서 paired AUC noise 자체가 ±0.10+.

**Carry-forward**: val을 early-stopping 신호로 쓰는 현 setup은 (a)/(b)가 사실이면 **systematically 나쁜 ckpt를 고르고 있다**. H002 이상에서 *둘 중 어느 것이 dominant인가*를 측정하는 ablation 의무. 옵션: (i) val을 random row sample (label_time 비독립) 로 바꾸기, (ii) early-stopping 신호를 OOF AUC로 교체 (overfit 위험 있지만 1000-row 라 무방).

### F-2 — RankMixerBlock T=16 viable on demo (sanity)
T = num_queries(2) × num_seqs(4) + num_ns(8) = 16, d_model=64, 64 % 16 = 0 ✅. 다른 hyperparameter 조합 시 d_model % T == 0 제약 사전 검증 필수 (organizer baseline은 d_model=64 가정).

### F-3 — emb_skip_threshold=1M에서 seq_c가 4/11 features skip
seq_c 의 11 features 중 4개가 vocab > 1M로 skip됨 (model.py:1357 zero vector). seq_a 1/8, seq_b 1/13, seq_c 4/11, seq_d 0/X. seq_c는 high-cardinality dominant 도메인 — H005 (LongerEncoder seq_d) 후 별도 H 후보로 seq_c 의 emb_skip threshold tuning.

### F-4 — Tensorboard / tqdm import는 graceful fallback 필수
sibling .venv는 tqdm/tensorboard 미설치. trainer.py:18 + train.py:222 두 곳 모두 try/except + no-op 클래스 패치 완료. 새 venv 만들거나 cloud 컨테이너 deps 명시 시 unwind 가능 (functional impact 0).

## Surprises
- **OOF > val AUC** (F-1) — 가장 큰 surprise.
- 학습이 epoch 1에서 epoch 2로 갈 때 val AUC가 0.4847 → 0.5088 로 0.024 pt 증가만. 1000-row × 158M sparse params 상황에서는 거의 모든 lift가 random init noise 분리에서 옴.
- `Re-initialized 95 high-cardinality Embeddings (vocab>0)` — `reinit_cardinality_threshold=0` 디폴트가 사실상 모든 embedding을 매 epoch reset함. KuaiShou MultiEpoch trick 의 의도와 다름. H002+ 에서 threshold 적정값 측정 필요 (별도 ablation 후보).

## Update to CLAUDE.md?
**Yes — §10 신규 anti-bias rule 후보 (verdict 작성 시점)**:
- §10.X (proposed) — **early stopping 신호 반박**: label_time 끝부분 valid가 OOF보다 신뢰 낮을 수 있다 (F-1). H002에서 측정 후 confirm 시 추가.
- §10.Y (proposed) — **reinit_cardinality_threshold=0 디폴트 회피**: organizer 디폴트는 매 epoch 95개 embedding 전부 reset (의도와 다름). 새 실험은 명시적으로 설정 (≥ 100k 권장).

## Anchor 자격 결론
H001은 **anchor 자격 충족**. E000을 모든 후속 H의 control로 사용. 측정 임계는 §17.3 (Δ ≥ +0.5 pt OOF AUC, seed×3, paired bootstrap CI > 0).

claim_scope: **"demo-only, full-data 도착 시 E000.full로 재실행 의무"**.
