# H004 — Predictions

> Pre-registered before run. 실행 후 verdict.md 의 measured 값과 비교. 본 H 는 anchor 이므로 §17.3 binary lift 임계 미적용 — 4 게이트 anchor 자격 조건이 falsification 기준.

## P1 — Code-path success (NaN-free 1 epoch 완주)

- Quantity: train.py 1 epoch 완주, `metrics.json` 생성, `model.pt` save.
- Predicted: NaN 0건, finite val_AUC, finite logloss.
- Falsification: NaN abort, OOM, schema mismatch, attention dim mismatch → **REFUTED**. mixed causal mask 또는 token taxonomy 구현 오류 신호 → 디버깅 후 1회 retry, 2회 fail 시 OneTrans backbone 자체 retire.

## P2 — Anchor 자격 (val_AUC threshold)

- Quantity: smoke val_AUC (organizer row-group split, train_ratio=0.05, num_epochs=1, seq_max_lens=a:64,b:64,c:128,d:128 — H002 와 동일 control envelope).
- Predicted: val_AUC ≥ **0.70** (hard floor — 학습 됨 입증). 또한 soft target val_AUC ≥ **0.80** (PCVRHyFormer-anchor 0.8251 와 같은 ballpark — anchor 비교 의미).
- Falsification:
  - Hard fail: val_AUC < 0.70 → 모델이 학습 안 됨, OneTrans backbone 자체 retire.
  - Soft warning: 0.70 ≤ val_AUC < 0.80 → anchor 자격은 인정하지만 PCVRHyFormer 와 비교 시 OneTrans 가 우리 데이터에 fit 안 함 → 미래 H mutation 우선순위는 PCVRHyFormer-anchor 위에서.
  - PASS: val_AUC ≥ 0.80 → 두 anchor 공존, 미래 H mutation 양쪽에서 paired Δ 측정.
- 의미: §17.3 binary lift 미적용 (anchor 면제). PCVRHyFormer 보다 무조건 높을 필요 없음 — paper 의 lift claim 은 별도 H 에서 검증.

## P3 — §10.9 attention entropy 적정성

- Quantity: 매 layer 의 attention probability 의 mean entropy. random batch 4개 평균. layer 수 = num_layers (H001 default 2 와 동일하게 config).
- Predicted: 모든 layer 에서 mean entropy < **0.95 · log(N_tokens)**. N_tokens = L_S + L_NS + 1 (candidate). 우리 smoke setup 에서 L_S ≤ 64+64+128+128 = 384, L_NS = 7, candidate=1 → N_tokens ≤ 392 → log(392) ≈ 5.97 → threshold ≈ 5.67.
- Falsification:
  - Hard fail (any layer): max layer entropy ≥ 0.95·log(N) → uniform attention collapse → OneTrans softmax routing 이 sample-scale 에서 비기능 → §10.9 룰대로 abort, full-data 학습 보류 + verdict.md `attn_entropy_violation: true` carry-forward.
  - PASS: 모든 layer 에서 entropy < threshold → backbone 정상 작동.
- 의미: paper claim 의 scale dependency (challengers.md Frame 1) 직접 측정. fail 이면 sample-scale 에서 paper claim 검증 불가능 신호 — 본 H 는 anchor 자격 미달.

## P4 — Submission round-trip

- Quantity: 다운로드된 ckpt 로 `submission/local_validate.py` 5/5 PASS (G1–G6).
- Predicted: G1–G6 모두 통과 (H001 과 동일 contract — `infer.py` 가 OneTrans ckpt 로드 가능하도록 model class registry 에 OneTrans entry 추가만 신규).
- Falsification:
  - 1개라도 fail → **REFUTED** + §13.7 매핑.
  - 가장 risky: G1 (signature) — `infer.py` 의 `load_model()` 에서 OneTrans class 라우팅 누락 가능성. 코드 작성 시점에 명시 검토.
  - G5 (determinism) — OneTrans attention 의 dropout / FloatPoint 누적 변화 검증 필요. seed 고정 + `model.eval()` + `torch.no_grad()` 로 충족 예상.
- 의미: §14.1 G1–G6 + H001 anchor 인프라 재사용 검증. 본 H 가 OneTrans 만 갈아끼우고 inference path 그대로 두면 자동 통과 예상.

## P5 — Bonus: Anchor pair stability

- Quantity: PCVRHyFormer-anchor val_AUC=0.8251 vs OneTrans-anchor val_AUC=X.
- Predicted: |0.8251 − X| ≤ 0.05 (두 backbone 이 같은 데이터에서 같은 ballpark — 두 anchor 의미 있는 비교 가능).
- Falsification 아님: 본 H 의 success 자체는 P5 와 무관. 차이가 크면 미래 H 의 anchor 선택 룰이 더 엄격해짐 (max(0.8251, X) 위에서만 mutation, 떨어지는 anchor archive). noted in verdict.md, 본 H REFUTED 처리 안 함.

## Reproducibility

- compute_tier: T2.4 smoke (Taiji organizer mode + OneTrans backbone args).
- seed: 42 (single — paired Δ 의도 없음, anchor 자격만 검증). multi-seed 는 H004 통과 후 mutation H 들에서.
- split: organizer row-group, train_ratio=0.05, valid 100 RGs (H002/E_baseline_organizer 와 동일).
- seq_max_lens: seq_a:64,seq_b:64,seq_c:128,seq_d:128 (H001/H002 anchor 와 동일).
- expected wall: ~3–6 분 (H002 2:47 + OneTrans single-stream 의 attention size 가 PCVRHyFormer 와 비슷, layer 수 동일이면 wall 도 비슷).
- code: `experiments/H004_onetrans_anchor/upload/` (12 파일, run.sh 가 `--backbone onetrans` baked).
- expected param count: ~198M (PCVRHyFormer 와 같은 ballpark — embedding tables dominant; single-stream block 자체 <1M 차이).

## Negative-result interpretation (§17.7 falsification-first)

본 H 가 REFUTED 인 경우 학습 가능한 정보:

- P1 fail (NaN/OOM/schema): 코드 오류 → 디버깅 + retry. 정보 가치 = 작음, 단순 implementation bug.
- P2 hard fail (val_AUC < 0.70): OneTrans backbone 이 우리 데이터에 fit 안 함. **정보 가치 = 큼** — backbone 다양성 확보 시도가 통합 깊이 한 단계 안으로 가는 것이 정답이 아니라는 신호. 미래 H 는 PCVRHyFormer-anchor 위 mutation 으로 좁혀짐 + 다른 backbone 후보 (HSTU?) 탐색.
- P2 soft warning (0.70–0.80): 두 anchor 우열 명확. 약한 anchor 사용성 낮음.
- P3 entropy fail: paper claim 의 scale dependency 직접 검증. **정보 가치 = 큼** — sample-scale 에서 OneTrans 검증 불가 + full-data 도착 전엔 anchor 미확정. Carry-forward: § 10.9 룰의 첫 active 적용으로 룰 자체 검증.
- P4 fail: submission infrastructure 회귀. **정보 가치 = 중간** — H001 anchor 의 인프라 가정 (model class registry) 의 견고성 검증.

→ 모든 negative-result 가 interpretable. malformed experiment 아님 (§17.7 충족).
