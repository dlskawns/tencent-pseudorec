# H005 — Predictions

> Pre-registered before run. 실행 후 verdict.md 의 measured 값과 비교.

## P1 — Code-path success

- Quantity: train.py 1 epoch 완주, `metrics.json` 생성, `model.pt` save.
- Predicted: NaN 0건, finite val_AUC, finite logloss.
- Falsification: NaN abort, OOM, focal loss 함수 호출 오류 → **REFUTED**. focal
  branch 가 organizer 코드라 가능성 매우 낮지만 0 아님.

## P2 — Primary lift (§17.3 binary success)

- Quantity: smoke val_AUC vs E_baseline_organizer (control: 0.8251).
- Predicted: **Δ ≥ +0.5 pt** ⇒ val_AUC ≥ **0.8301**.
- Confidence: single seed=42, 같은 split (organizer row-group, 100 valid RGs),
  같은 train_ratio=0.05, 같은 envelope. **Loss 만 변수**.
- Falsification: Δ < +0.5 pt → **REFUTED**. focal_loss_calibration 방향 retire.

## P3 — Logloss 검증 (보조 게이트)

- Quantity: smoke best_val_logloss vs E_baseline_organizer (control: 0.2538).
- Predicted: best_val_logloss ≤ **0.2792** (악화 ≤ 10%). focal 은 logloss 수치
  자체엔 불리할 수 있지만 (modulating factor 가 mean log-likelihood 하락) +50%
  이상 악화는 기능 이상.
- Falsification: 악화 > 10% → 정상 동작 의심 신호. P2 와 무관하게 별도 carry-forward
  (focal hyperparameter (α, γ) tuning 후보로 H 분리).

## P4 — Submission round-trip

- Quantity: 다운로드된 ckpt 로 `submission/local_validate.py` 5/5 PASS (G1–G6).
- Predicted: G1–G6 모두 통과. PCVRHyFormer 그대로 + loss 만 변경 → infer.py 변경
  0, contract 영향 없음.
- Falsification: 1개라도 fail → **REFUTED** + §13.7 매핑 (gate-별 fail 패턴 분석).

## P5 — Bonus: AUC vs LogLoss trade-off

- Quantity: (val_AUC 변화량, logloss 변화량) pair.
- Predicted: AUC 상승 + logloss 약간 하락 (focal 의 expected behavior). 좋은
  signature: ΔAUC > +0.5pt + Δlogloss ∈ (0, +0.02).
- Falsification 아님: pair 가 paper 패턴과 다르면 carry-forward 정보. 예:
  ΔAUC > +0.5pt + Δlogloss < 0 → **AUC 만 상승하고 logloss 도 좋아짐** = 매우 강한
  신호 (보통 trade-off 있는데 둘 다 좋아짐 = 본 mutation 이 representation 자체
  개선).

## Reproducibility

- compute_tier: T2.4 smoke (Taiji organizer mode + focal args).
- seed: 42 (single — paired Δ 의도, multi-seed 는 H005 PASS 후 별도 ablation).
- split: organizer row-group, train_ratio=0.05, valid 100 RGs (E_baseline_organizer
  와 byte-identical envelope).
- seq_max_lens: seq_a:64,seq_b:64,seq_c:128,seq_d:128 (anchor 와 동일).
- num_epochs: 1.
- expected wall: ~3 분 (E_baseline_organizer 동일, focal overhead ≈ 0).
- code: `experiments/H005_focal_loss_calibration/upload/` (12 파일, run.sh 가
  `--loss_type focal --focal_alpha 0.25 --focal_gamma 2.0` baked).

## Negative-result interpretation (§17.7 falsification-first)

본 H 가 REFUTED 인 경우 학습 가능한 정보:

- **P2 fail with Δ ∈ (-0.001, +0.5)** (소폭 lift 또는 noise): focal 효과 marginal.
  CTR 도메인 12% imbalance 에서 focal 의 paper claim 이 우리 데이터에 약하게 발현.
  **정보 가치 = 큼** — loss_calibration 카테고리 retire 신호. H006 = axis-strengthening
  mutation (longer_encoder 또는 target_attention) 으로 큐 재구성. logloss 진단 (P3)
  으로 focal 작동 여부 분리: logloss 가 의미 있게 변했으면 focal 자체는 작동했지만
  AUC 보정 효과 부재 → α/γ 튜닝 별도 H 필요할지 결정.
- **P2 fail with Δ < -0.3** (악화): focal 이 본격적으로 학습을 망침. α=0.25 가
  우리 데이터에 너무 강한 minority weighting 일 가능성. carry-forward: α=0.5 (균형)
  또는 default 0.1 (organizer-tuned) 로 별도 H 시도.
- **P2 fail + P3 fail (logloss 큰 악화)**: focal 함수 의도와 다른 동작. 코드 검증
  + α/γ 정상값 확인.
- **P2 fail + P4 fail**: contract 회귀 — H001 인프라 의존 깨짐. 디버깅.

→ 모든 negative-result 가 interpretable. malformed experiment 아님 (§17.7 충족).

## Decision tree (post-result)

| Result | Next action |
|---|---|
| Δ ≥ +0.5pt + P3/P4 PASS | H005 PASS. PCVRHyFormer-anchor 갱신 (val=val_AUC). H006 = long_seq_retrieval (D 도메인 1100 tail) — axis-strengthening 카테고리 rotation. |
| Δ ∈ [+0.0, +0.5pt) + P3/P4 PASS | weak signal. focal 자체는 작동, lift 마진 부족. H005 REFUTED. H006 = target_attention 또는 longer_encoder. loss_calibration 카테고리 일시 archive. |
| Δ < 0 + P3 PASS | focal 으로 망가짐. α tuning H 분리 후보 (α=0.5 시도) 또는 loss_calibration retire. |
| P3 fail | 진단 + retry. P4 영향 따라 contract 검증. |
| P4 fail | infrastructure 회귀. H005 보류, contract 패치 우선. |
