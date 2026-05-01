# H004 — Verdict (PASS as anchor / SOFT WARNING on absolute lift)

> 클라우드 학습 1회 완료 (2026-04-27 22:59 KST). 4 게이트 중 3 PASS (P1, P3,
> 임시 P5), P2 hard PASS / soft WARNING, P4 TBD. OneTrans-anchor 자격은 인정,
> 단 PCVRHyFormer-anchor 대비 약함.

## Status
`done` — anchor 자격 인정 (P1/P2-hard/P3 PASS) + soft warning (PCVRHyFormer 0.8251
대비 −0.77pt). **두 anchor 공존**, 약한 anchor (OneTrans) archive-pending.

## Source data
- Wall: 5분 16초 (예상 3–6분 정중앙).
- 종료 step: epoch 1, step 857/857, best_step=414.
- Ckpt: `.../global_step414.layer=2.head=4.hidden=64.best_model/model.pt`.
- 사용자 paste 로그 (transcript 2026-04-27 22:59).

## P1 — Code-path success (NaN-free 1 epoch 완주)
- Measured: 1 epoch NaN 0건 완주, finite val_AUC=0.8174, finite logloss=0.2564, 정상 ckpt save, sparse re-init 96 embeddings 정상.
- Predicted: NaN-free, finite best_val_AUC.
- **Verdict per P1: PASS**.

## P2 — Anchor 자격 (val_AUC ≥ 0.70 hard / ≥ 0.80 soft)
- Measured: val_AUC = **0.8174042709762075**.
- Control reference (E_baseline_organizer, PCVRHyFormer): val_AUC = **0.8251**.
- Δ vs PCVRHyFormer: **−0.0077 pt**.
- Predicted hard floor: ≥ 0.70.
- Predicted soft target: ≥ 0.80.
- **Verdict per P2: HARD PASS, SOFT WARNING**. anchor 자격 인정 (학습 됨 입증), 단 PCVRHyFormer 보다 약한 anchor.

## P3 — §10.9 attention entropy 적정성
- Measured: attn_entropy_per_layer = **[3.486, 3.910]**.
- Threshold (0.95 · log(N_tokens=392)): **5.6751**.
- Layer 1 entropy / threshold = 3.486 / 5.6751 ≈ 0.614 (uniform 5.97 대비 58%).
- Layer 2 entropy / threshold = 3.910 / 5.6751 ≈ 0.689 (uniform 대비 65%).
- attn_entropy_violation = **False** (모든 layer < threshold).
- Predicted: 모든 layer < threshold.
- **Verdict per P3: PASS**. **§10.9 룰 첫 active 적용 결과 = 룰 자체 검증 성공**. challengers Frame 1 위험 (paper claim의 sample-scale collapse 우려) 해소.

## P4 — Submission round-trip
- Measured: TBD (best_model 다운로드 + `submission/local_validate.py` 5/5 미실행).
- Predicted: 5/5 PASS.
- **Verdict per P4: PENDING** — H005 작업과 병행 가능.

## P5 — Anchor pair stability (보너스)
- Measured: |0.8251 − 0.8174| = **0.0077** < threshold 0.05.
- **Verdict per P5: PASS**. 두 anchor가 같은 ballpark 안에 있어 paired 비교 가능.

## Findings (F-N carry-forward)

- **F-1 (P3 PASS — 큰 정보)**: §10.9 OneTrans softmax-attention entropy abort 룰의 **첫 active 적용** 결과, smoke 47k rows + 1 epoch + 392 토큰에서 attention entropy [3.49, 3.91] / threshold 5.67. uniform collapse 위험 부재. challengers Frame 1 (paper claim의 100M+ scale dependency) 위험 **우리 데이터 규모에서 발현 안 함**. 룰 retain, 미래 OneTrans-기반 H 들에서 동일 threshold 적용. layer 2 가 layer 1 보다 entropy 높은 패턴은 정상 (layer 깊어질수록 representation 글로벌해짐).
- **F-2 (1 epoch unfair comparison)**: §17.5 sample-scale 룰 정확히 이 상황. PCVRHyFormer 는 organizer-tuned baseline 이라 1 epoch + train_ratio=0.05 envelope 에서도 빠르게 generalize. OneTrans 는 from-scratch single-stream block (~195K 신규 dense params) → 1 epoch 으로 underpowered 가능성 매우 높음. **−0.77pt 가 절대 lift 신호로 받아들여지면 안 됨** — 두 가능성 (underpowered training vs architectural fit) 구별은 epoch ≥ 5 + train_ratio ≥ 0.5 (또는 full-data) 시도로만 가능.
- **F-3 (두 anchor 공존 룰 발동)**: |Δ| = 0.0077 < 0.05 (predictions.md P5 stable threshold). **PCVRHyFormer-anchor (0.8251) 단기 우세**, OneTrans-anchor (0.8174) **archive-pending** (즉시 archive 아님). 미래 H 큐:
  - 단기 (H005-H006): PCVRHyFormer-anchor 위 mutation 우선.
  - 중기 (full-data 도착 시): OneTrans-anchor 재평가, paper claim의 scale dependency 직접 측정.
  - H006+ 부터 카테고리 rotation 의무 풀가동 (`unified_backbones` 4회 연속 차단 룰 활성).
- **F-4 (NS-token expansion 후보 T constraint)**: 메모리 `project_h005_candidate.md` 의 user_ns=8 + item_ns=4 = 12 expansion 은 PCVRHyFormer 의 `d_model % T == 0` 제약 (rank_mixer_mode='full') 위반. T = num_queries × num_sequences + num_ns = 8 + (8+1+4+0) = 21 → 64 % 21 ≠ 0. 단순 구현 불가. T=32 family 진입은 user+item=23 (3x jump) 필요 → §17.2 one-mutation 정신 위배. **carry-forward**: NS-token expansion 은 (i) rank_mixer_mode 동시 변경 (= 2-mutation, 별도 H), 또는 (ii) full-data + d_model 변경 환경에서만 의미. 본 후보 폐기, 메모리 갱신.
- **F-5 (H005 후보 변경)**: F-4 의 결과로 H005 = focal_loss_calibration 으로 promote. §17.4 rotation 첫 충족 (loss_calibration ≠ unified_backbones), §17.2 one-mutation 깔끔 (CLI flag 만 변경, 코드 수정 0), smoke 비용 H001 와 동일.

## Surprises
- attn_entropy 가 예상보다 한참 낮음 (threshold 의 60% 수준). attention pattern 이 매우 sparse 함 — 일부 토큰에 강하게 집중. domain ID embedding + per-domain causal mask 이 attention 을 명확하게 라우팅하는 것으로 추정.
- val_AUC 0.8174 가 PCVRHyFormer 0.8251 와 −0.77pt 차이는 **불공정 비교의 정상적 결과**. PCVRHyFormer 는 organizer 가 hyperparameter + dense params 를 tuning 한 베이스라인 시작점. OneTrans 는 zero-pretrained random-init dense block. 1 epoch 으로 따라잡지 못하는 게 이상한 게 아니라 따라잡았으면 (혹은 능가하면) 그게 더 큰 surprise.
- Wall 5:16 vs E_baseline_organizer 2:47 = +89% 증가. 5%-data smoke 에서도 OneTrans single-stream block 이 PCVRHyFormer 의 query-decoder + RankMixerBlock 보다 약 2x 느림. T=392 token attention 의 비용. full-data 시 4–6x 더 느려질 수 있어 T3 budget 영향 noted.

## Update to CLAUDE.md?
- §10.9 OneTrans softmax-attention entropy abort 룰: **첫 active 적용 + 결과 PASS**. 룰 자체 retain. 본 H 결과로 룰 본문에 "**threshold = 0.95 · log(N_tokens) 기준 통과 시 OneTrans-anchor 자격 측정 신뢰 가능**" 한 줄 carry-forward 후보. 단, 단일 run + smoke 라 두 번째 검증 (full-data 또는 multi-seed) 후 본 룰 확정. **현재는 noted, CLAUDE.md 미반영**.
- 신규 carry-forward rule 후보: "**§17.2 anchor exemption + smoke 결과는 absolute lift 신호 아님; PCVRHyFormer 같은 organizer-tuned baseline 과의 1-epoch 비교는 unfair → soft-warning 수준 −0.5pt 차이는 retire 사유 아님**". F-2 의 generalization. **noted, CLAUDE.md 미반영** (H005/H006 결과 패턴 누적 후 결정).
- F-4 의 NS-token expansion T constraint 는 **CLAUDE.md §10.6 관련 carry-forward** 후보 — "PCVRHyFormer rank_mixer_mode='full' 사용 시 T=num_queries·num_sequences+num_ns 가 d_model 의 약수여야 한다" — 이미 코드 raise로 강제되므로 룰 추가 불필요. 메모리에만 carry-forward.

## Carry-forward to H005

- F-3 → 미래 H 들은 **PCVRHyFormer-anchor (0.8251)** 위에서 mutation 우선. OneTrans 위 mutation 은 full-data 도착 후 재개.
- F-4 → NS-token expansion 후보 폐기. 메모리 `project_h005_candidate.md` replace.
- F-5 → H005 = focal_loss_calibration. loss_calibration 카테고리 (§17.4 rotation 첫 충족), §17.2 깔끔, smoke 빠름.
- F-1 → §10.9 룰의 active 적용 패턴 (`--log_attn_entropy` flag + train.py 진단 forward) 은 미래 OneTrans-기반 H 모두에서 의무 유지.
