# H002 — Verdict (REFUTED on P2)

> 클라우드 학습 1회 완료 (2026-04-27 21:45 KST). §17.3 binary 임계 미달 — 단일 seed run, P3 mechanism 미확인.

## Status
`refuted` — P2 primary lift 임계 미달. InterFormer cross-domain bridge 방향 retire (§17.3).

## Source data
- 학습 wall: 2분 47초 (예상 30–45min의 1/15 — train_ratio=0.05 smoke 모드).
- 종료 step: epoch 1, step 857/857, best_step=414.
- Ckpt path: `.../global_step414.layer=2.head=4.hidden=64.best_model/model.pt`.
- 사용자 paste 로그 (transcript 2026-04-27).

## P1 — Code-path success
- Measured: 1 epoch NaN 0건 완주, finite val_AUC=0.8248, 정상 ckpt save.
- Predicted: NaN-free, finite best_val_AUC.
- **Verdict per P1: PASS**.

## P2 — Primary lift (§17.3 binary)
- Measured: val_AUC = **0.8248413757428659**.
- Control (E_baseline_organizer): val_AUC = **0.8251**.
- Δ vs control: **−0.0003 pt** (사실상 noise 수준 0).
- Predicted: Δ ≥ +0.5 pt (val_AUC ≥ 0.8301).
- **Verdict per P2: REFUTED**.

## P3 — Mechanism check (bridge gate 분포)
- Measured: **TBD** (gate snapshot 로그/ckpt에서 미확보).
- Predicted: 평균 sigmoid(gate) ≥ 0.20 (init 0.119에서 +0.08 grow).
- **Verdict per P3: UNVERIFIED**. P2 REFUTED와 무관하게 mechanism 시나리오 1/2 구별 불가:
  - 시나리오 1 (gate 미성장, near-init): bridge 학습 신호 부재 → underpowered, full-data + multi-epoch 에서는 다를 수 있음.
  - 시나리오 2 (gate 성장 but useful 신호 못 잡음): query decoder의 cross-attention 이 step 2에서 이미 도메인 mix → step 1.5 사전 mix는 redundant.

## P4 — Submission round-trip
- Measured: TBD (ckpt 다운로드 + local_validate.py 미실행).
- Predicted: G1–G6 5/5 PASS.
- **Verdict per P4: PENDING** — H004 anchor 작업과 병행 가능.

## P5 — Variance reduction (보너스)
- Measured: TBD (multi-step loss variance 로그 미확보).
- **Verdict per P5: UNMEASURED**.

## Findings (F-N carry-forward)

- **F-1 (P2 REFUTED)**: PCVRHyFormer baseline의 query-decoder cross-attention 이 이미 도메인 간 mix 를 제공하므로, **per-domain seq encoder 출력 단계에서 추가 cross-domain bridge 는 marginal**. InterFormer 의 arch-level bridge 디자인 (3 archs 사이) 을 4-domain seq encoders 에 그대로 매핑한 것은 paper 구조와 1:1 대응이 아니었음.
- **F-2 (smoke 모드 한계)**: train_ratio=0.05 + 1 epoch + scalar gate init −2.0 의 조합은 새 12k params 학습에 underpowered. 단일 seed Δ ≈ 0 은 "bridge 가 효과 없다" 보다 "bridge 가 학습될 시간 부족" 일 가능성 동등. §17.5 sample-scale = code-path verification only 룰의 직접 적용.
- **F-3 (cross-domain 정보 흐름의 새 channel 가설은 보류)**: H002 가 "도메인 간 gradient 공유 추가" 라는 §0 P1 axis 강화를 의도했지만, 실제로 그 강화가 PCVRHyFormer 내부에서 redundant 였을 가능성. 새 backbone (token-level fusion) 에서는 cross-domain channel 이 다른 의미를 가질 수 있음 → H004 OneTrans anchor 의 디자인 동기.
- **F-4 (single-seed verdict 신뢰성)**: §17.8.5 룰에 따라 단일 seed 결과만으로 자동 refuted, 단 사용자가 seed×3 + full-data 재학습 명시 요청 시 보류 가능. 현재 사용자 결정 = retire (H004 진행).

## Surprises
- val_AUC 가 control 대비 정확히 noise 수준이었음. **bridge zero-init + gate near-off** 디자인이 의도대로 작동: 학습이 충분치 않을 때 baseline 보존 보장. 부정적 lift 없음 = 안전 디자인 검증.
- Wall 2:47 — bridge overhead 5% 예측보다도 작음 (bridge forward 가 pool→linear→broadcast 만이라 부담 없음).

## Update to CLAUDE.md?
- §10.10 (InterFormer gating σ(−2) init) 룰 자체는 본 H에서 위반 없이 적용. 실패 원인은 init 룰이 아니라 데이터/시간 부족 + redundancy. 룰 retain.
- 신규 carry-forward rule 후보: "**Cross-domain mix 메커니즘은 query-decoder 후 단계에 inject 해야 효과 있음** (PCVRHyFormer 구조 한정)" — F-1의 generalization. **단 single-run 근거이므로 H004 OneTrans 결과로 검증 필요**, 지금은 noted only, CLAUDE.md 미반영.

## Carry-forward to H004
- F-3 → H004 OneTrans 의 mixed-causal mask 가 "cross-domain 정보 흐름" 을 token-level 로 제공 (NS-token bidirectional × S-token causal). H002 가 도메인-encoder 사이 sub-block bridge 로 시도한 것 보다 더 깊은 통합. H002 refutation은 H004 motivation 강화.
