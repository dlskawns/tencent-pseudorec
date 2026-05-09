# H041 — Verdict (REFUTED 2026-05-07)

## Status
`REFUTED` — Val/OOF 둘 다 H019 anchor 대비 marginal degraded.
Cloud submit 안 함 (platform AUC 회수 무가치 — 이미 Val/OOF 둘 다 음의 Δ).
R2 reframe (cold-start dual branch) **archive — data signal mapping wrong**.

## Source data (cloud T2.4, seed=42, 2026-05-07)

| Epoch | Train Loss | Val AUC | Val LogLoss |
|---|---|---|---|
| 1 | — | 0.83051 | 0.28637 |
| **2** | **0.15937** | **0.83496** ★ | **0.28210** ★ |
| 3 | 0.15887 | 0.83457 | 0.28218 |
| 4 | 0.15737 | 0.83465 | 0.28359 |
| 5 | 0.15674 | 0.83465 | 0.28386 |

→ Best at epoch 2 (vs H019 ep4) — plateau 더 빠름. early-stop ep5 (vs H019 ep7).

| Metric | H041 | H019 anchor | Δ |
|---|---|---|---|
| Best Val AUC | 0.83496 | 0.83720 | **−0.00224pt** |
| Best Val LogLoss | 0.28210 | 0.28080 | +0.00130 |
| OOF AUC | 0.8601 | 0.8611 | **−0.00100pt** |
| OOF LogLoss | 0.2308 | 0.2309 | −0.00010 |
| Epochs trained | 5 | 7 | −2 (early plateau) |

## Paired Δ verdict
- **Val Δ = −0.00224pt** → §17.3 binary cut < +0.001pt → **REFUTED**, decision tree row 4 (degraded/unstable).
- **OOF Δ = −0.00100pt** → noise/marginal degraded, decision tree row 3 (gate not learned).
- 두 신호 결합 → **gate 가 거의 안 움직임 (init 0.88 부근) + cold_clsfier random init noise 가 main classifier 학습에 약한 negative perturbation**.

## Diagnosis (root cause of REFUTED)

R2 reframe 자체 결함:
- §3.5 `target_in_history=0.4%` 의 올바른 해석 = "**100% prediction 이 cold regime**"
  (familiar 0.4% 도 사실은 1번 본 item 의 예측, 진정한 familiar 아님).
- 즉 dual branch 의 split boundary 가 데이터에 없음 — splittable axis 부재.
- dual classifier 는 split 할 게 있어야 specialize 하는 mechanism.
- §0.5 step 2 (diagnose root cause) 잘못 → 0.4% 를 "familiar" 로 오해.

## P1 — Code-path success
- ✅ T0 sanity PASS (model.py byte-diff 만, trainer.py/dataset.py/infer.py md5 H019 동일).
- ✅ Cloud full-data 5 epoch 수렴 (early stop trigger 정상).

## P2 — Primary lift (§17.3 binary)
- **REFUTED** — Δ < +0.001pt 양 axis.

## P3 — Mechanism 작동 검증
- gate 학습 distribution logging 미수집 → 정확한 gate 분포 미상. 단 Val/OOF 둘 다 음의 Δ + epoch 2 plateau → gate 학습 부재 강한 정황 증거.

## P4 — §18 인프라 통과
- ✅ §18.7 + §18.8 H019 carry.

## Findings (F-N carry-forward)

- **F-H (data-signal driven mechanism도 mapping 결함 시 fail)** — H041 이 첫 사례. CLAUDE.md §0.5 step 2 (diagnose root cause) 의 정밀도 부족 시 mechanism 설계 자체 자기-부정. 향후 H 에서 data signal interpretation 의 *명시적 multiple-frame validation* 필요 (이번 0.4% 가 "split" 인지 "regime fraction" 인지 구분 안 함).
- **F-G 추가 confirm** — H041 Val 0.83496 도 12+ H ceiling band 안. mechanism class change 없는 axis 추가는 ceiling 못 깨고 noise 만 추가.

## Surprises
- Cold-start dual branch 의 epoch 2 plateau — H019 의 epoch 4 best 와 비교 시 학습 dynamics 자체 변경. 단순 capacity 추가 효과로는 설명 안 됨 (4.4K params = total 0.003%).

## Update to CLAUDE.md?
- §0.5 추가 후보 — "data signal interpretation multi-frame validation 필수" rule.
- §10 anti-bias rule 추가 후보 — "fraction-based signal 의 mechanism 매핑 시 *split axis* 가 데이터에 실재하는지 별도 검증".

## Carry-forward to 다음 H
- **R2 axis retire** — cold-start dual branch sub-H 만들지 말 것.
- **R1 (no-history baseline, H039)** 우선 — F-G ceiling 의 floor 측정 = "history 가 lever 인가" 라는 더 본질적 질문.
- **R4 (BPR pairwise, H040)** 그 다음 — loss-objective axis (14 H 모두 BCE blind spot).
- **다른 axis 후보**: capacity (depth/width unexplored), cohort drift 직접 attack, KLD prior matching (class prior 12.4% ↔ predicted prob 분포).

## Decision applied (per predictions.md decision tree)
- Row 4 (degraded/unstable): **R2 axis retire**. cloud submit 절대 안 함 (slot 보호).
