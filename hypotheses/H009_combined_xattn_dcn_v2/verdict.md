# H009 — Verdict (REFUTED — interference)

> 클라우드 학습 + inference 완료 (2026-04-30). 두 mechanism 동시 적용이 H008
> 단독보다 **−0.0023pt 악화**. anchor 대비로는 marginal lift 이지만
> strongest-single (H008) paired 비교에서 interference confirmed. predictions.md
> decision tree "interference" 분기 적용.

## Status
`done` — **REFUTED (interference vs H008)**. Platform AUC **0.8364** vs H008 0.8387
**−0.0023pt**. anchor (original_baseline) ~0.83~0.835 paired 시 +0.001~+0.006pt
(boundary). H007 (0.8352) 보다는 +0.0012pt 약간 위 — 두 단독 mechanism 사이
어딘가에서 interference 발현.

## Source data
- 학습: 10 epoch (full, patience=3 trigger 안 됨), train_ratio=0.3, label_time
  split + 10% OOF, **3시간 36분 28초 wall**.
- ckpt: best (epoch TBD — metrics.json 의 best_step 확인).
- Inference: §18 인프라 정상 통과 **259.51초 wall** (H008 220초 대비 +18%, 두
  mutation 추가 capacity 영향).

## P1 — Code-path success
- Measured: 10 epoch NaN-free 완주. CandidateSummaryToken + DCNV2CrossBlock 두
  dispatch 모두 정상.
- Verdict: **PASS**.

## P2 — Primary lift (§17.3 binary)
- Measured: **Platform AUC = 0.83644**.
- OOF AUC: **0.8595**, OOF LogLoss: **0.2310**.
- Δ vs anchor (original_baseline ~0.83X): **+0.001~+0.006pt** (anchor 정확값 의존).
- Δ vs H007 (0.8352): **+0.0012pt**.
- Δ vs H008 (0.8387, strongest single): **−0.0023pt**.
- Predicted (additivity sub-criterion):
  - additive: Δ vs anchor ∈ [+0.005, +0.010pt] → boundary 도달 (anchor=0.83 가정).
  - 실제 paired Δ vs H008: −0.0023pt → **strongest-single paired interference**.
- **Verdict: REFUTED (interference)**. predictions.md decision tree "Δ < anchor"
  엄격 적용은 안 되지만 (anchor=0.83 기준 marginal pass), strongest-single 기준
  으로 명확한 negative — combined 가 H008 단독보다 못함.

## P3 — Mechanism 두 쪽 작동 검증
- Measured: 두 mechanism weight 분포 직접 측정 안 함 (instrumentation 부재).
- Indirect evidence: P2 가 H007/H008 단독 사이에 위치 → 두 mechanism 모두 학습은
  됨. degenerate to identity 아님.
- Verdict: UNVERIFIED. P2 결과만으로 indirect.

## P4 — §18 인프라 통과
- Measured: inference 259초 wall, batch heartbeat + `[infer] OK: torch path
  produced 609197 predictions` 둘 다 보임. heuristic fallback 없음.
- Verdict: **PASS**.

## P5 — Additivity 정량 (보너스)
- H007 단독 lift vs anchor: +0.0035pt.
- H008 단독 lift vs anchor: +0.0035pt (vs H007 +0.0035pt).
- Additive 가정 시 combined: anchor + 0.007pt = 0.840~0.842.
- Measured combined: **0.8364**.
- Δ vs additive 가정: **−0.004~−0.006pt** (anchor 정확값 의존).
- **Classification: interference** (sub-additive 의 worst case — combined <
  strongest single).
- OOF AUC 비교:
  - H008 OOF 0.8585, platform 0.8387 → 갭 1.98pt.
  - H009 OOF **0.8595**, platform **0.8364** → 갭 **2.31pt**.
  - OOF 는 +0.001pt 미세 상승, platform 은 −0.0023pt 하락. **classic overfit
    signature**: OOF cohort fit 좋아졌지만 platform 일반화 악화.

## Findings (F-N carry-forward)

- **F-1 (interference signature confirmed)**: 두 mechanism stacking 이 strongest
  single 보다 못함. OOF +0.001 / platform −0.0023 의 분기 = **classic overfit
  signature**. capacity 증폭이 platform 일반화 악화시키는 패턴 직접 확인.
  carry-forward: 향후 stacking H 는 위치 충돌 분석 의무.
- **F-2 (interference 위치 가설)**: H007 candidate token 이 seq 시작 prepend →
  per-domain seq encoder 출력이 candidate-mixed → DCN-V2 cross 입력의
  effective representation 변경 → polynomial cross 가 candidate-mixed seq 위
  에서 작동. 두 mechanism 이 서로의 input 을 변경. block-level fusion 의 단점.
  carry-forward: candidate xattn + 다른 fusion mechanism stacking 시 candidate
  통합 위치를 fusion 이후로 옮기는 sub-H 후보.
- **F-3 (anchor recalibration 우선순위 강화)**: H006/H007/H008/H009 모두
  extended envelope (10ep × 30%) 측정, anchor 만 smoke (1ep × 5%). H008 의
  +0.0035pt lift 가 mechanism 효과 vs envelope 효과 분리 안 됨. H009 결과로
  anchor 정확값 의존성이 결론 분류 (additive vs interference) 자체를 흔드는 것
  직접 확인 — anchor 정확값 0.83 가정 시 marginal pass, 0.835 가정 시 fail.
  **H010 = anchor recalibration extended** 우선순위 1순위 confirmed.
- **F-4 (H008 still champion)**: Platform 0.8387 여전히 최고. block-level
  fusion swap (interaction axis) 가 우리 베이스라인 + extended envelope 위에서
  단독으로 가장 강한 lever. H010 anchor recalibration 후 H011+ 부터 H008 anchor
  위 single mutation 으로 진행.
- **F-5 (cost — H006~H009 누적 ~14시간)**: H009 wall 3.6시간 (H008 3.7시간 동급
  with patience=3 trigger 안 됨). §17.6 cost cap 압박 지속. H010 도 extended
  envelope 동일 → ~3-4시간 추가 예상.

## Surprises
- **patience=3 가 trigger 안 됨**: H008 F-4 carry-forward 였는데 H009 는 10
  epoch full 학습. 두 mechanism 추가 capacity 가 plateau 늦게 만든 듯. peak
  epoch 정보는 metrics.json 확인 후 보완.
- **OOF 와 platform 갭 다시 벌어짐**: H008 갭 1.98pt → H009 갭 2.31pt. 갭이
  좁아지는 H006→H007→H008 패턴 (3.5pt → 2.5pt → 2.0pt) 깨짐. capacity 증폭이
  cohort effect 다시 키운 것으로 추정.
- **anchor 정확값 의존성 노출**: H006~H008 까지는 결론이 anchor 0.83/0.835
  둘 다에서 같은 PASS/REFUTED 였는데 H009 는 갈림. anchor recalibration 의
  필요성 정량 확인.

## Update to CLAUDE.md?
- §0 P1 룰 ("seq + interaction 한 블록 gradient 공유") 의 한계 발견 — block-
  level gradient 공유가 항상 lift 보장하지 않음. capacity 증폭이 위치 충돌과
  결합되면 interference. **carry-forward 후보**: "**block-level fusion 에서
  여러 mechanism stacking 시 위치 충돌 분석 의무**". H011+ 결과 누적 후 본문
  반영 결정.
- 본문 갱신 보류.

## Carry-forward to H010 (anchor recalibration)

- F-1, F-2 → stacking 의 한계 confirmed. H010 은 mechanism mutation 0 = pure
  envelope mutation 으로 anchor 정확값 확정.
- F-3 → H010 핵심 동기. extended envelope 에서 original_baseline 직접 측정.
- F-4 → H010 결과 후 H011+ 는 H008 anchor (또는 새 anchor) 위 single mutation
  로 진행. interference 회피.
- F-5 → H010 cost ~3-4시간 추가, T2 cap 압박 지속.

## Decision applied (per predictions.md decision tree)

predictions.md table:
- "Δ < anchor → combined 가 baseline 보다 못함 — interference 강함" 분기는
  엄격히 적용 안 됨 (anchor=0.83 기준 marginal pass).
- "Δ sub-additive → mechanism interference. ablation H — H007/H008 단독 다시
  측정 후 통합 위치 재검토" 분기가 더 적합.
- 그러나 **strongest-single (H008) paired 비교** 에서 명확한 negative →
  사용자 합의로 **anchor recalibration H 우선** 으로 결정 (sub-H ablation 보다
  ground truth 확보가 가치 큼).

다음: **H010 = anchor_recalibration_extended**.
