# H011 — Verdict (REFUTED — degraded vs H010 anchor)

> 클라우드 학습 + inference 완료 (2026-05-01). Platform AUC **0.8347** vs
> H010 anchor 0.8408 **−0.0061pt** → predictions.md decision tree "degraded"
> 분기. OOF AUC −0.0007 / Platform −0.0061 = OOF 거의 보존, Platform 만
> 악화. attn_entropy 거의 동일 (0.813 vs H010 0.8127-0.8133) → mechanism
> dispatch 정상, 하지만 H010 NS xattn 의 selective routing 을 흔들지 못함.
> input-stage L1-norm weighted multiply (Option α) 가 baseline mean-pool
> 보다 platform generalization 악화.

## Status
`done` — **REFUTED (degraded vs H010 anchor)**. Platform AUC **0.8347** vs
H010 0.8408 **−0.0061pt**. Δ vs H008 (carry-forward control 0.8387)
**−0.0040pt**. Δ vs original_baseline (~0.83X) ≈ +0.4~+0.5pt (anchor
정확값 의존, marginal absolute lift 만).

## Source data
- 학습: 10 epoch, train_ratio=0.3, label_time split + 10% OOF, **2시간 46분
  54초 wall** (H010 3:44:54 대비 **−25% 빠름** — params 추가 0 + simpler
  weighted-mean).
- Inference: §18 인프라 정상 통과 **178.31초 wall** (H010 297초 대비
  **−40% 빠름** — H010 의 NS xattn layer cost 그대로지만 input 단계가
  parameter-free 라 inference 부하 작음).
- ckpt: best step (raw metrics.json paste 도착 시 보강).
- 학습 wall 단축은 일관성 있음 — params 0 추가 + L1 norm 곱셈만 추가.

## P0 — Audit gate ✅ PASS (pre-train)
- `eda/out/aligned_audit.json` PASS (8 fids 모두 1000/1000, position-wise).
- `eda/out/dense_value_stats.json` PASS — Option α 선택 근거 확보.

## P1 — Code-path success
- Measured: 학습 NaN-free 완주 (clamp(1e-8) zero-row safety 작동 추정,
  로그 paste 시 확인). `Training complete!` 로그 + metrics.json 생성.
- Verdict: **PASS**.

## P2 — Primary lift (§17.3 binary)
- Measured: **Platform AUC = 0.834699** (eval auc).
- OOF AUC: **0.8589**, OOF LogLoss: **0.2342**.
- Δ vs H010 anchor (0.8408): **−0.0061pt**.
- Δ vs H008 carry-forward (0.8387): **−0.0040pt**.
- Δ vs H007 (0.8352): −0.0005pt.
- Δ vs H009 (0.8364): −0.0017pt.
- Δ vs original_baseline (~0.83X): +0.4~+0.5pt (anchor 정확값 의존).
- Predicted classification (predictions.md):
  - super-additive ≥ +0.005pt: 미달.
  - measurable [+0.001, +0.005pt]: 미달.
  - noise (−0.001, +0.001pt]: 미달.
  - **degraded < −0.001pt**: 적용 (Δ vs H010 = −0.0061).
- **Verdict: REFUTED (degraded)**.

## P3 — NS xattn entropy 변화 (mechanism check)
- Measured: `attn_entropy_per_layer = [0.8130, 0.8132]`.
- H010 baseline: [0.8127, 0.8133].
- Δ: ~0 (변화 미세, 0.001 이하).
- Predicted classification: 더 sparse | **변화 미세** | 더 uniform.
- **Verdict: 변화 미세** — H011 의 weighted multiply 가 NS-token 표현을
  변경했지만 NS xattn 의 selective routing pattern 자체는 거의 바꾸지
  못함. predictions.md "Frame B" 신호 (NS-level binding 이 이미 충분).

## P4 — §18 인프라 통과
- Measured: inference 178초 wall, eval auc 0.834699 산출 (≠ 0.5 fallback).
- Verdict: **PASS** (eval auc ≠ 0.5 + inference 정상 종료).

## P5 — val ↔ platform 정합 (보너스)
- val_AUC: TBD (raw metrics.json paste 시 보강).
- platform_AUC: 0.8347.
- OOF AUC: 0.8589, gap vs platform = **2.42pt**.
- OOF-platform gap 비교: H006 3.5 → H007 ~2.5 → H008 1.98 → H009 2.31 →
  H010 1.88 → H011 **2.42pt** (다시 벌어짐).
- H011 의 OOF 거의 보존 (−0.0007) + platform 큰 폭 하락 (−0.0061) =
  **classic overfit signature** (H009 와 같은 패턴). H011 의 input-stage
  weighted multiply 가 cohort fit 은 유지 / generalization 악화.

## P6 — OOF-platform gap (보너스)
- Measured: 2.42pt.
- Predicted: ≤ 2pt. 미달 (H010 1.88 → 2.42).
- Capacity-overfit cohort effect 재발 (H009 패턴).

## Findings (F-N carry-forward)

- **F-1 (input-stage aligned encoding REFUTED)**: per-row L1-normalized
  weighted multiply 가 H010 anchor 위에서 −0.0061pt 악화. OOF (−0.0007)
  거의 보존, platform (−0.0061) 큰 폭 하락 = **classic overfit signature
  재발** (H009 와 같은 패턴). carry-forward: input-stage modify 가 cohort
  fit 유지하면서도 platform 악화 → mechanism class 자체가 wrong direction
  가능성.
- **F-2 (NS xattn routing 거의 변화 없음)**: attn_entropy 변화 ~0.001
  (H010 [0.8127, 0.8133] → H011 [0.8130, 0.8132]). predictions.md "Frame
  B" 신호 (NS-level binding 이 이미 충분 학습). H010 의 selective routing
  (entropy 0.81 = 384 tokens 중 ~2 만 attend) 이 H011 의 input-stage
  modify 흡수해버림. carry-forward: NS xattn 이 이미 dominant signal 잘
  찾고 있음 — input-stage variance 는 NS 가 이미 implicit 학습.
- **F-3 (scale handling 결정 잘못됐을 가능성)**: Option α (per-row L1
  norm) 이 Pattern X (count, max=18M) 와 Pattern Y (signed [-1,+1]) 통일
  처리. Pattern X 의 magnitude 정보 손실 (count 5 vs 5M 모두 normalize 후
  같은 row sum), Pattern Y 의 sign 도 strict sum-to-1 로 distortion. carry-
  forward: 두 pattern 별로 다른 scale handling sub-H 후보 (Pattern X = log1p,
  Pattern Y = raw 또는 abs).
- **F-4 (training wall −25% / inference −40%)**: params 추가 0 + simpler
  weighted-mean 으로 학습 빨라짐. positive 측면 — 향후 H 들도 input-stage
  parameter-free mutation 은 cost-effective 하다는 증거.
- **F-5 (anchor 갱신 안 함)**: H011 REFUTED → anchor 유지 H010 (Platform
  0.8408). H012+ 는 H010 anchor 위 single mutation 으로 진행. H008 carry-
  forward control 도 유지.
- **F-6 (cost — 누적 ~21시간)**: H006~H011 누적 ~21h. §17.6 cap 압박
  지속. H012 부터 fp16/batch=512 또는 train_ratio=20% 권장.

## Surprises
- **OOF 거의 보존 + Platform 큰 폭 하락**: H009 (combined) 와 같은 패턴
  재발. 두 mechanism 위치가 다른데 (H009 candidate prepend / H011 input
  embedding) 둘 다 같은 overfit signature → cohort effect 가 mechanism
  공간 어디서든 발현될 수 있다는 신호. **carry-forward**: 사용자 cohort
  drift 가 platform 일반화의 hard ceiling 일 가능성. 향후 H 들의 mechanism
  설계와 별개로 cohort 처리 자체가 별도 H 가치.
- **NS xattn entropy 거의 변화 없음**: H010 의 selective routing 이 NS-
  token 표현 변화에 robust. H010 의 sparse routing pattern 이 input
  variance 를 implicit 으로 흡수 — strong signal 이지만 H011 mechanism 이
  redundant 라는 의미.
- **Training/inference wall 단축**: 예상보다 큰 gain (학습 −25%, infer
  −40%). H010 의 NS xattn cost 가 inference 의 dominant 였는데 H011 이
  그대로 유지하고도 줄어듦 → 환경 노이즈 또는 cluster load 차이 가능성
  (Taiji wall 의 일관성 관찰 가치).

## Update to CLAUDE.md?
- §0 P1 룰 ("seq + interaction 한 블록 gradient 공유") 의 한계 또 노출 —
  block-level gradient 공유 강도가 lift 와 단조 관계 아님. H009 (block-
  level), H011 (input stage = strongest gradient 공유) 둘 다 REFUTED.
- §3 의 user_dense 3-pattern 분류 + Option α 선택 기록은 향후 H 의 input-
  stage mutation 시도 시 negative carry-forward 로 인용 가능.
- 본문 갱신 보류 (수치 1회만으로 결정 부족 — H013/H014 누적 후 결정).

## Carry-forward to H012

- F-1 → input-stage modify 의 mechanism class 회피 권장. 다른 axis 우선.
- F-2 → H010 NS xattn 이 이미 dominant signal 학습 — NS xattn sub-H (num
  layers, num heads) 도 marginal 가능성.
- F-3 → 만약 H012 이 input-stage 재시도 한다면 multi-form sub-H (Pattern
  X log1p / Pattern Y raw) 로 진행. 단 차단 카테고리 (feature_engineering
  재진입).
- F-5 → H012 control = H010 (Platform 0.8408). H008 carry-forward.
- F-6 → cost 절약 검토 (fp16, batch=512).

## Decision applied (per predictions.md decision tree)

predictions.md table:
- "Δ vs anchor < −0.001pt (degraded)" 분기 적용.
- → "REFUTED + scale/매핑 issue. retract 또는 sub-form (sigmoid 게이팅, log1p)."
- card.yaml decision_tree_post_result: `degraded` 분기 — anchor 갱신 안 함.

## Next H 후보 (rotation respect)

직전 2 H primary_category:
- H010: target_attention.
- H011: feature_engineering.

H012 차단 = `feature_engineering` (직전 1회). `target_attention` 도 직전
2 turn 안에 있어 재진입 시 정당화 필요.

**추천 = H012 = `multi_domain_fusion` (MMoE/PLE)**:
- 신규 카테고리 first-touch (§10.7 FREE).
- §0 north star: 4 도메인 (a/b/c/d) 길이/vocab/frac_empty 차이 큼 (§3.5).
  expert routing 자연 motivation.
- H010 anchor 위 single mutation (NS xattn 출력 → expert routing → DCN-V2).
- H011 의 negative carry-forward F-2: NS xattn 이 dominant signal 학습 중
  → 그 출력에 explicit routing 추가 가 보완적.

대안:
- H012 = **CAN co-action** (sparse_feature_cross 변형). category 직전 2
  turn 안엔 없음 (H008 이 마지막). 안전.
- H012 = NS xattn sub-H (multi-layer 또는 num_heads 증가) → target_attention
  재진입, 정당화 필요. F-2 신호상 marginal 가능성 큼.
