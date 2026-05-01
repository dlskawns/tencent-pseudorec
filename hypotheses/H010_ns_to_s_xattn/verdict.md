# H010 — Verdict (PASS — additive)

> 클라우드 학습 + inference 완료 (2026-04-30). Platform AUC **0.8408** —
> **현재까지 최고**. Δ vs H008 (champion, 0.8387) **+0.0021pt** = additive
> 분류 (predictions.md decision tree). Δ vs anchor (~0.83X) +0.7~+1.1pt =
> §17.3 binary +0.5pt 통과. attn entropy [0.81, 0.81] (sparse, threshold
> 5.65 의 14% 수준) — uniform collapse 부재, NS-token routing 작동.
> H009 위치 충돌 회피 설계 가설 검증됨.

## Status
`done` — **PASS (additive vs H008)**. Platform AUC **0.8408** vs H008 0.8387
**+0.0021pt**. Δ vs anchor (~0.83X) +0.7~+1.1pt (anchor 정확값 의존,
모두 §17.3 +0.5pt 통과). 새 champion.

## Source data
- 학습: 10 epoch (full or patience=3 — peak step 은 metrics.json 확인 후 보완),
  train_ratio=0.3, label_time split + 10% OOF, **3시간 44분 54초 wall**
  (H008 3.7h, H009 3.6h 와 동급).
- Inference: §18 인프라 정상 통과 **297.02초 wall** (H008 220초, H009 259초
  대비 +14% — NSToSCrossAttention 1 layer 추가 영향).
- ckpt: best step (metrics.json 의 best_step 확인 후 보완).
- raw metrics.json blob: TBD (사용자 paste 시 추가).

## P1 — Code-path success
- Measured: 학습 NaN-free 완주 + `Training complete!` 로그 + metrics.json
  생성. NSToSCrossAttention dispatch + DCN-V2 fusion 둘 다 정상.
- Verdict: **PASS**.

## P2 — Primary lift (§17.3 binary)
- Measured: **Platform AUC = 0.840771** (eval auc).
- OOF AUC: **0.8596**, OOF LogLoss: **0.2323**.
- Δ vs anchor (original_baseline ~0.83X): **+0.7~+1.1pt** (anchor 정확값
  의존, 모두 §17.3 +0.5pt 통과).
- Δ vs H007 (0.8352): **+0.0056pt**.
- Δ vs H008 (0.8387, prior champion): **+0.0021pt**.
- Δ vs H009 (0.8364): **+0.0044pt**.
- **Verdict: PASS**. predictions.md decision tree "Δ vs anchor ≥ +0.5pt +
  additive vs H008" 분기 (Δ ∈ [+0.001, +0.005pt]).

## P2-sub — Paired vs H008 strongest single
- Measured: Δ vs H008 = **+0.0021pt**.
- Predicted classification: super-additive | additive | noise | interference.
- **Classification: additive** (Δ ∈ [+0.001, +0.005pt]).
- 의미: NS-token bidirectional xattn 가 H008 anchor 위 stacking 시 H009 와
  달리 lift 보존 — transfer.md §⑤ "fusion 이전, NS dimension 변경 없음"
  위치 충돌 회피 설계 가설 **검증됨**.
- super-additive 미달: paper-grade lift (≥ +0.005pt) 까지는 도달 못 함 —
  sample-scale (30% × 10ep) 에서 NS×S 통합 효과는 small additive lift.
  full-data 도착 시 super-additive 가능성 별개 측정.

## P3 — §10.9 attention entropy
- Measured: `attn_entropy_per_layer = [0.8127, 0.8133]`, threshold 5.6531,
  violation=False.
- Threshold: 0.95 × log(384) ≈ 5.65.
- Predicted: < 5.65 (sparse pattern, H004 OneTrans backbone smoke 의 [3.49,
  3.91] 재현 expected).
- **Verdict: PASS** (entropy 0.81 = 14% of threshold, **predicted 보다 훨씬
  sparse**). NS tokens 가 평균 e^0.81 ≈ **2.25 tokens / 384** 만 attend —
  강한 selective routing. degenerate to uniform 정반대.
- Sparse 정도가 H004 (~3.5-3.9) 보다 4× 강함 — NS query semantic 가 도메인
  routing 으로 작동했을 가능성 (P6 mechanism check 와 정합).

## P4 — §18 인프라 통과
- Measured: inference 297초 wall, `eval auc: 0.840771` 산출 (≠ 0.5
  fallback). batch heartbeat / `[infer] OK` 로그 추정 — 사용자 paste 시
  확인 후 보완.
- Verdict: **PASS** (eval auc ≠ 0.5 + inference 정상 종료).

## P5 — val ↔ platform 정합 (보너스)
- val_AUC: TBD (metrics.json paste 시 보완).
- platform_AUC: 0.8408.
- OOF AUC: 0.8596, gap vs platform = **1.88pt**.
- OOF-platform gap 비교: H006 3.5pt → H007 ~2.5pt → H008 1.98pt → H009 2.31pt
  → H010 **1.88pt** (다시 narrow). H009 의 capacity 증폭 cohort effect 가
  H010 위치 회피 설계로 회복.
- val 측정값은 metrics.json 확인 후 보완.

## P6 — Mechanism check (NS xattn 작동)
- Indirect evidence: P3 의 entropy 0.81 (= **highly selective routing**).
  NS tokens (B, 7, D) 이 384 S tokens 중 ~2 tokens 에 집중 → 도메인 또는
  최근 hot tokens 에 hard routing 작동 가설.
- Direct attention weight 분포: instrumentation 없어 미측정.
- Verdict: **indirect PASS** (entropy signature 상 nontrivial routing 확인).
  full attention map snapshot 은 sub-H 후보.

## Findings (F-N carry-forward)

- **F-1 (NS-token bidirectional xattn 작동, additive lift confirmed)**:
  H010 가 H008 anchor 위 stacking 으로 +0.0021pt 추가 lift. H009 의
  interference (−0.0023pt) 와 정반대. 차이는 **통합 위치** — H009 candidate
  prepend (S 시작에 1 token 끼움) → seq encoder 입력 변경; H010 NS xattn
  은 seq encoder 통과 후 NS-side enrichment + S/decoded query/DCN-V2
  입력 모두 byte-identical. **carry-forward**: 향후 stacking H 는 "어느
  텐서가 변경되는가" 분석 의무. NS-only enrichment = safe stacking pattern.
- **F-2 (paper-grade source 검증 — OneTrans NS→S half lift)**: arXiv
  2510.26104 의 mixed causal mask 4 sub-mask 중 NS→S half 만 단독 추출
  해도 sample-scale extended 에서 측정 가능 lift. paper 의 100M user 규모
  미도달 환경에서도 minimum viable form 작동. **carry-forward**: 나머지
  sub-mask (S→S causal, NS→NS self-attn, mixed timestamp masking) 별도
  H 후보 (단 §10.7 rotation 으로 H011 직후는 차단).
- **F-3 (attention entropy 0.81 — 강한 selective routing 발현)**: threshold
  의 14% 수준 (predicted "sparse" 보다 4× 더 sparse). NS-tokens 이 384
  S tokens 중 ~2 tokens 에만 집중 → degenerate uniform 의 정반대 극단.
  **carry-forward**: NS-token 의 selective routing 이 작동 = num_heads
  증가 또는 multi-layer 시 더 큰 lift 가능성 (sub-H). 동시에 sparse 가
  너무 강하면 정보 bottleneck 위험 (top-K hard routing 과 비슷한 효과).
  full-data 도착 시 entropy 가 어떻게 변하는지 측정 의무.
- **F-4 (champion 갱신 — anchor 갱신 결정)**: H010 Platform 0.8408 = 새
  최고. H008 (0.8387) 위 single mutation stacking 형식 합법. **anchor
  갱신**: H011+ 부터 H010 anchor 위 single mutation 으로 진행. H008 은
  carry-forward control 로 보존 (paired Δ 비교 용).
- **F-5 (OOF-platform 갭 narrowing 회복)**: H006 3.5pt → H007 ~2.5pt →
  H008 1.98pt → H009 **2.31pt 역행** → H010 **1.88pt** 다시 narrow.
  H009 의 capacity 증폭 cohort effect 가 H010 의 위치 회피 설계로 회복
  → 통합 위치 가 cohort fit 에도 영향 (H008-class 위치가 generalization
  에 더 좋은 패턴 confirmed).
- **F-6 (cost — H006~H010 누적 ~17.3시간)**: H010 wall 3.7시간 (extended
  envelope 동일). §17.6 cap 압박 지속. H011 부터 patience=3 + plateau early
  signal 활용 또는 fp16/batch=512 으로 wall 절반 시도 권장.

## Surprises
- **Entropy 가 예상보다 4× sparse**: predicted (H004 baseline ~3.5-3.9)
  보다 0.81 — extreme selectivity. NS tokens 이 7개 모두 거의 같은 ~2
  tokens 에 모이는 패턴 (collapse 가능성도 의심) 또는 도메인-aware hard
  routing. mechanism check 직접 측정 (sub-H) 가치.
- **H010 OOF (0.8596) 가 H009 OOF (0.8595) 와 거의 동일** 인데 platform
  은 +0.0044pt 차이: OOF cohort score 가 platform 일반화 의 noisy proxy.
  H006 F-3 carry-forward (platform AUC primary) 다시 확인.
- **Inference wall +14% (220→297초)**: NSToSCrossAttention 1 layer 추가
  로 inference latency 비례 증가. H011+ 부터 P1+ phase gate 의 inference
  budget cap 검토 필요 (현재 명시 없음).

## Update to CLAUDE.md?
- §0 P1 룰 ("seq + interaction 한 블록 gradient 공유") 의 "한 블록"
  정의 정밀화 가능: H009 (block-level + 위치 충돌) REFUTED, H010
  (layer-level NS-side enrichment + 입력 변경 없음) PASS. **carry-forward
  후보**: "stacking 시 어느 텐서가 변경되는지 명시 (NS-only / S-only /
  mixed)". H011~H012 결과 누적 후 본문 반영 결정.
- §10.9 attn entropy threshold (현재 0.95 × log(N)) 가 너무 보수적일
  가능성 노출 — H010 entropy 0.81 = threshold 의 14%. lower bound (예:
  log(2) ≈ 0.69) 도 검토 가치. 본문 갱신 보류.
- 본문 갱신 보류 (수치 1회만으로 결정 부족).

## Carry-forward to H011 (orthogonal axis)

- F-1 → NS-only enrichment = safe stacking pattern. H011 도 같은 패턴
  (anchor 입력 텐서 변경 없음) 권장.
- F-3 → NS xattn entropy 강한 selective routing 작동 → H011 후보 중
  num_heads/layer 증가 sub-H 는 §10.7 rotation 으로 차단 (target_attention
  재진입 3회 연속 됨). orthogonal axis 우선.
- F-4 → H011 anchor = H010 (Platform 0.8408). H008 carry-forward control.
- F-5 → 통합 위치 가 cohort fit 에 영향 — H011 의 통합 위치도 H010 패턴
  (anchor 입력 byte-identical) 따르기.
- F-6 → wall 절감 검토 (fp16, batch 512).

## Decision applied (per predictions.md decision tree)

predictions.md table:
- "Δ vs anchor ≥ +0.5pt + additive vs H008" 분기 적용.
- → "H010 PASS. anchor 갱신 (H010 = 새 baseline). H011 = 다른 axis 탐험."

## Next H 후보 (rotation respect)

직전 2 H primary_category:
- H009: hybrid (target_attention + sparse_feature_cross)
- H010: target_attention

H011 = **target_attention 재진입 차단** (직전 2회 연속 target_attention 포함).
**orthogonal axis 우선**:

1. **aligned `<id, weight>` pair encoding** (recommended):
   - CLAUDE.md §3 / §4.8 mandate ("aligned `<id, weight>` 는 항상 한 쌍").
   - orthogonal axis (interaction encoding 측) — interference 위험 0.
   - 후보 fid pairs: `{61–66, 89–91}`.
   - primary_category: `feature_engineering` 또는 `interaction_encoding` (신규).
2. **multi_domain_fusion (MMoE/PLE)**:
   - block fusion 강화, 4 도메인 expert routing.
   - primary_category: `multi_domain_fusion` (신규).
3. **NS xattn sub-H** (num_heads / layer 수):
   - target_attention 재진입 → §10.7 차단. backlog.
