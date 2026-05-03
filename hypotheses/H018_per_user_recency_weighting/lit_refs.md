# H018 — Literature References

> H015 lit_refs carry-forward + per-user time-decay 추가 ref.

## Primary

- **Gama, J. et al. 2014** — "A Survey on Concept Drift Adaptation."
  ACM Computing Surveys 46(4). Online learning 의 per-user recency-based
  weight decay (exp form, tau hyperparameter) 의 표준 reference.
  → 본 H 의 exp decay form 직접 motivation.

- **Pan, S.J. & Yang, Q. 2010** — "A Survey on Transfer Learning."
  IEEE TKDE 22(10). Sample weighting in domain adaptation —
  source-target distribution mismatch mitigation.
  → 본 H 의 cohort drift framing (train↔platform distribution shift)
  의 이론적 frame.

- **Sugiyama, M. et al. 2007** — "Covariate Shift Adaptation by
  Importance Weighted Cross Validation." JMLR 8. Importance ratio 기반
  weighting 의 이론적 근거.
  → per-user weight 가 importance ratio 의 implicit form 인 frame.

## Secondary

- **Kang, W.C. & McAuley, J. 2018** — "Self-Attentive Sequential
  Recommendation" (SASRec). Sequential recommendation backbone, time-aware
  variant 들의 base.

- **Sun, F. et al. 2019** — "BERT4Rec: Sequential Recommendation with
  Bidirectional Encoder Representations from Transformer." Sequential
  rec backbone variant.

- **Zhou, G. et al. 2018** — "Deep Interest Network for Click-Through
  Rate Prediction" (DIN). 사용자 historical behavior weighted by attention.

- **Production blog posts** (no formal paper): Meta / Tencent / Google
  의 production CTR systems 에서 per-user time-decay loss weighting 이
  standard. tau ≈ 7~30 일 range 가 일반적.

## H015 carry-forward

- H015 의 source (Pan & Yang, Gama et al., Sugiyama et al.) 그대로 적용.
- H015 = per-batch linear minimum viable form. H018 = per-user exp
  finer specification.
- H015 verdict.md F-N (post submission, 2026-05-03):
  - F-1 expected: per-batch linear granularity 의 marginal Δ +0.0002pt.
    granularity 가 약점일 가능성.
  - → H018 carry-forward: per-user fine 가 직접 attack.

## H016 carry-forward

- H016 verdict (2026-05-03): OOF redefine infra PASS (gap −0.004pt vs
  platform). model lift REFUTED (Δ vs H010 anchor −0.0059pt).
- → H018 carry-forward: H016 redefined OOF 만 신뢰. legacy OOF (saturated
  0.858~0.860) noise 신호로 무시.

## H017 carry-forward (submission lost)

- H017 = H015 sub-form variant (linear → exp form). Triple-H setup
  primary 의 form 변경.
- 현 상태: submission 결과 회수 실패 → no platform AUC.
- → H018 가 per-user × exp 동시 적용. exp form 은 H017 carry-forward
  (한 mechanism class 의 sub-specification, single-mutation 정당, per
  challengers.md ④).

## What's NOT a clone

- 본 H 는 **production CTR per-user time-decay 의 1:1 재현 아님**:
  - production = per-user dataset-wide gap calculation. 본 H = per-batch
    online normalize.
  - production = tau learnable. 본 H = tau=14 fixed (sweep 은 sub-H).
  - production = clip [0.01, 100] wide. 본 H = [0.1, 3.0] tight (variance
    안전망).
  - production = 1B+ user / 1B+ event. 본 H = demo_1000 sample-scale
    sanity + cloud full-data measurement.
