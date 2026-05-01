# H006 — Literature References

## Primary (LongerEncoder 디자인 영감)

- **SIM** (Search-based Interest Modeling) — Pi, Bian, Zhang, et al. CIKM 2020. arXiv:2006.05639.
  - Two-stage: GSU (Generic Search Unit) for candidate-aware top-K retrieval from long history + ESU (Exact Search Unit) for fine-grained scoring.
  - Mechanism: target item embedding 을 query 로 history sequence 에 cross-attention → top-K retrieval.
- **ETA** (End-to-End Target Attention with Long History) — Chen, Zhang, et al. arXiv:2108.04468 (2021).
  - SimHash projection 으로 candidate-aware top-K 선택. paper 의 핵심: random projection 이 retrieve quality 보장.
- **TWIN** (Two-stage Interest Network for User Behavior Modeling) — Chang, Wang, et al. KDD 2023. arXiv:2302.02352.
  - SIM 후속, joint training of GSU + ESU for retrieve-attend pipeline.
- **HSTU** (Hierarchical Sequential Transduction Unit) — Meta, 2024. (FAIR blog).
  - Long-context backbone with hierarchical pruning. industrial-scale recommendation.

## Implementation source

- `competition/model.py:616 LongerEncoder` (organizer-supplied). 위 paper 들의 candidate-aware retrieval 이 아니라 **self-attention probability mass 기반 top-K selection**. paper-grade candidate-aware retrieval 은 target_attention 카테고리 별도 H 자료.
- `competition/model.py:811 create_sequence_encoder` factory. `--seq_encoder_type {transformer, swiglu, longer}` choices.

## Comparison anchor (control)

- **original_baseline** (anchor 2026-04-28). PCVRHyFormer + `--seq_encoder_type transformer` (default).
- val_AUC TBD (anchor measurement 대기).
- 본 H 가 paired 비교 대상. 같은 split, 같은 envelope, **encoder type 만 변수**.

## Carry-forward from prior H (archive 2026-04-28)

- **H001 (E_baseline_organizer)**: PCVRHyFormer + transformer encoder. val=0.8251 (당시 measurement, leakage 가능성). 본 H 의 control 영역과 같은 backbone, encoder 만 변수.
- **H002 (refuted)**: cross-domain mix 메커니즘 sub-block bridge marginal → 본 H 는 cross-domain 안 건드림. layer/token 단위 통합 후보로 carry-forward.
- **H004 (soft-warning)**: OneTrans backbone 자체 작동 검증. full-data 시 OneTrans + LongerEncoder 조합 carry-forward.
- **H005 (refuted)**: focal loss 효과 없음 → BCE 유지. 본 H 도 BCE.

## Carry-forward rules referenced

- **§10.7 카테고리 rotation**: H006 = 첫 long_seq_retrieval. 충족.
- **§17.2 one-mutation**: encoder type 만 변경. ✓
- **§17.3 binary success**: Δ ≥ +0.5pt 임계.
- **§17.5 sample-scale = code-path verification only**: smoke 결과는 mutation effect measurement.
- **§17.6 cost cap**: T2 per-job ≤ $5.
- **§17.7 falsification-first**: predictions.md 에 negative-result interpretation.
- **§17.8 cloud handoff discipline**: training_request.md + flat upload.
- **§18 inference infrastructure (2026-04-28 NEW)**: original_baseline 패키지에서 inherit, 모든 §18.1–§18.5 룰 적용.

## External inspirations (§10.4 P1+ 의무 주입)

- **Switch Transformer load balance loss** (Fedus et al. JMLR 2022) — 본 H 미적용. H007+ carry-forward (external_inspirations 카테고리).
- **TIGER** (Rajput et al. NeurIPS 2023 semantic ID) — P3 phase 후보.

본 H 는 long_seq_retrieval 카테고리 (paper-grade 영감 충분) 라 §10.4 외부 영감 의무 미충족 → H007 carry-forward.

## Future ablation candidates (post-H006)

- **top-K tuning** (50 → 100 → 200): single-mutation, sub-H of H006.
- **per-domain encoder type** (D 만 longer, 나머지 transformer): D-tail 효과 분리, 코드 수정 필요.
- **seq_max_lens 확장** (D=128 → 256 → 512): top-K compression 의 진가 발휘 영역.
- **candidate-aware retrieval** (SIM/TWIN paper 1:1 reproduce): target_attention 카테고리 별도 H.
