# H007 — Literature References

## Primary mechanism class (candidate-as-attention-query)

- **DIN** (Deep Interest Network) — Zhou, Zhu, Song, Fan, Zhu, Ma, Yan, Jin, Li, Gai. KDD 2018. arXiv:1706.06978.
  - Origin paper of candidate-aware target attention in CTR.
  - Mechanism: candidate item ID embedding → element-wise activation MLP weighting over history items → weighted sum.
  - 우리는 idea 차용, 2018 archaic activation MLP 대신 modern multi-head cross-attention 사용.
- **DIEN** (Deep Interest Evolution Network) — Zhou et al. AAAI 2019. arXiv:1809.03672.
  - DIN + GRU sequential evolution. session-level interest tracking.
- **DSIN** (Deep Session Interest Network) — Feng et al. IJCAI 2019.
  - DIN + session-aware split.
- **CAN** (Co-action Network) — Bian, Zhao, Liu, et al. KDD 2022. arXiv:2011.05625.
  - Candidate × history co-action matrix → MLP. dimensionally richer than dot-product attention.
  - 미채택 (별도 H 후보).
- **SIM** (Search-based Interest Modeling) — Pi, Bian, Zhang, et al. CIKM 2020. arXiv:2006.05639.
  - GSU (Generic Search Unit) for candidate-aware long-history retrieval. ESU (Exact Search Unit) for fine scoring.
  - 미채택 (별도 H 후보, long_seq_retrieval 카테고리).
- **TWIN** (Two-stage Interest Network) — Chang, Wang, et al. KDD 2023. arXiv:2302.02352.
  - SIM + ESU joint training.
- **HSTU** (Hierarchical Sequential Transduction Unit) — Meta 2024.
  - Hierarchical candidate-aware multi-layer attention. industrial scale.
- **OneTrans** (Single-stream Transformer) — Tencent UNI-REC, WWW 2026. arXiv:2510.26104.
  - Single-stream + candidate token + mixed-causal mask. 본 H 의 candidate token 디자인 영감.

## Implementation source

- 신규 클래스 `CandidateSummaryToken` (model.py 확장).
- modern transformer convention: Pre-LN + multi-head cross-attention + residual + output projection.
- 코드 ~80줄 추가 + PCVRHyFormer 통합 ~30줄.

## Comparison anchor (control)

- **original_baseline** (anchor 2026-04-28). PCVRHyFormer + transformer encoder + BCE + smoke envelope. Platform AUC ~0.83X.
- 본 H 가 paired 비교 대상. 같은 split, 같은 envelope, **`use_candidate_summary_token` flag 만 변수**.

## Carry-forward from prior H (archive 2026-04-28)

- **H001 (E_baseline_organizer)**: PCVRHyFormer + transformer encoder. anchor 의 직접 ancestor.
- **H002 verdict F-1**: cross-domain mix 메커니즘 sub-block bridge marginal — 본 H 는 cross-domain 안 건드림. layer/token 단위 통합 후보 carry-forward.
- **H004 verdict F-1 (P3 PASS)**: OneTrans candidate token + softmax routing sample-scale 작동 검증. 본 H 가 더 minimal viable form.
- **H005 verdict F-1**: BCE 12% imbalance 영역 충분 → 본 H BCE 유지.
- **H006 verdict F-1**: random/probability-based selection 한계 — 본 H 가 candidate-aware 로 mechanism class 이동.
- **H006 verdict F-3**: paired Δ 는 **platform AUC 으로만**. 본 H 도 platform AUC 기준.
- **H006 verdict F-5**: §18 인프라 룰 (batch=256 default, PYTORCH_CUDA_ALLOC_CONF, universal handler, 진단 로그) inherit.

## Carry-forward rules referenced

- §10.5 LayerNorm on x0 (Pre-LN).
- §10.7 카테고리 rotation (target_attention 첫 충족).
- §10.9 OneTrans softmax-attention entropy abort (cross-attention 도 softmax, instrumentation sub-H).
- §17.2 one-mutation.
- §17.3 binary success.
- §17.5 sample-scale.
- §17.6 cost cap.
- §17.7 falsification-first.
- §17.8 cloud handoff discipline.
- §18 inference infrastructure (NEW 2026-04-28).

## External inspirations (§10.4 P1+ 의무 주입)

- **Switch Transformer load balance loss** (Fedus et al. JMLR 2022) — 미적용. H008+ carry-forward.
- **TIGER** (Rajput et al. NeurIPS 2023 semantic ID) — P3 phase 후보.

본 H 는 target_attention 카테고리, paper grade 영감 충분. §10.4 외부 영감 의무 미충족 → H008 carry-forward.

## Future ablation candidates (post-H007)

- **candidate token 구성**: mean pool → first token / learnable weighted / item_id direct. sub-H of H007.
- **prepend vs append vs separate token stream**: position 영향 ablation.
- **Per-domain shared CandidateSummaryToken**: 4 modules → 1 shared.
- **CAN co-action style**: candidate × history co-action matrix. 별도 H.
- **HSTU multi-layer hierarchical**: candidate-aware multi-layer pruning. 별도 H.
- **Multi-head 수 tuning**: 4 → 8.
