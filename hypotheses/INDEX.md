# Hypotheses Registry — TAAC 2026 UNI-REC

> CLAUDE.md §9 + §10 + §17 + §18.

## Active Phase

`P0` → `P1 진입 검증 중`. **anchor 갱신: H010 (Platform 0.8408)** = 새 champion.
- **H010 PASS — additive** (Platform **0.8408**, target_attention paper-grade, OneTrans NS→S 직접 구현). Δ vs H008 +0.0021pt (additive 분류). attn entropy [0.81, 0.81] = threshold 의 14% (highly selective routing).
- 통합 위치 회피 설계 (NS-only enrichment, anchor 입력 byte-identical) → H009 interference 패턴 회피 검증됨.
- OOF-platform 갭 narrowing 회복 (H009 2.31pt → H010 **1.88pt**).
- **anchor 갱신**: H011+ control = H010 (0.8408). H008 (0.8387) carry-forward.
- → **H011 = orthogonal axis** (target_attention 직전 2회 연속 → §10.7 rotation 차단). 후보: aligned `<id, weight>` pair encoding (CLAUDE.md §3 / §4.8 mandate, interference 위험 0) 또는 multi_domain_fusion (MMoE/PLE).

## Active Anchor

| ID | Slug | What | Status |
|---|---|---|---|
| **H010_ns_to_s_xattn** | ns→s bidirectional xattn on H008 (DCN-V2 fusion) | PCVRHyFormer + transformer encoder + RankMixer fusion + DCN-V2 cross + NSToSCrossAttention layer + extended envelope (10ep × 30%, patience=3) | **active anchor** — Platform AUC **0.8408** (champion) |
| original_baseline | organizer-pure + leak-fix smoke | smoke envelope baseline | carry-forward (smoke ~0.83X, anchor 정확값 미확정) |
| H008 | dcn_v2_block_fusion | extended envelope, DCN-V2 cross fusion | carry-forward control (Platform 0.8387, paired Δ 비교 용) |

## Active Pipeline

| Order | ID | Slug | Mutation | primary_category | Compute | Status |
|---|---|---|---|---|---|---|
| 1 | H011 | aligned_pair_encoding | aligned `<id, weight>` pair (verified shared fids `{62, 63, 64, 65, 66, 89, 90, 91}`, 8 fids 출처 `competition/ns_groups.json`) 의 input embedding lookup 단계에서 element-wise multiply (DLRM/FwFM/DIN family minimum viable form). CLAUDE.md §3 / §4.8 mandate 직접 구현. parameter-free (params 추가 0). 통합 위치 = RankMixerNSTokenizer 입력, anchor 의 NS xattn / DCN-V2 fusion 출력 byte-identical → H010 F-1 안전 stacking 패턴. P0 audit gate = `n_k == dim_k` 검증. | feature_engineering (신규 카테고리 first-touch — §10.7 FREE) | T2.4 extended (10ep × 30%, patience=3) ~3-3.5h | **scaffold** (6 files + papers/feature_engineering/ 3 entries, fid set 정정 후) |

## Recent Findings (carry-forward)

- **H010 F-1**: NS-token bidirectional xattn 가 H008 anchor 위 stacking 으로 +0.0021pt 추가 lift (additive). H009 (interference, −0.0023pt) 와 정반대 — 차이는 **통합 위치** (H010 = NS-only enrichment, anchor 입력 byte-identical / H009 = candidate prepend, S 입력 변경). carry-forward: stacking H 는 "어느 텐서가 변경되는가" 분석 의무. NS-only enrichment = safe stacking pattern.
- **H010 F-2**: paper-grade source (OneTrans NS→S half, arXiv 2510.26104) 가 sample-scale extended (30% × 10ep) 에서도 측정 가능 lift. paper 100M user 규모 미도달 환경에서 minimum viable form 작동.
- **H010 F-3**: attn entropy [0.81, 0.81] = threshold 5.65 의 14% — **highly selective routing** (predicted "sparse" 보다 4× 더 sparse). NS-tokens 이 384 S tokens 중 e^0.81 ≈ 2 tokens 만 attend. degenerate uniform 의 정반대 극단. carry-forward: §10.9 threshold (현재 0.95×log(N) upper) 는 보수적 — lower bound 검토 가치. full-data 도착 시 entropy 변화 측정 의무.
- **H010 F-4**: **champion 갱신 + anchor 갱신**. Platform 0.8408 = 새 최고. H011+ control = H010. H008 carry-forward control (paired Δ 비교 용).
- **H010 F-5**: OOF-platform 갭 narrowing 회복 (H006 3.5 → H007 ~2.5 → H008 1.98 → H009 **2.31 역행** → H010 **1.88**). H009 capacity 증폭 cohort effect 가 H010 위치 회피 설계로 회복 → 통합 위치 가 cohort fit 에도 영향.
- **H010 F-6**: cost — H006~H010 누적 ~18시간. §17.6 cap 압박 지속. H011 부터 fp16/batch=512 또는 patience early signal 활용 권장 (wall 절반 시도).
- **H009 F-1**: 두 mechanism stacking 이 strongest single 보다 못함 (combined 0.8364 < H008 0.8387). OOF +0.001 / platform −0.0023 = **classic overfit signature**. capacity 증폭이 platform 일반화 악화시키는 패턴 직접 확인.
- **H009 F-2**: interference 위치 가설 — candidate token prepend → seq encoder 출력 candidate-mixed → DCN-V2 cross 입력 변경 → 두 mechanism 이 서로의 input 을 변경. block-level fusion 단점.
- **H009 F-3**: anchor 정확값 의존성이 결론 분류 흔듦 → **H010 anchor recalibration 우선순위 1순위 confirmed**.
- **H009 F-4**: H008 (Platform 0.8387) 여전히 최고. H011+ 부터 H008 anchor 위 single mutation.
- **H009 F-5**: §17.6 cost cap 압박 — H006~H009 누적 ~14시간 + H010 ~3시간 = ~17시간.
- **H008 F-1**: DCN-V2 fusion swap (interaction axis) PASS, +0.0035pt vs H007. **block-level fusion swap 이 우리 데이터에 작동 confirmed**.
- **H008 F-2**: §0 P1 룰 ("같은 block gradient 공유") active 검증 — concat-late anti-pattern 회피 정당화. 단 H009 F-1 으로 한계 노출.
- **H008 F-4**: patience=3 + aggressive early stop 으로 cost 절약 가능 (단 H009 에서 patience=3 trigger 안 됨 — capacity 증폭이 plateau 늦춤).
- **H008 F-5**: OOF-platform cohort 갭 좁아지는 패턴 (H006 3.5pt → H008 2pt) — H009 에서 다시 2.31pt 로 벌어짐.
- **H007 F-1**: candidate-as-attention-query mechanism (sequence axis) PASS marginal. H001–H006 noise floor 안 묻힘 패턴 깸 — first measurable lift.
- **H007 F-3**: anchor extended 측정 미래 H 명시 — H010 직접 충족.

## Archive

| ID | Slug | Status | Platform AUC | Verdict 핵심 |
|---|---|---|---|---|
| H001 | unified_baseline_full | invalid | unknown | 인프라 bug 영향 |
| H002 | interformer_domain_bridges | refuted | unknown | sub-block bridge marginal |
| H004 | onetrans_anchor | soft-warning + invalid | 0.5 (heuristic fallback) | OneTrans backbone 작동 (P3 PASS), full-data 시 재평가 |
| H005 | focal_loss_calibration | refuted | unknown | 12% imbalance 영역 효과 없음 |
| H006 | longer_encoder_d_domain | refuted | 0.82 vs anchor 0.83 | top-K=50 < L=128 단순 truncation, non-targeted |
| H007 | candidate_aware_xattn | **PASS marginal** | **0.8352** | sequence axis mechanism class 작동 first confirmed |
| H008 | dcn_v2_block_fusion | **PASS** | **0.8387** | interaction axis mechanism class 작동, **지금까지 최고** |
| H009 | combined_xattn_dcn_v2 | **REFUTED — interference** | **0.8364** | combined < strongest single (H008). OOF+/platform− overfit signature. block-level fusion 위치 충돌 가설. |
| H010 | ns_to_s_xattn | **PASS — additive** | **0.8408** | OneTrans NS→S bidirectional 직접 구현, H008 anchor 위 NS-only enrichment stacking. Δ vs H008 +0.0021pt = additive. attn entropy 0.81 (highly selective routing). **새 champion + anchor 갱신**. |

자세한 carry-forward = 메모리 `project_failed_h_archive.md` + 각 verdict.md.

## Backlog (정렬 — H010 PASS additive 결과 반영, 2026-04-30)

**H011 top candidates (orthogonal axis, target_attention rotation 차단)**:
- **Aligned `<id, weight>` pair encoding** — CLAUDE.md §3 / §4.8 mandate. verified shared fids `{62, 63, 64, 65, 66, 89, 90, 91}` (8). orthogonal axis (input-stage encoding), interference 위험 0. **H011 진행 중** (scaffold 정정 완료).
- **multi_domain_fusion** (MMoE/PLE) — block fusion 강화, 4 도메인 expert routing. orthogonal axis, H011 후보.

**Deferred (rotation/cost/완성도 사유)**:
- **NS xattn sub-H** — num_heads 증가 (4→8), multi-layer, attention map snapshot. **target_attention 재진입** → H011 직후 차단. H012+ 후보.
- **mixed causal mask injection** — OneTrans 의 4 sub-mask 중 NS→S 외 (S→S causal, NS→NS, candidate→{S,NS}). target_attention family → H012+ 후보.
- **anchor recalibration extended** — original_baseline extended retrain. measurement H, mechanism lift 0. H011+ 부터 H010/H008 paired 비교 가 주가 되면 가치 작음. 폴더 `hypotheses/anchor_recalibration_extended_backlog/` 보존.
- **DCN-V2 tuning** — layer 수 (2→4), rank (8→16) sub-H of H008.
- **CAN co-action** — sparse_feature_cross 변형, candidate × history element-wise.
- **target_attention 변형** — multi-layer, CAN-style. H007 sub-H.
- **switch_load_balance_inject** — external_inspirations. §10.4 P1+ 의무.
- **onetrans_anchor_full_data_revisit** — H004 재평가, full-data 도착 시. extended envelope 측정 미실행.
- **lgbm_tabular_control** — 측정 도구.
- **interference recovery sub-H** — H009 의 candidate token 통합 위치 변경 (prepend → fusion 이후).
- **Speed optimizations** — fp16 autocast, batch=512, num_workers=4. 적용 시 wall ~50% 단축. §17.6 cap 압박 (~18h 누적) 으로 H011+ 채택 권장.

## TEMPLATE

새 H 생성: `cp -r hypotheses/TEMPLATE hypotheses/HXXX_slug` 후 본 파일 Active Pipeline 갱신.
