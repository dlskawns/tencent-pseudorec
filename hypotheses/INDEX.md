# Hypotheses Registry — TAAC 2026 UNI-REC

> CLAUDE.md §9 + §10 + §17 + §18.

## Active Phase

`P0` → `P1 진입 검증 중`. **anchor 갱신 — H010 corrected eval = 0.837806** (organizer eval data 수정 2026-05-02~03). 5회 연속 H REFUTED (H011/H012/H013/H014).

- **H014 REFUTED — L4 retire** (Platform **0.833587**, long-seq envelope expansion seq 192 uniform). Δ vs H010 corrected −0.0042pt. dense expansion 효과 없음. OOF-Platform gap 2.59pt = 9 H 중 가장 큼 (cohort drift 강화 신호).
- **eval data correction**: H010 prior 0.8408 → corrected 0.837806 (−0.003 shift). prior H011/H012/H013 corrected 재측정 안 됨 → **ranking 비교 정밀도 낮음**. 사용자 직관 "H012 가 가장 높다" = invalid comparison (H010 corrected vs H012 prior).
- **4-layer ceiling diagnosis 종료**.

### 4-layer ceiling diagnosis (H014 후 종료)

| Layer | 가설 | 상태 |
|---|---|---|
| L1 | hyperparameter regime | ❌ retire (H013 REFUTED) |
| L2 | OOF-platform gap 1.9~2.6pt = cohort drift hard ceiling | **남은 마지막 가설** |
| L3 | NS xattn selective routing → NS-token level mechanism 추가 무용 | ❌ retire (H011/H012/H013 누적) |
| L4 | truncate 64-128 = §3.5 p90 1393~2215 의 95%+ 정보 손실 | ❌ retire (H014 REFUTED — dense expansion 효과 없음, retrieval form 별도) |

→ **paradigm shift inevitable**. H015 후보 = (a) cohort drift 처리 (L2 직접 검증), (b) retrieval-based long-seq (TWIN/SIM/HSTU — L4 의 retrieval form), (c) backbone replacement. **(d) 사전 작업: H011/H012/H013 corrected eval 재측정** mandatory (paired Δ 정합성).

## Active Anchor

| ID | Slug | What | Status |
|---|---|---|---|
| **H010_ns_to_s_xattn** | ns→s bidirectional xattn on H008 (DCN-V2 fusion) | PCVRHyFormer + transformer encoder + RankMixer fusion + DCN-V2 cross + NSToSCrossAttention layer + extended envelope (10ep × 30%, patience=3) | **active anchor** — Platform AUC **0.8408** (champion) |
| original_baseline | organizer-pure + leak-fix smoke | smoke envelope baseline | carry-forward (smoke ~0.83X, anchor 정확값 미확정) |
| H008 | dcn_v2_block_fusion | extended envelope, DCN-V2 cross fusion | carry-forward control (Platform 0.8387, paired Δ 비교 용) |

## Active Pipeline

| Order | ID | Slug | Mutation | primary_category | Compute | Status |
|---|---|---|---|---|---|---|
| 1 | **~~Tier 0 (cancelled)~~** | ~~H011/H012/H013 corrected re-inference~~ | **CANCELLED 2026-05-03** — over-engineered. User correctly noted verdict already REFUTED for all 3 H, re-inference doesn't change verdict. **Replaced by retroactive log paste analysis** (see each H's `training_result.md` §18.8-style SUMMARY). H011 best=0.8336/ep4 OOF=0.8589 / H012 best=0.8331/ep4 OOF=0.8589 / H013 best=0.8316/ep6 OOF=0.8573. corrected platform AUC remains uncorrected (−0.003pt heuristic shift documented per H). | infrastructure | (skipped) | **COMPLETED via log paste 2026-05-03** |
| 2 | H018 | per_user_recency_weighting | **Single mutation** — per-batch linear recency (H015) → per-user exp time-decay weighting. weight = exp(-days_since_last_event / tau=14), per-batch normalize mean=1.0, clip [0.1, 3.0]. Carry-forward: H016 redefined OOF (future_only) default. §18.8 SUMMARY block emit (FIRST H to use). control_exp_id=H015. | temporal_cohort (4th sibling, §17.4 re-entry justified in challengers.md) | T2.4 extended ~3-4h | **SCAFFOLDED 2026-05-03** — hypothesis docs (6 files) + card.yaml + upload_patch.md complete. upload/ package NOT YET BUILT. **Updated outcome distribution (post F-A~F-D): noise probability ~50%** → H019 사전 scaffold 정당. |
| 3 | H019 | twin_long_seq_retrieval | **Paradigm shift first entry** — TWIN (Tencent 2024) GSU+ESU per-domain retrieval. GSU = inner product (parameter-free) → top-K=64, ESU = MultiHeadAttention. seq_max_lens 64-128 → 512 cap. control_exp_id=H010 corrected anchor. §18.8 SUMMARY emit (second H). | retrieval_long_seq (NEW first-touch — §17.4 auto-justified) | **T3** ~6h, ~$15 | **SCAFFOLDED 2026-05-03** — hypothesis docs (6 files) + card.yaml complete. upload/ + upload_patch.md NOT YET BUILT. **CONDITIONAL launch**: depends on H018 outcome (H018 noise → H019 immediate; H018 strong → H019 보류). |

## Recent Findings (carry-forward)

- **F-A (cross-H val→platform pattern, 4 H)**: best_val_AUC consistently UNDER platform AUC for legacy OOF — H011 −0.0011 / H012 −0.0049 / H013 −0.0028 / H015 −0.0023 (mean −0.003pt, all negative). H016 (redefined OOF) flips to +0.0080 (overestimates). carry-forward: **best_val 가 platform 의 conservative estimate (~−0.003pt)**, ranking decision 시 +0.005pt 보정 권장.
- **F-B (OOF saturation 9 H confirm)**: H006~H015 의 legacy OOF = 0.8562~0.8596 (0.34pt 폭) vs platform 0.8336~0.8408 (0.72pt 폭). OOF 가 platform 변동의 1/2 만 표현. **legacy OOF 영구 retire as ranking signal.** redefined OOF (H016 framework) 만 H018+ 에서 사용.
- **F-C (best_epoch convergence)**: H011=4/7, H012=4/7, H013=6/9, H015=5/8 — 모두 early-mid (50~70%). overfit_gap 0.0009~0.0016pt (best−last) 거의 없음. recency loss / lr / mechanism 변경 무관 같은 패턴. carry-forward: H018+ 에서 epoch budget 8~10ep 적정, early-stopping patience=3 sufficient.
- **F-D (H012 ≈ H015 platform tied — 사용자 직관 검증)**: H012 platform 0.8380 (uncorrected) vs H015 platform 0.83805 (corrected). 직접 비교 invalid (다른 eval data). −0.003pt heuristic shift 적용 시 **H012 corrected estimate 0.8350 < H015 corrected 0.83805** = Δ −0.003pt. **사용자 직관 "H012 가 가장 높다" REFUTED in corrected comparison**. 단 H012 mechanism (MoE multi-domain) 이 H015 (recency loss) 와 비슷한 영역의 ceiling — 두 mechanism class 모두 cohort drift hard ceiling 부딪힘.
- **F-E (train_loss scale incomparable)**: H011/H012/H013 train_loss ≈ 0.12 (focal loss 또는 different reduction) / H015 train_loss ≈ 0.32 (standard BCE). 동일 H 내 epoch trajectory 만 비교 valid, cross-H train_loss 비교 INVALID. carry-forward: §18.8 SUMMARY 의 epoch_history 에서 train_loss 컬럼은 H 내부 monotonic check 만 (cross-H ranking 신호 아님).
- **F-F (Tier 0 over-engineering 회수, 2026-05-03)**: 직전 PRD 의 Tier 0 cloud re-inference 제안이 over-engineering. 사용자 정확히 지적 — 이미 verdict REFUTED 인 H 들 re-inference 로 verdict 안 바뀜. log paste 분석 으로 충분 (이 PRD US-002). carry-forward: 이미 REFUTED H 의 corrected 재측정 우선순위 낮음, 새 H 의 launch 비용 안 드는 retroactive 분석이 더 ROI 높음.
- **H015 F-1 (REFUTED — marginal)**: per-batch linear recency [0.5, 1.5] platform AUC 0.83805 vs corrected anchor 0.837806 = Δ **+0.00024pt**. **§17.3 binary < +0.5pt cut → REFUTED**. 부호 positive (mechanism direction 작동) + magnitude 매우 작음. carry-forward: per-batch coarse 가 약점일 가능성 → H018 per-user fine granularity 직접 attack.
- **H015 F-2**: best_epoch=5 / 8 epochs total. overfit gap = best_val 0.8358 − last_val 0.8349 = +0.0009pt (거의 없음). recency 가 학습 destabilize 안 함.
- **H015 F-3**: val ↔ platform gap = 0.8358 − 0.83805 = +0.0023pt. **best_val 이 platform 과 가장 가까운 indicator** 처음 검증 (5 H 비교). legacy OOF (0.8592) saturated → ranking signal 무용 carry-forward 강한 confirm.
- **H016 F-1 (REFUTED model / PASS infra)**: OOF future-only redefine platform AUC 0.83192 vs corrected anchor 0.837806 = Δ **−0.00589pt** → **REFUTED for model lift**. 단 redefined OOF 0.8283 vs platform 0.83192 = gap **−0.0036pt ≈ 0** → **PASS for measurement infra** (redefined OOF 가 platform 분포에 align). carry-forward: H018+ 모든 H 의 oof_redefine future_only default.
- **H016 F-2**: legacy OOF saturation 종료 신호 — 5 H 동안 0.858~0.860 saturated 였던 OOF 가 redefine 후 0.828 (drop) ↔ platform 0.832 (drop) → **redefined OOF Δ 가 platform Δ 와 ranking 일치 첫 evidence**. H018+ ranking signal 회복 가능성.
- **H017 F-1 (INVALID)**: submission lost. 결과 회수 실패 → no platform AUC. Triple-H setup 의 form variant 측정 불가. **carry-forward to H018**: exp form 은 H018 의 sub-spec (per-user × exp 동시) 으로 재진입 — single-mutation 정당성은 challengers.md ④ 에 명시 (한 mechanism class 안 finer specification).
- **eval data correction (organizer 2026-05-02~03)**: H010 prior 0.8408 → corrected 0.837806. H011/H012/H013 prior 측정 미재측정 → ranking 비교 invalid. **Tier 0 prerequisite (Ralph PRD US-001)**: H011/H012/H013 corrected re-measurement (inference-only ~5min × 3) MANDATORY before H018 verdict 정합성.
- **§18.7 룰 first-add (2026-05-03)**: H015 dataset.py `label_times = batch.column('label_time').to_numpy().astype(np.int64)` 가 inference data 의 null label_time 에서 ArrowInvalid (zero_copy_only=True default). H015/H017 둘 다 같은 base 카피 → 같은 버그. fix `fill_null(0).to_numpy(zero_copy_only=False)`. notes/refs/inference_lessons.md §18.7 신규 + dataset-inference-auditor 서브에이전트 신규 (`.claude/agents/dataset-inference-auditor.md`).
- **§18.8 룰 first-add (2026-05-03)**: Taiji platform 이 metrics.json 등 artifact 미노출 → stdout 만 회수 가능. train.py 끝에 단일 SUMMARY 블록 print 의무 (`==== TRAIN SUMMARY (HXXX) ====` ~ `==== END SUMMARY ====` 마커, ~15줄). verify-claim 스킬이 정규식 파싱. H018 = first-ever H to emit §18.8 SUMMARY.
- **H014 F-1**: L4 (truncate) retire — dense long-seq expansion (uniform 192) 가 H010 corrected anchor 위 −0.0042pt 악화. truncate 정보 손실이 ceiling 의 진짜 정체 아님. dense form 만 retire (retrieval form 별도 가설).
- **H014 F-2**: cohort drift 강한 confirm — OOF-Platform gap 2.59pt = 9 H 중 가장 큼. long-seq 가 OOF 보존 (0.8595 = H010 0.8596) / Platform 악화 → **Frame B (cohort hard ceiling) 가 ceiling 의 진짜 정체** 매우 강한 신호.
- **H014 F-3**: 4-layer ceiling diagnosis 종료 — L1 + L3 + L4 모두 retire. **L2 (cohort drift) 만 남음**. paradigm shift inevitable.
- **H014 F-4**: NS xattn routing 비율 더 sparse — entropy 0.99 (vs H010 0.81) 절대값 증가지만 e^entropy / total = 0.35% (vs H010 0.58%). dominant signal sparse capture pattern long-seq 에서도 유지. mechanism class 한계 강한 confirm.
- **H014 F-5**: eval data correction (organizer 2026-05-02~03) — H010 prior 0.8408 → corrected 0.837806. prior H011/H012/H013 corrected 미재측정 → ranking 비교 정밀도 낮음. H011/H012/H013 corrected 재측정 mandatory.
- **H013 F-1**: Frame A REFUTED — Linear scaling rule (lr 1e-4 → 8e-4 for batch 2048) 이 ceiling 풀지 못함. Δ vs H010 anchor −0.0064pt = 명확한 degraded. **Hyperparameter artifact 가설 무효, L1 retire confirmed**.
- **H013 F-2**: Keskar large-batch generalization gap 적용 케이스 — OOF AUC 0.857 = 8 H 중 처음 0.86 미달. lr 8e-4 + batch 2048 = 학습 자체가 less optimal.
- **H013 F-3**: 4-layer ceiling diagnosis 갱신 — L1 (hyperparameter) ❌ + L3 (NS xattn sparse) ❌ retire. **L2 (cohort drift) + L4 (truncate) 만 남음**.
- **H013 F-6**: §10.3 challenger rule trigger — 3회 연속 H010 anchor 위 mutation REFUTED (H011/H012/H013) → H014 envelope mutation (다른 axis) 필수.
- **H012 F-1**: Frame B confirmed — MoE gate entropy [1.378, 1.363] ≈ uniform threshold 1.386. expert specialization 학습 못 함. H010 NS xattn selective routing (entropy 0.81) 이 이미 dominant signal mixed 형태로 capture 중 → explicit NS-token level mechanism 추가 retire. carry-forward: H013+ NS-token level mutation 우선 안 함.
- **H012 F-2**: Hyperparameter measurement bias 노출 — 사용자 batch=2048 (default 256 의 8×) + lr default 1e-4 → effective lr 1/8 underpowered. 7개 H 모두 같은 regime → 상대 paired Δ valid 하지만 절대 lift 작은 게 hyperparameter artifact 가능성. carry-forward: **H013 = hyperparameter calibration (measurement H, parametric mutation justified)**.
- **H012 F-3**: 4-layer ceiling diagnosis — L1 hyperparameter / L2 cohort drift / L3 NS xattn sparse routing / L4 truncate 정보 손실. H013 calibration 결과로 L1 분리 → L2 (cohort) 또는 L4 (long-seq) 결정.
- **H012 F-4**: multi_domain_fusion category retire 후보. 재진입 시 강한 정당화 (PLE progressive 또는 hard top-K routing) 필요. backlog 후순위.
- **H012 F-5**: IO bound 신호 — H011/H012 wall 학습 −24~25%, infer −40~54% vs H010. params 추가 0~66K 만으로 설명 안 됨. GPU idle 가능성. H013 calibration 시 num_workers/buffer_batches 같이.
- **H011 F-1**: input-stage aligned encoding REFUTED — per-row L1-norm weighted multiply 가 H010 anchor 위 −0.0061pt 악화. OOF (−0.0007) 거의 보존, Platform (−0.0061) 큰 폭 하락 = **classic overfit signature 재발** (H009 와 같은 패턴). carry-forward: input-stage modify 가 cohort fit 유지하면서도 platform 악화 → mechanism class 자체가 wrong direction 가능성.
- **H011 F-2**: NS xattn routing 거의 변화 없음 (entropy Δ ~0.001). predictions.md "Frame B" 신호 — H010 의 selective routing (entropy 0.81 = 384 tokens 중 ~2 attend) 이 H011 input-stage modify 흡수. NS xattn 이 이미 dominant signal implicit 학습. carry-forward: NS xattn sub-H (multi-layer, num_heads) 도 marginal 가능성.
- **H011 F-3**: Option α (per-row L1 norm) 이 Pattern X (count, max=18M) 와 Pattern Y (signed [-1,+1]) 통일 처리. Pattern X 의 magnitude 정보 손실, Pattern Y 의 sign distortion 가능성. carry-forward: 만약 input-stage 재시도 시 multi-form sub-H (Pattern X = log1p, Pattern Y = raw 또는 abs).
- **H011 F-4**: training wall −25% / inference −40% (parameter-free input mutation 의 cost-effective 신호). carry-forward: 향후 input-stage parameter-free mutation 후보 cost 부담 작음.
- **H011 F-5**: cohort drift 가 platform 일반화의 hard ceiling 가능성 — H009 (block-level) / H011 (input stage) 둘 다 같은 OOF+/Platform− 패턴. mechanism 위치 무관하게 발현. carry-forward: cohort 처리 자체가 별도 H 가치 (예: temporal cohort weighting, recency-aware OOF 재정의).
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
| H011 | aligned_pair_encoding | **REFUTED — degraded** | **0.8347** | input-stage per-row L1-norm weighted multiply on aligned 8 fids. Δ vs H010 anchor −0.0061pt. OOF +/Platform − (cohort overfit signature 재발, H009 패턴). attn_entropy 변화 ~0 (Frame B 신호 — NS-level binding 충분). |
| H012 | multi_domain_fusion | **REFUTED — Frame B (uniform routing)** | **0.8380** | MoE single-layer 4 experts on NS-tokens. Δ vs H010 anchor −0.0028pt. **MoE gate entropy [1.378, 1.363] ≈ uniform threshold 1.386** → expert specialization 학습 못 함. H010 selective routing 이 이미 dominant signal capture 중 → explicit MoE redundant. **4-layer ceiling diagnosis 노출**: L1 hyperparameter / L2 cohort drift / L3 NS xattn sparse / L4 truncate. H013 = calibration. |
| H013 | hyperparameter_calibration | **REFUTED — Frame A REFUTED** | **0.8344** | Linear scaling lr 1e-4 → 8e-4 for batch 2048. Δ vs H010 anchor −0.0064pt = degraded. OOF AUC 0.857 = 8 H 중 처음 0.86 미달 (Keskar large-batch generalization gap). attn_entropy [0.8138, 0.8140] (변화 미세). **L1 retire confirmed**, 4-layer diagnosis 의 L2 + L4 만 남음. §10.3 challenger trigger → H014 envelope mutation. |
| H014 | long_seq_envelope | **REFUTED — L4 retire** | **0.8336** (corrected eval) | seq_max_lens 64-128 → 192 uniform (a/b 3×, c/d 1.5×). Δ vs H010 corrected 0.837806 = −0.0042pt. **OOF-Platform gap 2.59pt = 9 H 중 가장 큼** (cohort drift 증폭). attn_entropy [0.99, 0.99] (절대값 증가지만 비율 더 sparse). **4-layer diagnosis 종료**: L1+L3+L4 retire, L2 (cohort) 만 남음. paradigm shift inevitable. |

자세한 carry-forward = 메모리 `project_failed_h_archive.md` + 각 verdict.md.

## Backlog (정렬 — H014 REFUTED + 4-layer diagnosis 종료 + paradigm shift 결정 turn, 2026-05-03)

**Prerequisite (H015 진행 전 mandatory)**:
- **H011/H012/H013 corrected eval 재측정** — eval data correction 후 prior 측정값 비교 invalid. inference only (~5min × 3) → corrected ranking 명확. 사용자 직관 "H012 가 가장 높다" 검증.

**H015 top candidates (paradigm shift, 4-layer 종료 후)**:

| Rank | Option | Axis | Cost | UNI-REC alignment |
|---|---|---|---|---|
| **1** | **L2 cohort drift 처리** — recency-aware loss / temporal cohort embedding / OOF 재정의 (label_time future-only holdout) | training procedure (UNI-REC 안 새 layer) | medium | 사용자 cohort handling = UNI-REC 의 deployment realism |
| **2** | **Retrieval-based long-seq** — TWIN (Tencent) / SIM (Alibaba) / HSTU (Meta) | sequence axis (L4 의 retrieval form) | large | sequence axis 강화, P2 phase entry |
| **3** | **Backbone replacement** — OneTrans full single-stream / HSTU trunk / InterFormer 3-arch | backbone | very large (cost cap 위협) | UNI-REC paradigm 안 다른 transformer |

**Deferred (rotation/category retire 사유)**:
- **L1 hyperparameter sub-H** (lr 4e-4 / batch 256) — H013 REFUTED 후 가치 작음.
- **NS xattn sub-H** — L3 retire 일관 (H011/H012/H013/H014).
- **multi_domain_fusion 재진입** — H012 retire.
- **input-stage sub-form** — H011 retire.
- **mixed causal mask injection** — L3 retire 적용.
- **CAN co-action** — paradigm shift 결정 후 재평가.
- **dense long-seq sub-H** (seq 256/512 OOM 후 무용 confirm via H014 192 uniform) — retire.

**4-layer ceiling diagnosis 종료 의미**:
- H011/H012/H013/H014 (4 H) 모두 H010 anchor 위 mutation REFUTED.
- L2 (cohort drift) 가 마지막 mechanism axis 가설 → 직접 검증 또는 우회 (다른 backbone 으로 다른 axis 찾기).
- §17.6 cap 임박 (32h 누적) → H015 cost-effective 결정 critical.
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
