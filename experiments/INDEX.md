# Experiments Registry — TAAC 2026 UNI-REC

> CLAUDE.md §4.6 + §9 + §17 + §18.

## Schema

| EXP_ID | hypothesis_id | seed | val_AUC | OOF_AUC | platform_AUC | compute_tier | wall (학습) | wall (infer) | status |
|---|---|---|---|---|---|---|---|---|---|
| **original_baseline** | (anchor smoke) | 42 | ~0.83 | TBD | **~0.83X** | T2.4 smoke | ~3 min | 180 sec | **active anchor (smoke)** |
| E_H006 | H006_longer_encoder_d_domain | 42 | ~0.82 | 0.8562 | **0.82** | T2.4 extended (10ep × 30%) | 4시간 10분 | 180 sec | refuted |
| E_H007 | H007_candidate_aware_xattn | 42 | 0.8321 (epoch 3 peak) | TBD | **0.8352** | T2.4 extended (3ep early stop × 30%) | 3시간 7분 | ~3 min | PASS marginal |
| E_H008 | H008_dcn_v2_block_fusion | 42 | TBD | **0.8585** | **0.8387** | T2.4 extended (10ep × 30%) | 3시간 41분 | 220 sec | **PASS — 지금까지 최고** |
| E_H009 | H009_combined_xattn_dcn_v2 | 42 | TBD | **0.8595** | **0.8364** | T2.4 extended (10ep × 30%, patience=3 미발동) | 3시간 36분 | 259 sec | **REFUTED — interference** |
| E_H010 | H010_ns_to_s_xattn | 42 | TBD | **0.8596** | **0.8408** | T2.4 extended (10ep × 30%, patience=3) | 3시간 44분 54초 | 297.02초 | **PASS — additive vs H008, 새 champion** |
| E_H011 | H011_aligned_pair_encoding | 42 | **0.8336** (best ep4/7) | 0.8589 (legacy) | **0.8347** (uncorrected eval) | T2.4 extended (10ep × 30%, patience=3) | 2시간 46분 54초 | 178.31초 | **REFUTED — degraded vs H010** (val_platform_gap −0.0011, overfit +0.0013) |
| E_H012 | H012_multi_domain_fusion | 42 | **0.8331** (best ep4/7) | 0.8589 (legacy) | **0.8380** (uncorrected eval) | T2.4 extended (10ep × 30%, patience=3, batch=2048) | 2시간 49분 43초 | 137.39초 | **REFUTED — Frame B (uniform routing)** (val_platform_gap −0.0049, overfit +0.0014) |
| E_H013 | H013_hyperparameter_calibration | 42 | **0.8316** (best ep6/9) | 0.8573 (legacy) | **0.8344** (uncorrected eval) | T2.4 extended (lr 8e-4, batch 2048, num_workers 4, buffer 8) | 4시간 8분 13초 | 95.57초 | **REFUTED — Frame A (Keskar generalization gap)** (val_platform_gap −0.0028, overfit +0.0016) |
| E_H014 | H014_long_seq_envelope | 42 | TBD | 0.8595 | **0.833587** (corrected eval) | T2.4 extended (10ep × 30%, batch=1024, seq=192 uniform, OOM iter-4) | 3시간 30분 49초 | 229.87초 | **REFUTED — L4 retire, paradigm shift inevitable** |
| **(re-run)** | H010_ns_to_s_xattn (corrected eval) | 42 | TBD | TBD | **0.837806** (new anchor) | (re-inference only) | — | 92.74초 | **new corrected anchor** |
| E_H015 | H015_recency_loss_weighting | 42 | 0.8358 (best ep5/8) | 0.8592 (legacy) | **0.83805** | T2.4 extended (10ep × 30%, batch 2048, lr 1e-4, recency [0.5, 1.5]) | (8 ep completed) | TBD | **REFUTED — Δ vs corrected anchor +0.00024 (§17.3 < +0.5pt cut), marginal positive direction** |
| E_H016 | H016_oof_future_redefine | 42 | 0.8399 (best ep4/7) | 0.8283 (redefined) | **0.83192** | T2.4 extended (oof_split_type future_label_time, train ~85%) | (7 ep completed) | TBD | **REFUTED model lift (Δ −0.0059), PASS infra (OOF redefined gap −0.004)** |
| E_H017 | H017_recency_exp_decay | 42 | TBD | TBD | **submission lost** | T2.4 extended (H015 + recency_weight_form exp) | TBD | TBD | **INVALID — submission lost, no platform AUC** |
| **~~Tier 0 (cancelled)~~** | ~~H011/H012/H013 corrected re-inference~~ | — | — | — | — | — | — | — | **CANCELLED 2026-05-03** — over-engineered. User correctly noted verdict already REFUTED for all 3 H, re-inference doesn't change verdict. **Replaced by retroactive log paste analysis** (see each H's training_result.md §18.8-style SUMMARY). 3 `training_request_remeasure.md` docs DELETED. corrected platform AUC remains uncorrected — −0.003pt heuristic shift estimate documented per H. |
| E_H018 | H018_per_user_recency_weighting | 42 | TBD | TBD | TBD | T2.4 extended (per-user exp tau=14, clip [0.1, 3.0], oof_redefine future_only) | TBD (~3-4h) | TBD | **SCAFFOLDED 2026-05-03** — patch spec (`upload_patch.md`) + card.yaml + 6 hypothesis docs complete. upload/ package NOT YET BUILT (out of Ralph PRD scope). config_sha256=TBD. control_exp_id=H015. updated outcome dist (noise ~50%). |
| E_H019 | H019_twin_long_seq_retrieval | 42 | **0.83720** (best ep4/7) | **0.8611** | **0.839674** | T2.4 ~3.5-4h ~$5-7 (TWIN GSU top-K=64 + ESU MHA per-domain, residual ADD post-backbone, gate sigmoid(-2)=0.12, seq cap 256, batch=1024) | (7 ep, early stop ep4) | TBD | **PASS measurable 2026-05-05 (Val/OOF backfilled 2026-05-07)** — Δ vs H010 corrected (0.837806) Platform = +0.001868pt. retrieval_long_seq class 첫 valid measurement. F-G ceiling Val 0.832~0.836 부분 부정 (Platform 만 0.838~0.840 확장). control_exp_id for H020/H021/H033/H034/H035/H041. |
| E_H041 | H041_cold_start_branch | 42 | **0.83496** (best ep2/5) | **0.8601** | not submitted | T2.4 cancelled — REFUTED via Val/OOF (cold_clsfier + per-sample cold_gate, init bias=2.0 → sigmoid≈0.88) | (5 ep, early stop ep2) | TBD | **REFUTED 2026-05-07** — Val Δ vs H019 (0.83720) = −0.00224pt, OOF Δ vs H019 (0.8611) = −0.00100pt → 둘 다 §17.3 < +0.001pt. R2 reframe (target_in_history=0.4% → familiar/novel split) wrong — 데이터에 split boundary 부재 (100% extrapolation regime). cloud submit STOP, R2 axis retire. F-H 첫 사례 (data-signal mapping 결함). |
| E_H038 | H038_aux_timestamp | 42 | **0.83735** (best ep6/9) | **0.8623** | TBD | T2.4 ~3.5h ~$5-7 (BCE main + λ=0.1 MSE aux on standardized log1p(timestamp), action_num=2, H019 base byte-identical) | (9 ep, no early stop) | TBD | **MEASURABLE 2026-05-07** — Val Δ vs H019 +0.00015pt (noise) / OOF Δ vs H019 +0.00120pt (measurable) / OOF LogLoss −0.0013 (calibration 개선). 14 H 모두 sample-level loss 였던 axis 의 first-touch — output_distribution_supervision class. **Late convergence ep6 vs H019 ep4** = aux signal 학습 dynamics 안정화. control=H019. Platform AUC 회수 보류 (Val noise 라 platform 우선순위 H042 다음). |
| E_H042 | H042_kld_prior_matching | 42 | TBD | TBD | TBD | T2.4 ~3.5h ~$5-7 (H038 base + per-sample Bernoulli KL(σ(logit)‖Bern(0.124)) λ_kld=0.01, kld_lambda=0 시 H038 byte-identical) | TBD | TBD | **BUILT 2026-05-07** — upload.tar.gz 66KB. T0 sanity ALL PASS (AST + 4 KLD math test + md5 verify 6 unchanged files). trainer.py loss term 만 추가 (~10 lines), trainable params +0. control=H038, secondary_control=H019. axis 누적 vs H019 ≥ +0.003pt OOF → output-distribution lever 강한 confirm. cloud-ready (`bash run.sh --seed 42`). **H038 platform 0.839071 회수 후 OOF→Platform transfer 실패 confirm (F-A 패턴 재발) — H042 platform 도 risk 가능성**. |
| E_H043 | H043_item_side_dcnv2 | 42 | TBD | TBD | TBD | T2.4 ~3.5h ~$5-7 (H019 base + DCN-V2 cross on item_ns: layers=2 rank=4, +1.28K params) | TBD | TBD | **BUILT 2026-05-07** — upload.tar.gz 110KB. T0 sanity ALL PASS (AST + 8 sanity test + md5 verify 6 unchanged files). model.py only mutation, trainer.py / dataset.py / infer.py byte-identical to H019. control=H019. **item-side modeling axis 14 H 첫 시도** — H042 (output-dist) 와 직교 hedge. user-side DCN-V2 (H008 PASS +0.0035pt) 의 item-side first 적용. data signal §3.5 4 도메인 disjoint vocab + target_in_history=0.4% 직접 motivation. cloud-ready (`bash run.sh --seed 42`). |
| E_H044 | H044_dann_cohort_debias | 42 | TBD | TBD | TBD | T2.4 ~3.5h ~$5-7 (H038 base + GradReverse on aux timestamp path, dann_lambda=0.5, +0 params) | TBD | TBD | **BUILT 2026-05-07** — upload.tar.gz 68KB. T0 sanity ALL PASS (AST + shellcheck + 5 GRL math test + md5 verify 6 unchanged files). trainer.py mutation only, model.py / dataset.py / infer.py byte-identical to H038. control=H019, secondary=H038. **cohort_drift_attack axis 14 H 첫 시도** — F-A 패턴 (OOF over Platform 9 H 누적) + H019/H038/H039 platform 비교 후 cohort drift 가 transfer 실패 근본 원인 정량 confirm. Ganin & Lempitsky DANN GRL form. dann_lambda=0 시 H038 byte-identical (safe carrier). cloud-ready (`bash run.sh --seed 42`). |
| E_H045 | H045_cross_domain_bridge | 42 | TBD | TBD | TBD | T2.4 ~3.5h ~$5-7 (H019 base + CrossDomainBridge: 4-head MHA on (B, 4, D) per-domain pooled tokens, residual ADD pre-TWIN, gate sigmoid(-2)=0.12, +21K params) | TBD | TBD | **BUILT 2026-05-07** — upload.tar.gz size TBD. T0 sanity ALL PASS (AST + 9 sanity test + md5 verify 6 unchanged files). model.py mutation only, trainer.py / dataset.py / infer.py byte-identical to H019. control=H019. **cross_domain_modeling axis 14 H 첫 시도** — §3.5 4 도메인 jaccard 0.7~10% overlap signal 직접 attack. H042/H043/H044 와 4-axis hedge 완성. cloud-ready (`bash run.sh --seed 42`). |
| E_H046 | H046_proper_dann | 42 | TBD | TBD | TBD | T2.4 ~3.5h ~$5-7 (H019 base + proper DANN: separate cohort_head + GRL between backbone and cohort_head, dann_cohort_lambda=0.1, +65 params) | TBD | TBD | **BUILT 2026-05-08** — upload.tar.gz 67KB. T0 sanity ALL PASS (AST + shellcheck + GRL math + DANN structure verify + md5 5 unchanged files). model.py + trainer.py mutation, dataset.py / infer.py / utils.py byte-identical to H019. control=H019. **H044 design fix** — H044 GRL on raw_logits[:, 1] post-clsfier 가 clsfier-aux + backbone 둘 다 reverse → loss 발산. H046 = proper structure (separate cohort_head positive grad, backbone reversed). dann_cohort_lambda=0.1 (H044 0.5 의 1/5 보수적). cloud-ready. |
| E_H047 | H047_per_domain_aux | 42 | TBD | TBD | TBD | T2.4 ~3.5h ~$5-7 (H019 base + 4 per-domain auxiliary classifier heads, multi-task BCE × 0.25, +260 params) | TBD | TBD | **BUILT 2026-05-08** — upload.tar.gz 66KB. T0 sanity ALL PASS (AST + shellcheck + md5 5 unchanged files). model.py + trainer.py mutation, dataset.py / infer.py / utils.py byte-identical to H019. control=H019. **multi_task_per_domain axis NEW first-touch** — 14 H 동안 main classifier 1개 만, per-domain prediction multi-task 0회. §3.5 4 도메인 disjoint vocab + per-domain seq length imbalance signal 직접 attack. cloud-ready. |
| E_H022 | H022_h010_multi_seed_variance | **42/43/44** | TBD | TBD | TBD (3 values, mean ± stdev) | T2.4 × 3 (~3.5h parallel / ~10.5h serial), ~$15 (measurement H, NO mutation, H010 byte-identical) | TBD per seed | TBD per seed | **SCAFFOLDED 2026-05-03** — 6 hypothesis docs + card.yaml complete. No new upload package (H010 + minimal §18.7/§18.8 patch). control_exp_id=H010 anchor. seed=42 anomaly (val 0.8322, OOF=N/A) → H023 redo. |
| E_H023 | H023_variance_baseline_redo | **42/43/44** | 0.8334 (seed=42 best ep6) | 0.8596 (seed=42) | TBD | T2.4 × 3 (~3.5h parallel / ~10.5h serial), ~$15 (bce explicit + OOF debug + batch=1024 OOM safety) | (9 ep / seed=42) | TBD | **BUILT 2026-05-04** — upload.tar.gz 64KB. seed=42 result 1 회수 (best_val 0.8334, OOF 0.8596 — H023 fix 작동 confirm). seed=43, 44 pending. |
| E_H025 | H025_dcnv2_stronger | 42 | 0.8336 (best ep6) | 0.8602 | TBD | T2.4 ~3.5h ~$5 (DCN-V2 rank 16, layers 4 — interaction axis sub-H of H008 winner) | (9 ep) | TBD | **BUILT + RAN 2026-05-04** — upload.tar.gz 63KB. val 0.8336 (H011/H023/H025 모두 0.833~0.834 영역, val ceiling 추가 confirm). config_sha256=TBD. control=H010 corrected. |
| E_H026 | H026_onetrans_backbone | 42 | 0.8330 (best ep10) | 0.8576 | TBD | T2.4 ~3.5h ~$5 (--backbone onetrans, backbone class first valid measurement) | (10 ep, no early stop) | TBD | **BUILT + RAN 2026-05-04** — upload.tar.gz 63KB. val 0.8330 (no early stop trigger, 더 많은 epoch 가능성 검증 가치 있음). |
| E_H027 | H027_train_ratio_expand | 42 | TBD | TBD | TBD | T2.4 ~5-6h ~$5-7 (train_ratio 0.3 → 0.5, OOM at 2048 → fallback 1024) | TBD | TBD | **BUILT 2026-05-04 + retried after OOM** — upload.tar.gz 63KB. 사용자 batch=1024 fallback 으로 재 launch. control=H010 corrected. |
| E_H028 | H028_split_seed_variance | **42/43/44** | TBD | TBD | TBD | T2.4 × 3 ~3.5h parallel / ~10.5h serial, ~$15 (cohort saturation diagnostic, batch=1024) | TBD per seed | TBD per seed | **BUILT 2026-05-04** — upload.tar.gz 63KB. caller passes --split_seed via "$@". control=H023. |
| E_H029 | H029_original_default_regime | 42 | TBD | TBD | TBD | T2.4 ~5-7h ~$5-7 (Keskar trap diagnostic, batch=256, lr=1e-4 — H010 train.py default 처음 측정) | TBD | TBD | **BUILT 2026-05-04** — upload.tar.gz 63KB. control=H010 corrected anchor. |
| E_H030 | H030_loss_type_focal | 42 | TBD | TBD | TBD | T2.4 ~3.5h ~$5 (loss isolation diagnostic, focal α=0.25 γ=2.0 explicit, batch=1024) | TBD | TBD | **BUILT 2026-05-04** — upload.tar.gz 62KB. control=H023 (bce explicit). |
| E_H031 | H031_item13_explicit_head | 42 | TBD | TBD | TBD | T2.4 ~3.5h ~$5 (Item13UserCrossBlock 32-dim item_13 emb × 5 user_int 16-dim emb outer-product cross → residual ADD post-backbone, gate init sigmoid(-2)=0.12) | TBD | TBD | **BUILT + HOTFIX 2026-05-04** — upload.tar.gz 68KB. **Initial cloud launch (5ep×1024) SIGABRT** — cause: vs hardcode (demo=10) vs cloud full-data schema vs (>>10) → CUDA index OOB. **Hotfix**: train.py 가 schema 로드 후 동적 (offset, vs) 추출 → PCVRHyFormer 에 전달. T0 hotfix sanity PASS (vs=200 cloud-realistic). control=H023. EDA: `eda/out/item_int_signal_audit.json` (item_13 univariate GBDT AUC 0.6561). |
| E_H032 | H032_timestamp_input_features | 42 | TBD | TBD | **0.83328** | T2.4 (TimeFeaturesBlock 3 derived categorical hour=24/dow=7/recency=8 → 16-dim sum → LN → Linear → residual ADD post-backbone, gate sigmoid(-2)=0.12) | TBD | TBD | **REFUTED 2026-05-05** — eval 0.83328 vs H023 control 0.8334 = Δ −0.00012pt = **NOISE band** (§17.3 < +0.001pt). 12 H 모두 0.832~0.836 ceiling 안. F-1: input-axis temporal signal REFUTED (loss-axis H015~H018 와 합쳐 temporal axis 전반 retire). |

## Archive (anchor reset 2026-04-28, H001–H005)

이전 측정값들은 인프라 bug 들 영향 → invalid.

| EXP_ID | hypothesis_id | val_AUC (당시) | platform AUC | Status |
|---|---|---|---|---|
| E000 | H001 | 0.5088 (demo) / 0.7055 OOF | n/a | archived |
| E_baseline_organizer | H001 | 0.8251 | unknown | archived |
| E001 | H002 | 0.8248 | unknown | archived (refuted) |
| E002 | H004 | 0.8174 | 0.5 (heuristic fallback) | archived |
| E_focal_smoke | H005 | 0.8253 | unknown | archived (refuted) |

## Cumulative cost (CLAUDE.md §17.6)

- T0/T1: $0
- T2 (Taiji): H006 (4h) + H007 (3h) + H008 (3.7h) + H009 (3.6h) + H010 (3.75h) + H011 (2.78h) + H012 (2.83h) + H013 (4.14h) + H014 (3.51h) + H015 (~3.5h) + H016 (~3.5h) + H017 (~3.5h, submission lost) = **~42시간 누적 학습** + 11+회 inference + H010 corrected re-inference. ~~Tier 0 H011/H012/H013 corrected re-inference~~ **CANCELLED 2026-05-03** (over-engineered per user pushback — verdict already REFUTED for all 3, log paste analysis sufficient). Taiji 가격 미공개 — 사용자 확인 필요.
- **§17.6 cap audit** (2026-05-03): H018 추가 시 ~46h. paradigm shift candidates (TWIN/OneTrans, T3 ~$20+/job) cost cap 위협 — H018 결과 후 재평가 mandatory.
- T3: $0
- **§17.6 budget cap 도달**: 32시간 누적 → cap 한계. H015 진행 시 cap 위협.
- **eval data correction (2026-05-02~03)**: organizer 가 eval data 수정. H010 재측정 = **0.837806** (이전 잘못된 eval 0.8408 −0.003 shift). prior H011/H012/H013 corrected 재측정 안 됨 → **ranking 비교 정밀도 낮음**. corrected ranking 명확히 위해 prior H 들 re-inference 권장.
- **H014 결과: L4 (truncate 정보 손실) REFUTED** (Δ vs H010 corrected −0.0042pt). dense long-seq expansion 가 ceiling 풀지 못함. OOF-Platform gap 2.59pt = 9 H 중 가장 큼 (cohort drift 강한 신호).
- **4-layer ceiling diagnosis 종료**: L1 + L3 + L4 모두 retire. **L2 (cohort drift) 만 남음**. paradigm shift inevitable.

## Conventions

- `EXP_ID = E_HXXX`.
- Platform AUC 측정 의무: §18 룰 통과 (batch heartbeat + `[infer] OK: torch path produced N predictions`).
- §18 인프라 (batch=256 default + PYTORCH_CUDA_ALLOC_CONF + universal handler + 진단 로그) 모든 H 패키지 적용 — H006/H007/H008 검증 완료.

## Anchor pair / 비교 룰

- **paired Δ 는 platform AUC 으로만** (H006 F-3 carry-forward).
- val ↔ platform 정합 확인됨 (H006/H007 패턴).
- OOF AUC 는 supplementary — H008 OOF-platform 갭 2pt (H006 의 3.5pt 보다 좁아짐). H009 에서 2.31pt 역행. **H010 에서 1.88pt 다시 narrow** (통합 위치 회피 설계 효과).
- **H010 PASS additive → anchor 갱신**: H011+ 부터 H010 (Platform 0.8408) 이 새 anchor. H008 (0.8387) carry-forward control 로 보존 (paired Δ 비교 용).
- **H010 = 새 champion** (Δ vs H008 +0.0021pt). H009 의 interference 와 정반대 — 통합 위치 (NS-only enrichment, anchor 입력 byte-identical) 가 안전 stacking 패턴 confirmed.
