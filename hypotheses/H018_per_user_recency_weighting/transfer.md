# H018 — Method Transfer

## ① Source

- **Production CTR engineering** (Meta / Google / Tencent): per-user
  time-decay loss weighting 이 standard practice (논문 부재, production
  know-how). 본 H 의 1차 source.
- **Concept drift adaptation** (Gama et al. 2014, "A Survey on Concept
  Drift Adaptation"): online learning 의 per-user recency-based weight
  decay (exp form, tau hyperparameter).
- **H015 carry-forward** (per-batch linear): 본 H base. mechanism class
  (recency loss weighting) 검증 framework. Δ vs H015 paired 비교.
- **H016 carry-forward** (OOF future-only): 본 H 의 measurement infra.
  redefined OOF 만 신뢰 (legacy OOF saturated retire).
- **카테고리 family** (`temporal_cohort/`): H015/H016/H017 sibling 4번째
  entry. §17.4 재진입 정당화 in challengers.md.

## ② Original mechanism

**Per-user exponential time-decay loss weighting** (1단락 재서술):

각 user 의 event history 기준, sample 의 timestamp 와 user 의 most-recent
domain seq event timestamp 차이를 `days_since_last_event` 로 산출. weight
= `exp(-gap / tau)` (tau = decay constant, 본 H = 14 일). user-level
batch normalization (batch 안 weight mean=1.0 유지, loss scale 보존).

**핵심 차이 vs H015**:
- H015 (per-batch linear): batch 내 label_time min-max → percentile →
  linear weight [w_min, w_max]. batch composition 의존.
- H018 (per-user exp): user 의 own event history 기준 gap → exp decay.
  batch composition 무관, user-stable.

**우리 적용**:
- `dataset.py` `_convert_batch`: 각 row 의 max(domain_*_seq_<ts_fid>)
  계산 (4 도메인 ts_fid: a→39, b→67, c→27, d→26 per §3.1) → row 의
  current `timestamp` 와 차이 → `days_since_last_event` (음수 = 미래 event,
  → 0 으로 clip). batch dict 에 노출.
- `trainer.py._train_step`: weight = exp(-gap/tau) → per-batch normalize
  (mean=1.0) → bce/focal loss reduction='none' + weighted mean.

## ③ What we adopt

- **Mechanism class**: per-user exp time-decay loss weighting. minimum
  viable form (tau=14 fixed, no sweep).
- **변경 내용 (3 files + run.sh)**:
  1. `dataset.py`: `_convert_batch` 가 4 도메인 ts_fid 의 max 추출 →
     `days_since_last_event` (per row, max over domains) 계산 → batch
     dict 노출 (~10 줄).
  2. `trainer.py.__init__`: 3 args (`use_per_user_recency`,
     `recency_tau_days`, `recency_weight_clip` [0.1, 3.0]).
  3. `trainer.py._train_step`: weighting branch (~25 줄). exp decay +
     per-batch normalize + clip + weighted mean.
  4. `train.py`: argparse 3 + Trainer 3 keys.
  5. `run.sh`: 3 H018 flags + 2 H010 default 명시 bake. H015 flags 제거
     (per-batch linear off).
- **CLI**: `--use_per_user_recency --recency_tau_days 14
  --recency_weight_clip 0.1 3.0`.
- **Carry-forward (H016 infra)**: `--oof_redefine future_only` (default
  on, redefined OOF measurement framework).

## ④ What we modify (NOT a clone of H015 or production)

- **Per-user (not per-batch)**: 본 H 의 핵심 mutation.
- **Exponential (not linear)**: H017 form variant 이었음 (submission lost).
  H018 = per-user × exp 동시 적용. **§17.2 single-mutation 우려**: per-user
  granularity change 가 primary mutation. exp form 은 H017 carry-forward
  (이미 분리 검증 시도). 두 변경이 같은 mechanism class (recency loss
  weighting) 안 finer specification.
- **Per-batch normalize (mean=1.0)**: H015 의 mean=1.0 보존 carry-forward.
  loss scale 보존 → lr/optim 영향 0 → paired Δ confound 작음.
- **Weight clip [0.1, 3.0]**: tau=14 + gap > 60일 user 의 weight ≈ 0.014
  → underflow 방지. tau=14 + gap < 1일 user 의 weight ≈ 0.93 → upper clip
  3.0 도 거의 안 걸림. variance 안전망.
- **Tau=14 fixed**: production 표준 (1주 = active, 2주 = recent, 1개월
  = stale). sweep 은 sub-H.

## ⑤ UNI-REC alignment (HARD)

- **Sequential reference**: 변경 없음 (H010 NS xattn + per-domain encoder
  그대로).
- **Interaction reference**: 변경 없음 (H008 DCN-V2 fusion 그대로).
- **Bridging mechanism**: 변경 없음.
- **Training procedure**: H015 axis 강화 (per-batch coarse → per-user
  fine). production CTR cohort handling 의 finer form.
- **primary_category**: `temporal_cohort` (H015/H016/H017 sibling, 4번째
  entry. §17.4 재진입 정당화 in challengers.md).
- **Innovation axis**: per-user level cohort handling. 9 H 누적 OOF
  saturated / Platform 변동 패턴의 진짜 source (user-level recency) 를
  finer granularity 로 attack.

## ⑥ Sample-scale viability (Rule UB-1, §10.6)

- 추가 trainable params: **0** (loss 가중치만, model byte-identical).
- §10.6 cap: 위반 없음.
- Sample-scale risk:
  - **Per-user gap signal sparse on 1000 rows**: 1000 row sample 에서
    user count ~few hundred → per-user signal 약함. cloud full-data 에서만
    real measurement.
  - **Tau=14 fixed**: sample-scale 검증 없이 선택. cloud 결과 약하면
    H018-sub = tau sweep (7, 14, 30).
  - **Loss scale**: per-batch normalize (mean=1.0) → 영향 없음.

## ⑦ Carry-forward rules to honor

- **§10.5 LayerNorm on x₀**: 변경 없음.
- **§10.6 sample budget cap**: 변경 없음 (params 추가 0).
- **§10.7 카테고리 rotation**: temporal_cohort 4번째 entry. challengers.md
  §17.4 정당화 명시. **post-result mandatory rotation**.
- **§10.9 OneTrans softmax-attention entropy**: 변경 없음.
- **§10.10 InterFormer bridge gating σ(−2)**: 미적용.
- **§17.2 one-mutation**: per-user granularity change. exp form 은 H017
  carry-forward (한 mechanism class 의 sub-specification, multi-mutation
  아님 per challengers.md ④).
- **§17.3 binary success**: Δ vs H015 corrected ≥ +0.5pt → PASS strong.
  Δ ∈ [+0.1, +0.5pt] → PASS measurable but mechanism class retire 권고.
  Δ < +0.1pt → REFUTED (recency mechanism class 영구 retire).
- **§17.4 카테고리 rotation 재진입 정당화**: challengers.md 에 5 항목
  명시.
- **§17.5 sample-scale = code-path verification only**.
- **§17.6 cost cap**: extended ~3-4h (H010 envelope 동일). 누적 ~40h +
  4h = ~44h. cap 압박 큼. **H018 REFUTED 시 다음 H 의 cost budget audit
  강제**.
- **§18.6 dataset-inference-auditor**: H018 upload/ ready 직전 PASS
  의무.
- **§18.7 nullable to_numpy**: `timestamp` non-null + domain seq ts_fid
  non-null 확인. 새 컬럼 (gap) 은 derived 라 nullable 영향 없음.
- **§18.8 emit_train_summary**: H018 의 train.py 끝에 SUMMARY 블록
  의무. verify-claim 이 처음으로 §18.8 format 으로 받음.
