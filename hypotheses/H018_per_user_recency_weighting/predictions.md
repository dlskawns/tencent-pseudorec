# H018 — Predictions

> H015 sibling. **per-user × exp** specification within recency mechanism
> class. Paired Δ primary vs H015 (granularity isolation), secondary vs
> H010 corrected (anchor recovery check).

## ⚠️ Updated expectation (2026-05-03 post H011-H013 trajectory analysis)

**New evidence (Recent Findings F-A through F-D)**:
- 4 H (H011/H012/H013/H015) val→platform gap = mean −0.003pt (consistent direction).
- H012 corrected estimate (~0.8350) vs H015 corrected (0.83805) ≈ ceiling tied. 두 mechanism class (MoE multi-domain, recency loss) 모두 cohort drift hard ceiling 영역.
- best_epoch convergence early-mid 일관 (4-6 / 7-9), overfit gap 0.001~0.002pt 일관.

**Revised outcome distribution (was: equal 4-class)**:

| Outcome | Δ vs H015 | **Prior probability** | **Updated probability** | reasoning |
|---|---|---|---|---|
| strong | ≥ +0.005pt | ~25% | **~10%** | 4 H 누적 모두 H010 anchor 위 +0.005 도달 못 함 (H015 +0.0002 만, H012 ≈ H015 tied). per-user fine 만으로 ceiling 깨질 가능성 낮음. |
| measurable | [+0.001, +0.005pt] | ~25% | **~25%** | 변동 없음. per-user 효과 약 가능성. |
| noise | (−0.001, +0.001pt] | ~25% | **~50%** | per-user variance 가 per-batch 위 무 effect 가능성 높음 (cohort drift 가 진짜 ceiling). |
| degraded | < −0.001pt | ~25% | **~15%** | mean=1.0 normalize 가 loss scale 보존, gradient noise 위험 작음. |

**Implication**:
- noise outcome 가 50% 로 가장 likely → **H019 paradigm shift scaffold 사전 준비 정당**.
- strong PASS 가 unlikely 라도 falsification value 는 유지 (negative result 가 cohort hard ceiling 가설 강화).
- H019 = TWIN long-seq retrieval 사전 scaffold (이 PRD US-005). H018 결과 따라 즉시 launch.

## P1 — Code-path success

## P1 — Code-path success
- PASS expected. dataset.py `_convert_batch` 가 4 도메인 max ts_fid →
  `days_since_last_event` 계산, trainer.py exp decay branch + per-batch
  normalize + clip.
- Failure mode: 도메인 ts_fid 추출 실패 (§18.7 nullable check 필수). 또는
  `timestamp − max_seq_ts` 가 모두 음수 (사용자 schema 오해석) → all-zero
  weights → loss=0 → NaN.

## P2 — Primary lift (vs H015 corrected, §17.3 binary)

| 비교 | classification | Δ 임계 | mechanism implication |
|---|---|---|---|
| **Δ vs H015 corrected** | strong | ≥ +0.005pt | per-user granularity 가 핵심 lever, mechanism class 영구 confirm |
| **Δ vs H015 corrected** | measurable | [+0.001, +0.005pt] | per-user 약 effect, mechanism class 약, retire 권고 |
| **Δ vs H015 corrected** | noise | (−0.001, +0.001pt] | per-user 무 effect, granularity 가설 REFUTED |
| **Δ vs H015 corrected** | degraded | < −0.001pt | per-user weight variance 가 학습 disrupt |

**§17.3 binary cut**: Δ ≥ +0.5pt = strong PASS. (단, sample-scale relaxed
임계는 +0.005pt — H015 가 +0.0002 이라 동일 magnitude 비교).

**Secondary**: Δ vs H010 corrected anchor (0.837806).
- Δ ≥ +0.005pt → anchor 갱신 후보.
- Δ ∈ [+0.001, +0.005pt] → marginal, anchor 보류.
- Δ < +0.001pt → anchor 미달.

## P3 — Mechanism 작동 검증

- **per-user weight distribution stat** (학습 로그 마지막 단계 print):
  - mean ≈ 1.0 (normalize 작동 확인).
  - p10/p50/p90/p99 — 합리적 spread (예: 0.2/0.9/1.4/2.5).
  - 만약 모두 ≈ 1.0 이면 → tau 가 너무 크거나 (decay 효과 미미) gap 분포가
    너무 좁음 → **mechanism 무력 신호**.
- **clip 발동률**: weight < 0.1 또는 > 3.0 비율 < 5% expected. > 20% 면
  clip 이 효과 dominate → tau 재선택 sub-H.
- **NS xattn entropy**: H010 baseline [0.81, 0.81] 변화 미세 expected
  (loss weighting 만 변경, attention layer 영향 없음).

## P4 — §18 인프라 PASS

- §18.7 (nullable `to_numpy`): `label_time` / `label_type` fill_null +
  zero_copy_only=False. 신규 derived 컬럼 (`days_since_last_event`) 은
  numpy array 라 영향 없음.
- §18.8 (emit_train_summary): train.py 마지막에 SUMMARY 블록 출력 의무.
  verify-claim 이 처음으로 §18.8 format 으로 paired Δ 계산.
- §18.6 dataset-inference-auditor invocation: upload/ ready 직전 PASS.

## P5 — val ↔ platform gap (H016 infra carry-forward)

- best_val_AUC vs platform AUC gap. H015 +0.0023pt / H016 +0.0080pt 가
  prior baseline. H018 expected: gap ≈ +0.005pt (H015~H016 중간).
- gap > +0.01pt → val 신호 신뢰도 약화 → 다음 H 의 의사결정 비용 증가.

## P6 — OOF (redefined) ↔ Platform gap

- H016 redefined OOF gap −0.004pt = baseline. H018 expected: gap ∈
  [−0.005, +0.005pt] (H016 framework 동일하게 cohort align).
- gap > +0.01pt → cohort drift 다시 벌어짐 → recency mechanism class 의
  cohort handling 한계 노출.

## P7 — verify-claim §18.8 SUMMARY 파싱 dry-run

- **First-ever H** 가 §18.8 format 사용. 사용자가 SUMMARY 블록 paste 시
  verify-claim 이 모든 필드 (best/last/overfit/calib/epoch_history) 정확히
  추출하는지 검증.
- 실패 시: parser regex 수정 (verify-claim SKILL.md §0).

## Decision tree (post-result)

| Outcome | Action | next H 후보 |
|---|---|---|
| **strong** Δ vs H015 ≥ +0.005pt | per-user granularity 검증, mechanism class 영구 saved. anchor = H018. | H019 = tau sweep sub-H (7, 30) 또는 per-user × per-batch hybrid sub-H |
| **measurable** Δ vs H015 ∈ [+0.001, +0.005pt] | per-user 약 effect. mechanism class 약, retire 권고. | H019 = paradigm shift (TWIN retrieval / OneTrans). temporal_cohort retire. |
| **noise** Δ vs H015 ∈ [−0.001, +0.001pt] | granularity 가설 REFUTED. recency mechanism class 영구 retire. | H019 mandatory paradigm shift. cost cap audit (T3 가능 여부). |
| **degraded** Δ vs H015 < −0.001pt | per-user variance 학습 disrupt. | H018-sub = clip [0.5, 1.5] tight cap 또는 tau=30 smoother. paradigm shift 우선순위 ↑. |
| **P5 fail** (val-platform gap > +0.01pt) | val 신호 약화. | 다음 H 의 의사결정에 platform paid 검증 필수. |
| **P6 fail** (OOF-platform gap > +0.01pt) | cohort handling 한계. | recency mechanism class retire 가속. |

## Falsification claim (반증 가능)

H018 의 mutation 은 **단 1개의 측정 가능한 claim**:
> "Per-user exp time-decay weighting 이 per-batch linear weighting (H015)
> 위 corrected platform AUC Δ ≥ +0.005pt 추가 lift 만든다."

위 claim 이 거짓 (Δ < +0.005pt) → recency mechanism class 의 cohort drift
attack 가설이 위 형식으로 작동 안 한다는 강한 evidence.
