# H010 — anchor recalibration at extended envelope

## What we're trying to explain

H006/H007/H008/H009 모든 measurement 가 **extended envelope** (10ep × 30%,
patience=3-5, halved seq_max_lens) 에서 측정. anchor (original_baseline) 만
**smoke envelope** (1ep × 5%, patience=5) 에서 측정.

H007 verdict F-3 가 명시: "**mechanism 효과 + envelope 효과 합산 +0.005pt**.
mechanism 단독 효과 separate 하려면 추후 anchor recalibration H 필요."

H009 verdict F-3 이 정량 노출: anchor 정확값 0.83 가정 vs 0.835 가정에서
H009 의 P2 분류 (additive vs sub-additive vs interference) 가 갈림. 즉 후속
모든 H 의 paired Δ 결론이 anchor 정확값에 의존 — 현재 우리는 anchor 를
range "~0.83~0.835" 로만 알고 있음.

본 H = original_baseline 코드 (mechanism mutation 0) 를 H006~H009 의 동일
extended envelope (10ep × 30%, patience=3) 에서 학습 → **순수 envelope 효과만
isolation**. 결과:

- **anchor extended platform AUC** = 새 ground truth.
- 기존 H007/H008/H009 의 paired Δ 를 **이 새 anchor** 에 다시 맞춰 재계산.
- mechanism 효과 (target_attention, sparse_feature_cross, hybrid stacking) 의
  진짜 lift 분리.

## Why now

- **§17.2 single mutation 깔끔 (envelope only)**: mechanism 코드 변경 0,
  CLI args 만 envelope (smoke → extended) 변경. 가장 깨끗한 mutation.
- **§17.3 binary 임계 적용 안 함**: anchor 측정이라 ≥ +0.5pt 룰 미적용.
  measurement objective.
- **H007 F-3 carry-forward 직접 후속**: 이미 명시된 미래 H. 미루다 H009 에서
  anchor 의존성 정량 노출 → priority 1순위 confirmed.
- **H009 F-3 직접 후속**: anchor 정확값이 결론 분류 흔드는 것 직접 확인.
- **§17.6 cost cap 효율**: 한 번 측정으로 모든 기존 H paired Δ 재계산 가능.
  ~3-4시간 wall, T2 cap 안.
- **H011+ ground truth 확보**: 모든 후속 H 의 baseline 명확.
- **§17.4 카테고리 rotation**: anchor 측정이라 mechanism category 부여 안 함
  (n/a). rotation 룰 적용 안 됨.

## Scope
- In:
  - **코드: original_baseline/upload/ 의 12 파일을 그대로 복사**. mechanism
    mutation 0. H006~H009 의 model.py 변경 (longer encoder, candidate xattn,
    DCN-V2 등) 어떤 것도 포함 안 함. **PCVRHyFormer pure baseline + label_time
    split + 10% OOF + §18 인프라 룰**.
  - run.sh envelope 변경:
    - `--num_epochs 1` → `--num_epochs 10`.
    - `--train_ratio 0.05` → `--train_ratio 0.3`.
    - `--patience 5` → `--patience 3` (H008 F-4 carry-forward).
    - 그 외 모든 args byte-identical.
  - card.yaml 의 envelope 항목만 변경.
  - 기타 모든 config: original_baseline 와 byte-identical.
- Out:
  - 어떤 mechanism mutation 도 추가 안 함.
  - DCN-V2/candidate xattn/longer encoder 등 H006~H009 의 클래스 추가 코드는
    upload 패키지에 포함 안 함.
  - hyperparameter tuning (lr, dropout 등) — 별도 H.
  - 데이터 변경 (train_ratio 외 sampling 변경 등) — 별도 H.

## UNI-REC axes
- **Sequential axis**: 변경 없음 — original_baseline 의 transformer encoder +
  query decoder 그대로.
- **Interaction axis**: 변경 없음 — RankMixerBlock fusion 그대로.
- **Bridging mechanism**: 변경 없음. measurement objective.
- **primary_category**: n/a (envelope mutation, mechanism category 부여 안 함).
- **Innovation axis**: n/a.

## Success / Failure conditions

§17.3 binary lift 임계 미적용 — measurement objective.

**Primary measurement**:
- `platform_AUC_anchor_extended` = anchor 의 extended envelope ground truth.
- **두 시나리오 분기**:
  - **A. anchor extended ≈ 0.83~0.835** (smoke 와 비슷): envelope 효과 작음
    → H007/H008 lift 가 mechanism 효과 dominant. 기존 결론 유지.
  - **B. anchor extended ≈ 0.835~0.840** (smoke 보다 +0.5pt 이상 높음): envelope
    효과 large → H007/H008 lift 의 일부가 envelope. mechanism 효과 separate
    필요. H011+ 우선순위 재정렬.
  - **C. anchor extended < 0.83** (smoke 보다 낮음): extended envelope 자체가
    overfit/regression — 매우 의외. envelope 룰 자체 재검토.

**부수 게이트**:
1. Train 10 epoch NaN-free 완주 (또는 patience=3 early stop 정상).
2. Inference: §18 인프라 통과 (batch heartbeat + `[infer] OK` 로그, no fallback).
3. `metrics.json` 에 `{seed, git_sha, config_sha256, host, best_val_AUC,
   best_oof_AUC, num_epochs=10, train_ratio=0.3, patience=3}` 모두 채워짐.
4. infer.py: original_baseline 와 byte-identical (새 cfg key 없음).

## Frozen facts referenced
- Anchor (original_baseline smoke) Platform AUC: ~0.83X (range, exact unknown).
- H006 Platform 0.82 (refuted, longer encoder).
- H007 Platform 0.8352 (PASS marginal, candidate xattn).
- H008 Platform 0.8387 (PASS, DCN-V2 fusion swap, **현재 최고**).
- H009 Platform 0.8364 (REFUTED interference, combined H007 + H008).
- H006/H007/H008/H009 모두 extended envelope (10ep × 30%, patience=3-5).
- §18 인프라 룰 (CLAUDE.md 신설 2026-04-28) — original_baseline 패키지에 이미
  포함.

## Inheritance from prior H

- **H007 F-3**: anchor extended 측정 미래 H 명시 — 본 H 로 직접 충족.
- **H008 F-4**: patience=3 + early stop aggressive — 본 H envelope 에 적용.
- **H009 F-3**: anchor 정확값 의존성 노출 → 본 H priority 강화.
- **§18 인프라 룰**: original_baseline 패키지가 이미 포함 → byte-identical 재사용.
