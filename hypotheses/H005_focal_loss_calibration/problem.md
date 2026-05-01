# H005 — Focal loss calibration (PCVRHyFormer-anchor 위 첫 mutation)

## What we're trying to explain
PCVRHyFormer-anchor (E_baseline_organizer, val_AUC=0.8251) 는 BCE loss 로 학습.
demo_1000 EDA 기준 prior=0.124 (876:124), 즉 **moderate class imbalance**. 학습
신호의 **majority class (negatives) 가 gradient 를 dominate** 해 hard-positive
(소수 클래스 어려운 샘플) 의 학습이 underweighted 될 수 있다. Focal loss
(Lin et al. ICCV 2017) 는 well-classified 샘플의 loss 를 down-weighting (`(1−p)^γ`
modulating factor) 하고 class balance term `α` 로 minority class gradient 강화 →
hard-positive 학습 강화 + AUC 보정. CTR 도메인에서 표준화된 lever 중 하나.

## Why now
- H004 verdict F-3 발동: 두 anchor 공존, **PCVRHyFormer-anchor 단기 우세** (0.8251 > 0.8174).
  미래 H 큐 PCVRHyFormer-anchor 위 mutation 우선.
- §17.4 카테고리 rotation 의무: H001 (unified_backbones) → H002 (unified_backbones)
  → H004 (unified_backbones, anchor exemption) → **H005 = loss_calibration** 로
  rotation 첫 충족. unified_backbones 4회 연속 차단 룰 활성 막음.
- §17.2 one-mutation 깔끔: `--loss_type bce` → `--loss_type focal --focal_alpha 0.25 --focal_gamma 2.0`
  CLI 변경 only. 코드 수정 0. PCVRHyFormer 가 focal 이미 지원 (train.py 의 loss
  branch).
- 비용 cheap: smoke ~3min (E_baseline_organizer envelope, attention overhead 영향 없음).
  T2 budget 안에서 무시 가능.
- §10.4 P1+ external_inspirations 의무 주입은 본 H 미충족 (loss_calibration 별도
  카테고리). H006 후보로 carry-forward.

## Scope
- In:
  - Loss: BCE → focal(α=0.25, γ=2.0). Lin et al. 표준 하이퍼파라미터.
  - 그 외 모든 config: PCVRHyFormer-anchor (E_baseline_organizer) 와 byte-identical envelope:
    organizer row-group split, train_ratio=0.05, num_epochs=1, seq_max_lens 절반
    (a:64,b:64,c:128,d:128), NS=5+2=7, num_queries=2, d_model=64, num_hyformer_blocks=2,
    seed=42.
  - 결함 A/B/C/D/E 패치 인프라 H001 그대로 (dataset.py, train.py path defaults,
    make_schema.py, infer.py prior fallback).
- Out:
  - Backbone 변경 (PCVRHyFormer 유지 — H004 OneTrans-anchor 는 archive-pending).
  - 시퀀스/NS-token granularity 변경.
  - Architecture mutation.
  - Multi-seed (P2 통과 후 별도).

## UNI-REC axes
- Sequential: 변경 없음 — TransformerEncoder per-domain 그대로.
- Interaction: 변경 없음 — RankMixerNSTokenizer + MultiSeqHyFormerBlock 그대로.
- Bridging mechanism: 변경 없음 — block fusion 그대로.
- **Loss-level effect**: focal 의 modulating factor 가 sample-level gradient 재가중
  → 두 axis 모두에서 hard-positive 패턴이 더 강하게 학습됨. 직접 axis 강화는 아니지만
  axis representation 의 calibration 개선.
- **primary_category**: `loss_calibration` (§17.4 rotation 첫 충족).
- **Innovation axis**: 본 H 는 axis 자체가 아니라 **학습 신호 calibration** 측면. §0
  의 두 축 동시 의무는 anchor (PCVRHyFormer) 가 이미 충족, 본 H 는 그 위에 loss
  calibration 만 적용.

## Success / Failure conditions
**§17.3 binary lift 임계 적용** (anchor exemption 아님 — one-mutation):

- **Success**: Δ vs PCVRHyFormer-anchor (val=0.8251) ≥ **+0.5 pt** → val_AUC ≥
  **0.8301**. **+ 4 부수 게이트**:
  1. Train 1 epoch NaN-free 완주.
  2. `submission/local_validate.py` 5/5 PASS (G1–G6).
  3. `metrics.json` 에 `{seed, git_sha, config_sha256, host, focal_alpha, focal_gamma}` 모두 채워짐.
  4. logloss 가 BCE-baseline (0.2538) 대비 악화 < 10% (focal 은 logloss 수치 자체엔
     불리하지만 +50% 이상 악화는 기능 이상).
- **Failure**:
  - Δ < +0.5 pt → **REFUTED**. focal_loss_calibration 방향 retire (LossFocal 쪽
    추가 carry-forward 가능: γ tuning 제외).
  - 부수 게이트 1–4 중 1개라도 fail → 코드/계약 위반 신호. 별도 진단 후 retry.

## Frozen facts referenced
- E_baseline_organizer val_AUC=0.8251 (H001 verdict).
- demo_1000 prior 0.124 (CLAUDE.md §3 / EDA out — demo-only, full-data prior 는 미공개).
- H002 val_AUC=0.8248 (control 비교 기준 정확성 검증 — 같은 envelope 에서 BCE 가
  bridges 변경 후 0.0003pt 차이).
- H004 val_AUC=0.8174 (OneTrans-anchor, archive-pending).
- `papers/loss_calibration/` 카테고리 미존재 — 본 H 의 paper card (focal Lin et al. 2017)
  을 lit_refs.md 에서 직접 인용. 카테고리 디렉토리 신설은 H005 launch 후 결과
  도착 시 (literature-scout) 처리.

## Inheritance from prior H
- H001 의 결함 A/B/C/D/E 패치 인프라 그대로 재사용.
- H002 의 cloud handoff discipline + flat upload + run.sh local-fallback 제거 운영
  패턴 그대로 재사용.
- H004 의 backbone router (`--backbone hyformer` default) + OneTrans 코드 무관 (사용 안 함).
  단 코드는 H004 패키지 그대로 재사용 (model.py 1714→2023 lines, OneTrans 미사용 path).
- H004 verdict F-3, F-5 carry-forward: PCVRHyFormer-anchor 위 mutation, loss_calibration 카테고리.
