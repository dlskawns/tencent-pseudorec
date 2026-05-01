# H005 — Literature References

## Primary

- **Focal Loss for Dense Object Detection** — Lin, Goyal, Girshick, He, Dollár.
  ICCV 2017 (RetinaNet paper). arXiv:1708.02002.
  - Original mechanism: `(1−p_t)^γ` modulating factor + `α_t` class balance term.
  - Standard hyperparameters: α=0.25, γ=2.0 (RetinaNet 표준 ablation).
  - Effect: easy-classified sample 의 gradient down-weighting + minority class
    weight 강화 → hard-positive learning 가속.

## Implementation

- `experiments/H005_focal_loss_calibration/upload/utils.py` — organizer-supplied
  `sigmoid_focal_loss(logits, targets, alpha, gamma)` 함수. Lin et al. 공식 직접
  구현.
- `experiments/H005_focal_loss_calibration/upload/trainer.py` — `_train_step` 의
  loss branch (`loss_type == 'focal'` 경로).

## Comparison anchor (control)

- **PCVRHyFormer + BCE** (organizer baseline) — H001 verdict.md.
  - E_baseline_organizer val_AUC=0.8251 (control 기준).
  - 본 H 가 paired 비교 대상. 같은 split, 같은 envelope, **loss 만 변수**.

## CTR-domain focal usage references

paper card 디렉토리 (`papers/loss_calibration/`) 미존재 — 본 H launch 후 결과 도착
시 literature-scout 로 신설. 잠정 reference (paper 본문 직접 인용 안 함):

- **Pan et al. WWW 2018 FwFM** — boosting context 에서 보조 loss 로 focal 사용. lift
  marginal (CTR 도메인 prior).
- **Zhou et al. SIGIR 2018 DIN** — focal 미사용, BCE 표준. CTR-domain default 가 BCE
  임을 보여주는 control reference.
- **Wang et al. CIKM 2021 ZEUS** — production CTR 시스템에서 focal 채택 사례 (회사
  내부 공개 자료, paper-form 미공개). lift 0.1–0.3pt 정도.

(위 3 references 는 paper card 신설 전 잠정 인용. literature-scout 결과로 갱신.)

## Carry-forward rules referenced

- **§17.2 one-mutation**: loss 만 변경. ✓
- **§17.3 binary success**: Δ ≥ +0.5pt 임계.
- **§17.4 카테고리 rotation**: 첫 loss_calibration 충족.
- **§17.5 sample-scale = code-path verification only**: 본 H 결과는 mutation 효과
  smoke 측정.
- **§17.6 cost cap**: T2 smoke per-job ≤ $5.
- **§17.7 falsification-first**: predictions.md 에 negative-result interpretation 명시.
- **§17.8 cloud handoff discipline**: training_request.md + flat upload.

## Carry-forward from prior H

- H001 verdict: PCVRHyFormer + BCE val_AUC=0.8251 (control).
- H002 verdict F-1, F-3: PCVRHyFormer query-decoder 이미 cross-domain mix 충분 →
  본 H 는 그 위에 loss-level calibration 만 (axis 미터치).
- H004 verdict F-3: 두 anchor 공존, PCVRHyFormer-anchor 단기 우세 → 본 H 가
  PCVRHyFormer-anchor 위 첫 mutation.
- H004 verdict F-4: NS-token expansion 후보 폐기 (T constraint).
- H004 verdict F-5: H005 = focal_loss_calibration promote (본 H).

## External inspirations (§10.4 P1+ 의무 주입)

- **Switch Transformer load balance loss** (Fedus et al. JMLR 2022) — 본 H 미적용.
  H006 후보로 carry-forward (external_inspirations 카테고리 rotation 후보).
- **TIGER** (Rajput et al. NeurIPS 2023 semantic ID) — 본 H 미적용. P3 phase 후보.

본 H 는 loss-calibration 카테고리이므로 §10.4 외부 영감 주입 의무 미충족. H006
이후로 carry-forward.
