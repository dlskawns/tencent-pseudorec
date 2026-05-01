# H005 — Method Transfer

## ① Source
**Focal Loss for Dense Object Detection** — Lin, Goyal, Girshick, He, Dollár.
ICCV 2017 (RetinaNet paper). arXiv:1708.02002.
PCVR/CTR 도메인 응용: 다수 paper (Pan et al. WWW 2018 FwFM 보조 loss, 다양한 production
CTR 시스템에 ad-hoc 채택). 우리 코드는 organizer 가 이미 제공 (`utils.sigmoid_focal_loss`,
`trainer._train_step` 의 focal branch).

## ② Original mechanism

표준 BCE: `L = −[y log p + (1−y) log(1−p)]`. easy-classified 샘플이 너무 많으면
easy-negative 의 gradient 합이 hard-positive 를 dominate.

Focal loss: `L = −α_t (1−p_t)^γ log(p_t)` where:
- `p_t = p if y=1 else 1−p` — 예측 확률 정렬.
- `(1−p_t)^γ` — **modulating factor**. p_t → 1 (well-classified) 이면 0 으로,
  p_t → 0 (hard) 이면 1 로 → easy 샘플의 loss down-weighting.
- `α_t = α if y=1 else 1−α` — class balance term. minority class (y=1) 에 더 큰
  weight α 부여.
- 표준 하이퍼: α=0.25, γ=2.0 (Lin et al. RetinaNet ablation 결과).

Effect: hard-positive 의 gradient relative weight 강화 → minority class learning 가속,
easy-negative dominance 완화. 일반적으로 imbalanced binary classification 에서
+0.1–2.0 pt AUC lift (paper-by-paper variance 큼).

## ③ What we adopt

- 표준 focal loss 정의 그대로 (organizer `utils.sigmoid_focal_loss` 가 Lin et al.
  공식 그대로 구현).
- Hyperparameters:
  - `--loss_type focal`
  - `--focal_alpha 0.25` (Lin et al. RetinaNet 표준)
  - `--focal_gamma 2.0` (Lin et al. RetinaNet 표준)
- 그 외 모든 hyperparameter (lr, sparse_lr, batch_size, seed, dropout, num_epochs,
  num_hyformer_blocks, d_model, NS-tokens 등) PCVRHyFormer-anchor (E_baseline_organizer)
  와 byte-identical.

## ④ What we modify (NOT a clone)

- **α=0.25 vs train.py default α=0.1**: train.py 의 `--focal_alpha` 기본값은 0.1
  (positive 에 매우 강한 weight). 우리는 Lin et al. ICCV paper 의 0.25 채택. 이유:
  - 0.1 = positive 에 90:10 inverse weighting → 우리 데이터 prior 0.124 와 거의 inverse.
    minority 강화 너무 aggressive 가능성.
  - 0.25 = positive 에 75:25 → 더 보수적. minority 약간 강화 + majority gradient 유지.
  - **이유 명시**: organizer default 0.1 은 production 데이터 prior (미공개) 에
    fitted 가능성. demo_1000 prior=0.124 외 검증 데이터 부재 → Lin et al. 표준 default
    가 더 reproducible.
- **γ=2.0 그대로** (paper standard, 변경 정당화 없음 → §17.2 one-mutation 깔끔).
- **배치 normalize 미변경**: focal 은 sample-level loss 라 batch-level scale 차이가
  optimizer step size 에 영향. 우리는 mean reduction (`F.binary_cross_entropy_with_logits`
  과 동일) 그대로. 이유: optimizer hyperparameter (lr=1e-4) 와 paired 비교 유지.
- **Loss 외 architectural change 없음**: §17.2 one-mutation 엄격 적용. focal 의
  학습 dynamics 가 결과적으로 sparse param re-init 패턴 (epoch 1 후 96 high-cardinality
  embeddings re-init) 과 interaction 가능성 있으나 본 H 에선 측정 안 함.

## ⑤ UNI-REC alignment

- **Sequential reference**: 변경 없음. PCVRHyFormer 의 per-domain TransformerEncoder
  (model.py:544) 그대로.
- **Interaction reference**: 변경 없음. RankMixerNSTokenizer (model.py:1070) 그대로.
- **Bridging mechanism**: 변경 없음. MultiSeqHyFormerBlock (model.py:850) 의 token
  fusion 단계 그대로.
- **Loss-level effect**: focal 의 modulating factor 가 sample 별 gradient 가중치를
  바꿔 hard-positive 의 representation update 강화. 직접 axis 강화 아니지만 두 axis
  의 representation calibration 개선.
- **primary_category**: `loss_calibration` (§17.4 rotation 첫 충족).
- **Innovation axis**: 본 H 는 axis 자체 mutation 아님. Axis-level mutation 들은
  H006+ 자료 (longer_encoder, target_attention).

## ⑥ Sample-scale viability (Rule UB-1, §10.6)

- 추가 trainable params: **0**. Loss 함수만 변경.
- Total params: PCVRHyFormer 198M (embedding tables dominant) 그대로.
- §10.6 sample budget cap (≤ 2146): anchor envelope 와 동일. 면제 (PCVRHyFormer-anchor
  와 동일).
- demo_1000 / smoke 47k rows 에서 focal 의 학습 효과 검증 가능 — Lin et al. 도
  validation 시 small subset 에서 modulating factor 효과 측정.

## ⑦ Carry-forward rules to honor

- **§10.5 LayerNorm on x0 MANDATORY**: 변경 없음. PCVRHyFormer Pre-LN 그대로 충족.
- **§10.6 sample budget cap**: anchor envelope 동일.
- **§10.7 카테고리 rotation**: H001/H002/H004 모두 unified_backbones → 본 H 가 첫
  rotation 충족 (loss_calibration). 추가 정당화 불필요.
- **§10.9 OneTrans softmax-attention entropy abort**: 본 H 는 OneTrans 미사용 →
  미적용. `--log_attn_entropy` flag 도 off (run.sh 에서 제거).
- **§10.10 InterFormer bridge gating σ(−2) init**: 본 H 는 새 bridge 추가 없음 →
  미적용.
- **§17.2 one-mutation-per-experiment**: loss 만 변경. ✓ 깔끔 충족.
- **§17.3 binary success**: Δ ≥ +0.5pt val_AUC vs E_baseline_organizer (0.8251).
  미달 → REFUTED.
- **§17.4 카테고리 rotation 첫 충족**.
- **§17.5 sample-scale = code-path verification only**: 본 H 결과는 mutation 효과
  smoke 측정. full-data 결과는 별도 row.
- **§17.7 falsification-first**: predictions.md 에 negative-result interpretation 명시.
- **§17.8 cloud handoff discipline**: training_request.md + flat upload + git_sha pin.
- **§4.3 label_time-aware split**: H001 패치 그대로 코드 보존 (`--use_label_time_split`
  flag), 단 본 H 는 organizer split 사용 (E_baseline_organizer 와 paired 비교).
- **§4.4 OOF holdout 10%**: organizer split 모드라 OOF 비활성. H001 organizer-pure
  envelope 와 동일.
