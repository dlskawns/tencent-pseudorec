# H002 — InterFormer inter-domain bridges between per-domain seq encoders

## What we're trying to explain
Organizer baseline (E_baseline_organizer) achieves val_AUC=0.8251 (5% data, 1 epoch). 4개 도메인 (a/b/c/d) seq encoder 가 `MultiSeqHyFormerBlock` 안에서 **완전히 독립** — 정보 공유 0. 도메인 간 user behavior 패턴 (e.g., domain A 에서 본 product line이 D 에서도 등장하는 cross-domain signal) 이 model 내부에서 전혀 흐르지 않고, 마지막 RankMixer 단계에서만 query token level로 joint 처리됨.

InterFormer paper (Meta CIKM 2025) 의 핵심 주장: **late fusion** (concat-only at head) 은 정보 손실. 매 layer 마다 architecture 간 bidirectional bridge 를 두면 정보 흐름이 풍부해지고 variance 감소 (paper: AUC variance 30–50% 감소).

## Why now
- H001 anchor 확보 (val_AUC 0.8251).
- 다음 H 후보 5개 중 InterFormer bridge가 **가장 작은 변경 + 가장 검증된 paper-derived 메커니즘**:
  - 변경 80줄 이하
  - param overhead 0.6% (12,312 / 2.18M = 0.57%)
  - paper 주장 lift +0.3–0.8 pt
- H001 transfer.md F-1 carry-forward: HyFormer는 4 도메인이 RankMixer 단계에서만 fuse — 그 사이 layer 들이 비어 있음.
- §17.4 카테고리 rotation: H001 → H002 둘 다 unified_backbones. **재진입 정당화** (challengers.md 참조).

## Scope
- In:
  - `_DomainBridge` 클래스 추가 (`model.py`, ~30줄): low-rank (rank=4) + scalar gate σ(α=-2 init).
  - `MultiSeqHyFormerBlock.__init__` + `forward` 수정 (~30줄): 4×3=12 bridges + per-domain pool + broadcast update.
  - `PCVRHyFormer.__init__` 에 3 args 노출 (`enable_inter_domain_bridges`, `bridge_rank`, `bridge_gate_init`).
  - `train.py` 에 같은 3 args CLI flag.
  - `infer.py` 에서 train_config.json 읽어 bridge 재구성.
- Out:
  - 다른 architecture mutation (OneTrans, multi-domain MMoE 등 — 후속 H로).
  - Hyperparameter tuning (rank, gate init — 본 H 통과 후 별도 ablation H).

## UNI-REC axes
- Sequential: 4 도메인 seq encoder 가 다른 도메인 정보로 enriched (paper R1).
- Interaction: NS-token path (RankMixer) 변경 없음.
- Bridging mechanism: **bridges 가 sequential axis 간 정보 흐름**. NS-token 과 seq 간 interaction (UNI-REC original) 은 RankMixerBlock 단계에서 그대로 유지. **CLAUDE.md §0 P1 (seq + interaction 한 블록 gradient 공유) 충족.**

## Success / Failure conditions
- **Success (anchor 능가)**: val_AUC ≥ 0.8301 (= 0.8251 + 0.5pt). 단 5% data + 1 epoch 기준.
- **Falsification**: 
  - val_AUC < 0.8301 → REFUTED. Bridge 방향 retire (다른 H로 진행).
  - bridge gate가 학습 후 모두 0.05 이하로 수렴 → 모델이 bridge 안 쓰겠다는 signal. (gate 역시 측정 의무.)

## Frozen facts referenced
- `papers/unified_backbones/interformer_meta.md` (R1, R2, R3)
- `experiments/E_baseline_organizer/ckpt/metrics.json` (anchor val_AUC=0.8251)
- demo_1000 검증: bridges enabled = +12,312 params (0.57%) — 로컬 smoke 확인 완료.

## Inheritance from prior H
- H001 (E_baseline_organizer): val_AUC=0.8251 control. 같은 split (organizer row-group, 100 valid RGs), 같은 train_ratio=0.05, 같은 seq_max_lens, 같은 seed=42 → **paired Δ 비교 가능**.
