# H033 — Method Transfer

## ① Source

- **H019 (TWIN paradigm shift, Tencent 2024 RecSys)** — base. cloud measurable PASS 0.839674.
- **H020 (learnable GSU)** — scoring quality axis sub-H. patch carry.
- **H021 (per-domain top_k)** — scoring quantity axis sub-H. patch carry.
- 카테고리 (`retrieval_long_seq/`): re-entry from H019/H020/H021. stacking sub-H.

## ② Original mechanism

H020 + H021 의 두 patch 동시 적용. H020 patch (TWINBlock learnable_gsu support) + H021 patch (PCVRHyFormer int|dict twin_top_k 처리) 가 직교라서 간섭 없이 합쳐짐.

## ③ What we adopt

- **Mechanism**: H020 + H021 동시 mutation. base = H020 upload (이미 learnable_gsu 적용) + H021 의 wiring 추가.
- **변경 내용 (4 files)**:
  - `model.py`: PCVRHyFormer __init__ 의 `twin_top_k` 가 int|dict 둘 다 받게 (H021 carry). TWINBlock 의 `learnable_gsu` (H020 carry).
  - `train.py`: `--twin_top_k_per_domain` argparse + `_parse_twin_top_k_arg` helper (H021 carry). `--twin_learnable_gsu` (H020 carry).
  - `run.sh`: 둘 다 flag bake.
  - `README.md`: H033 identity.
- **CLI**: `--use_twin_retrieval --twin_learnable_gsu --twin_top_k_per_domain "64,64,64,96" --twin_gate_init -2.0 --twin_num_heads 4`.

## ④ What we modify (NOT a clone)

- **Stacking 형태**: TWIN paper 의 single GSU+ESU 와 다름 — 본 H = learnable GSU + per-domain K 동시.
- **per-domain K 분배**: §3.5 정량 motivation (domain seq_d p90=2215). H020 의 learnable scoring 이 추가 token 에 더 잘 fit 하면 stacking 시너지.
- **§17.2 strict reading 위반**: 2 axis. 정당화 = stacking H 의 *발견 → 통합* 단계 구분 (challengers.md).

## ⑤ UNI-REC alignment

- Sequential reference: H019 carry — TWIN GSU+ESU per-domain.
- Interaction reference: 변경 없음 (DCN-V2 + H010 NS xattn).
- primary_category: `retrieval_long_seq` (re-entry, 4회 연속).
- Innovation axis: 직교 axis 통합. UNI-REC sequence axis 의 다중 lever 통합 검증.

## ⑥ Sample-scale viability

- 추가 params: H020 (8K) + H021 (0) = +8K. TWIN module = H019 71K + 8K = 79K. total 161M 의 0.049%.
- §10.6 sample budget 친화 (H020 carry).
- T0 sanity: TWINBlock(K=96, learnable_gsu=True) forward NaN-free 검증 완료.

## ⑦ Carry-forward rules

- **§17.2 one-mutation**: 위반 인지. 정당화 = challengers.md (stacking H insurance).
- **§17.3 binary success**: Δ vs H019 ≥ +0.003pt strong / [+0.001, +0.003pt] measurable / (−0.001, +0.001pt] noise / < −0.001pt degraded.
- **§17.4 rotation**: retrieval_long_seq 4회 연속 RE_ENTRY_JUSTIFIED.
- **§10.5/§10.9/§10.10**: H019 carry.
- **§18.6/§18.7/§18.8**: H019 carry. dataset.py / infer.py / make_schema.py / TWINBlock byte-identical (H020 carry).
