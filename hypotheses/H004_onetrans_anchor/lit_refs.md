# H004 — Literature References

## Primary

- **OneTrans** (Tencent UNI-REC team, WWW 2026) — arXiv:2510.26104.
  - Paper card: [`papers/unified_backbones/onetrans_tencent.md`](../../papers/unified_backbones/onetrans_tencent.md).
  - 본 H 의 backbone source. Mixed causal mask + S/NS token + single-stream + pyramid pruning.
  - 우리는 mixed causal mask + S/NS token + single-stream 만 채택. pyramid pruning 별도 H.

## Comparison anchor (control)

- **PCVRHyFormer** (organizer baseline, TAAC 2026 사이트).
  - Paper card: [`papers/unified_backbones/pcvrhyformer_baseline.md`](../../papers/unified_backbones/pcvrhyformer_baseline.md).
  - H001 anchor (E_baseline_organizer val_AUC=0.8251).
  - 본 H 가 OneTrans-anchor 로 paired 비교 대상. 미래 H 들은 max(PCVRHyFormer, OneTrans) anchor 위에서 mutation.

## Sequential axis references (§0)

- **SASRec** (Kang & McAuley, ICDM 2018) — causal self-attention 기준. OneTrans S→S sub-mask 가 SASRec 의 attention 과 동등.
- **HSTU** (Meta, 2024) — long-context sequence backbone. 본 H 미적용 (별도 H 후보, longer_encoder 카테고리).
- **DIN/DIEN** (Alibaba) — target attention. OneTrans candidate token 의 target attention 이 DIN candidate-aware attention 과 등가.

## Interaction axis references (§0)

- **DCN-V2** (Wang et al., WWW 2021) — explicit cross. OneTrans NS→NS full attention 이 DCN-V2 의 explicit cross 를 attention 으로 재구현.
- **CAN** (Bian et al., KDD 2022) — co-action network. NS→S 의 token-level cross 와 직접 비교.
- **FwFM / FmFM** (Pan et al., WWW 2018; Sun et al., WSDM 2021) — field-weighted factorization. NS-token equal-split 이 FwFM 의 field 처리와 유사.

## Carry-forward rules referenced

- **§10.9 OneTrans softmax-attention entropy abort** — 본 H 가 룰의 첫 active 적용. attention prob mean entropy ≥ 0.95·log(N) ⇒ abort.
- **§10.5 LayerNorm on x0** — Pre-LN convention 으로 자동 충족.
- **§10.6 sample budget cap** — anchor 면제 (H001 과 동일).
- **§17.2 one-mutation** — anchor 면제 (backbone 전체 replacement).
- **§17.4 카테고리 rotation** — challengers.md §재진입정당화 충족.
- **§17.5 sample-scale = code-path verification only** — anchor 자격 검증 용도.
- **§17.7 falsification-first** — predictions.md 에 4 게이트 모두 interpretable.
- **§17.8 cloud handoff discipline** — training_request.md + flat upload + git_sha pin.

## Carry-forward from prior H

- H001 verdict.md: PCVRHyFormer anchor val_AUC=0.8251 (control 비교 baseline), 결함 A/B/C/D/E 패치 인프라 (재사용).
- H002 verdict.md F-1, F-3: cross-domain mix 가 PCVRHyFormer query-decoder 후 redundant → 본 H 가 layer-level fusion 으로 통합 깊이 한 단계 안으로.

## External inspirations (§10.4 P1+ 의무 주입)

- **Switch Transformer** (Fedus et al., JMLR 2022) — load balance loss. 본 H 미적용 (별도 H 후보, external_inspirations 카테고리).
- **TIGER** (Rajput et al., NeurIPS 2023) — semantic ID. 본 H 미적용 (P3 phase 후보).

본 H 는 anchor 이므로 §10.4 외부 영감 주입 의무 anchor 면제. 미래 mutation H 들에서 의무화.
