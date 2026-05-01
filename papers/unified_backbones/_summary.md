# Unified Backbones — Summary

> **Backbone-level UNI-REC redesign 카테고리.** "E004/E009 block에 또 한 개 augmentation"이 아니라, sequential × feature-interaction을 **단일 backbone 안에서 first-class로 통합**하는 트랜스포머 계열 아키텍처. method-transfer 1순위 후보. iter-17 cold-start.

## Takeaway (3–5 lines)
- 16-iter 캠페인은 E004→E009 base block에 5회 연속 augmentation을 시도(H011/H012/H013/H014)했으나 모두 CASE 2b plateau — sample-scale iterative deltas SATURATED.
- 외부 SOTA 3개(OneTrans Tencent WWW 2026 / InterFormer Meta CIKM 2025 / PCVRHyFormer 대회 baseline)는 모두 **single-stream transformer + token-level UNI-REC fusion** 패턴으로 수렴 — 우리도 backbone-level redesign 후보로 진입 시점.
- §10.8 rotation: 직전 2 H = H013 target_attention + H014 external_inspirations → `unified_backbones/` FREE first-touch이고 carry-forward 충돌 없음.
- 핵심 design lever: ① S/NS-token split, ② parameter-free token chunking (RankMixer NS tokenizer), ③ mixed causal attention mask, ④ 3-arch bidirectional bridging.
- 샘플 스케일 위험: 백본 교체는 §10.6 param budget(≤ 2146)을 쉽게 초과 — H015 design은 **delta-on-E009** 형태로 시작해 budget 안에서 token-restructuring만 시도하는 minimal variant부터.

## Entries
- `onetrans_tencent.md` — Tencent WWW 2026, arXiv:2510.26104. **S-tokens (sequence) + NS-tokens (non-sequence) single-stream 트랜스포머** + mixed causal attention + pyramid pruning. UNI-REC unification을 attention pattern 자체로 해결.
- `interformer_meta.md` — Meta CIKM 2025, arXiv:2411.09852. **3-arch (Interaction × Sequence × Cross) bidirectional bridging.** 두 축이 서로 read-write하도록 explicit cross-arch attention.
- `pcvrhyformer_baseline.md` — TAAC 2026 organizers, `competition/` baseline. **HyFormer-named multi-domain seq encoders → per-domain query decoders → joint token fusion → RankMixerBlock** with parameter-free NS-token equal-split. Dual optimizer (Adagrad sparse / AdamW dense). 2개의 CLAUDE.md 위반(§4.3 Row-Group split, §4.4 OOF 부재) — 운영 시 패치 필수.

## Carry-forward rules (post-H013/H014, applied to backbone designs)
- **Rule UB-1 (from §10.6 sample budget)**: 백본 교체 H는 sample-scale에서 trainable params ≤ 2146 (E002의 2배) 안에서 **delta-on-E009** 형태로 minimal variant 우선. 688k-param full backbone은 full-data 도착 전까지 archival.
- **Rule UB-2 (from H014 F-2 arm-conditional antagonism)**: backbone redesign이 새 loss term이나 confidence-shaping을 동반할 경우, x0 LN+DCN-V2 anchor가 보존되어야 seq_only arm divergence 방지. NS-token chunking이 x0 path를 끊으면 분산 위험.
- **Rule UB-3 (from H013 F-2 mean-pool redundancy)**: 백본의 token-level 새 channel은 seq encoder가 이미 소비하는 tensor의 scalar READ가 아니어야 함 — 정보 OUTSIDE-of-encoder-input span (recency, session boundary, cross-domain match, anchor query) 형태로.
- **Rule UB-4 (from H010 F-1 softmax routing collapse)**: backbone에 softmax-routed multi-expert/attention block을 추가하는 경우 sample-scale에서 uniform-collapse 트랩에 빠짐. Hard routing OR n_experts ≤ 2 OR full-data 한정.
- **Rule UB-5 (from competition/baseline §4.3/§4.4 위반)**: 외부 backbone 코드를 참조 시 split/holdout이 우리 §4.3 (label_time) §4.4 (10% OOF seed=42) 와 호환되도록 패치 후에만 sample-scale run. Row-Group split + 누락 OOF는 차단 사유.

## Candidate sources to fetch later
- HSTU (Meta-Hierarchical Sequential Transducers) — 본 연도 P2 long-seq trunk 후보, 별도로 `long_seq_retrieval/`에도 entry 가능
- HyFormer 원논문 — PCVRHyFormer가 명명한 family의 정의 출처, 별도 카드 가치 있음
- TIM (Target-aware Interaction Mixer) 2024 — RankMixer 계열 query boosting 변형 추적
