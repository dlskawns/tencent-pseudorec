# H011 — Problem Statement

## What we're trying to explain

CLAUDE.md §3 / §4.8 에 따르면 `user_int_feats_{fid}` 와 `user_dense_feats_{fid}`
는 **같은 fid 를 공유하는 aligned `<id, weight>` pair** — 동일 entity/signal
의 ID 와 강도를 jointly 표현.

**Verified shared (aligned) fids (출처: `competition/ns_groups.json`
`_note_shared_fids`, 2026-04-30 검증)**: `{62, 63, 64, 65, 66, 89, 90, 91}`
— **8 fids**.

**Dense-only fids (user_int 측 매칭 없음, aligned 효과 미적용)**: `{61, 87}`.

**user_dense_feats flat layout**: 10 fids (`{61, 62, 63, 64, 65, 66, 87,
89, 90, 91}`) 의 multi-dim list 가 concat 되어 per-row total_dim=918. per-fid
offset/dim 매핑은 `competition/dataset.py` 의 `_user_dense_plan` 참조.

**aligned fids 의 user_ns_groups 위치**: U2 (`[48, 49, 89, 90, 91]`) + U7
(`[3, 4, 55–59, 62–66]`) — 비-aligned fids 와 섞여 있음. group-aware binding
필요.

(이전 H011 scaffold 가 §3 의 미검증 "후보 9 fids" 직접 인용 → §4.9 위반.
Layer 3 gap 재생산 사례. Fix 1+2+3+4 적용 후 본 정정.)

**현재 baseline (PCVRHyFormer / H010 champion) 의 처리 방식 (코드 audit
결과)**:
- `user_int_feats` (46 fids) → `RankMixerNSTokenizer`: 46 fid embeddings concat
  → split → project → 5 NS tokens (`model.py:1760`).
- `user_dense_feats` (10-dim list<float>) → `user_dense_proj` (단일 Linear +
  LayerNorm) → 1 NS token (`model.py:1785`).
- **두 stream 이 완전히 분리** — `user_int[fid=61]` 와 `user_dense[fid=61]`
  사이의 binding 정보는 NS-token level cross-attention 에 도달하기 전까지
  표현 불가. 모델이 implicit 학습해야 함.

**측정 가능한 gap**: aligned pair 가 explicit binding 없이 지나가는 코드 path
가 §4.8 hard rule ("Aligned `<id, weight>` 는 항상 한 쌍으로 이동. 한쪽만
쓰는 코드는 leakage-audit 미통과") 를 위반. baseline 이 룰 미통과 상태일
가능성 — H011 첫 단계 = audit + binding 추가.

## Why now

직전 H 들이 **representation enrichment** 만 시도:
- H006 (longer encoder), H007 (candidate xattn), H008 (DCN-V2 fusion),
  H009 (combined), H010 (NS→S xattn) — 전부 sequence/interaction axis 의
  post-encoder 또는 fusion-stage mutation.
- **input embedding stage** 의 mutation 은 한 번도 안 측정.

H010 verdict carry-forward:
- F-1: stacking 안전 패턴 = NS-only enrichment. 본 H 도 input-stage 변경
  (anchor 의 NS xattn 출력 / fusion 입력 byte-identical) → 위치 충돌 위험 0.
- §10.7 rotation: H009 (hybrid, target_attention 포함) + H010 (target_attention)
  → H011 차단되는 카테고리 = target_attention. 본 H = `feature_engineering`
  / `interaction_encoding` (신규 카테고리, FREE first-touch).

§0 UNI-REC north star: sequence × interaction 통합. 우리는 sequence axis
(target_attention) 에 3번 invest, interaction axis 에 H008 1번뿐. **input
stage interaction** 은 0번. 새 axis 측정 의무.

## Scope

- **In**:
  - aligned fid pair `{62, 63, 64, 65, 66, 89, 90, 91}` (8 fids) 의 explicit
    binding mechanism: per-fid k 마다 `E_id(user_int_feats_k) * user_dense_feats[offset_k:offset_k+dim_k]`
    (element-wise broadcast 또는 group-aware reduction).
  - 통합 위치: input embedding lookup stage (NS tokenizer 입력 텐서 변경,
    NS tokenizer 자체와 downstream 텐서 byte-identical).
  - per-fid offset/dim 추출: `competition/dataset.py` 의 `_user_dense_plan`
    참조 (또는 `make_schema.py` 의 동등 산출).
- **Out**:
  - dense-only fids (`{61, 87}`): aligned 효과 미적용 (user_int 측 매칭 없음).
    `user_dense_proj` 단일 NS token 처리 그대로.
  - non-aligned user_int fids (38 fids = 46 − 8): 변경 없음.
  - item_int / item_dense 의 aligned pair (CLAUDE.md §3 의 item_dense
    명시 없음 — 사실상 0). 별도 sub-H 후보.
  - sequence aligned pair (domain_a/b/c/d_seq 의 dense weight). 별도 sub-H.
  - dense feature 의 분포 변환 (AutoDis bucketization 등). 별도 sub-H.
  - aligned pair 의 multi-layer interaction. minimum viable form 우선.

## UNI-REC axes

- **Sequential**: 변경 없음 (transformer encoder 그대로, NS xattn 그대로).
  단 NS tokenizer 입력 embedding 이 weighted form 으로 들어가서 NS 표현
  enrichment.
- **Interaction**: input embedding stage 에 explicit `<id, weight>` cross
  추가. DCN-V2 fusion (H008 anchor) 그대로 — NS-token level interaction 에
  enrichment 가 propagate.
- **Bridging mechanism**: input-stage `id × weight` element-wise → NS tokenizer
  → seq encoder + NS xattn (H010) → DCN-V2 fusion. **input stage**
  interaction enrichment 가 sequence (NS xattn) 와 interaction (DCN-V2)
  axis 모두에 propagate → §0 P1 ("seq + interaction 한 블록 gradient 공유")
  의 strongest form (input stage = 모든 downstream gradient 공유).

## Success / Failure conditions

- **Success (PASS)**:
  - §17.3 binary: Δ vs anchor (H010, Platform 0.8408) ≥ +0.5pt → Platform ≥
    0.8458.
  - 또는 (sample-scale 한계 인정) Δ vs anchor ≥ +0.001pt + paired bootstrap
    CI > 0 (seed×3 시).
  - audit 통과: aligned fid pair 매핑 confirmed, weighted embedding shape
    consistent, NaN-free 학습.
- **Failure (REFUTED)**:
  - Δ vs anchor < +0.5pt + Δ vs H008 (0.8387) < +0.001pt → input-stage
    binding 효과 marginal, mechanism class retire.
  - audit 실패 (fid 매핑 inconsistent, dense feature 가 fid 61 와 무관) →
    INVALID, 가설 자체 retract.
  - NS tokenizer dispatch 오류, NaN abort.

## Frozen facts referenced

- CLAUDE.md §3 (verified 2026-04-30): User Int Features 46, User Dense
  Features 10, aligned shared fids `{62, 63, 64, 65, 66, 89, 90, 91}` (8),
  dense-only `{61, 87}`, total_dim=918. 출처 = `competition/ns_groups.json`.
- CLAUDE.md §4.8: aligned `<id, weight>` 는 항상 한 쌍으로 이동. leakage-audit.
- CLAUDE.md §10.7: 카테고리 rotation. 직전 2 H 차단 (target_attention).
- CLAUDE.md §17.3: binary success Δ ≥ +0.5pt.
- HF README (`TAAC2026/data_sample_1000`, 2026-04-10 update).
- Code audit: `experiments/H010_ns_to_s_xattn/upload/model.py:1760-1788` —
  현재 user_int / user_dense 분리 처리.

## Inheritance from prior H

- **H010 F-1** (NS-only enrichment safe pattern) → H011 input-stage enrichment
  도 같은 원리 (anchor 의 NS xattn / fusion 입력 텐서 byte-identical).
- **H010 F-3** (entropy 0.81 = highly selective routing) → input embedding
  enrichment 시 NS xattn 의 selective routing 패턴이 어떻게 변하는지 관찰
  (P2 mechanism check 후보).
- **H010 F-4** (anchor 갱신) → control = H010 (Platform 0.8408). H008
  (0.8387) carry-forward.
- **H009 F-1** (interference 위치 가설) → H011 통합 위치 = input embedding
  lookup, anchor downstream 텐서 byte-identical → 충돌 위험 0.
- **H008 F-1** (interaction axis mechanism class 작동) → H011 = interaction
  axis 의 새 form (input-stage explicit cross).
- **§4.8** mandate → H011 첫 단계 = audit (현재 baseline 룰 위반 여부 확인).
