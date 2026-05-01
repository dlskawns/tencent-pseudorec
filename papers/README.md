# Papers — 카테고리별 장기기억 라우팅 맵

> CLAUDE.md §8 참조. 새 개념 검색 전에 **항상 이 파일과 관련 카테고리 `_summary.md`를 먼저 읽는다.** 이미 덮이는 주제면 새 검색 금지.

## 카테고리

| 디렉토리 | 주제 | UNI-REC axis | Phase 활성 |
|---|---|---|---|
| `unified_backbones/` | seq + interaction을 single backbone에 통합 (OneTrans, InterFormer, PCVRHyFormer) | seq+int | P1+ 1순위 |
| `long_seq_retrieval/` | 긴 시퀀스 retrieval/compression (SIM, ETA, TWIN, HSTU) | seq | P2 |
| `multi_domain_fusion/` | 도메인 간 fusion (MMoE, PLE, STAR, MiNet) | seq | P1 |
| `semantic_id/` | 생성형 추천 토큰화 (TIGER, OnePiece, RQ-VAE) | seq | P3 |
| `target_attention/` | candidate-aware attention (DIN, DIEN, DSIN) | seq | P1 |
| `sparse_feature_cross/` | post-encoder explicit cross at interaction layer (DCN-V2, CAN) | int | P0–P1 |
| `feature_engineering/` | **input-stage** sparse-dense fusion / `<id, weight>` binding (DLRM, FwFM, DIN, AutoDis) | int (upstream) | P0–P1 |
| `loss_calibration/` | 손실/캘리브레이션 (Focal, class-balanced) | -- | P0–P1 |
| `external_inspirations/` | 타 도메인 아이디어 (method-transfer 1순위 재료) | varies | P1+ 의무 주입 |

## literature-scout 루틴

1. 본 파일 + 관련 `_summary.md` 읽기.
2. 이미 다룬 주제 → 기존 entry 재활용.
3. 새 검색은 WebSearch + arXiv WebFetch.
4. 신규 entry는 `{category}/{source_slug}.md`로:
   - **Claim** (한 단락)
   - **Method** (메커니즘 bullet)
   - **What it guarantees** (formal property + empirical claim)
   - **Applicability to ours** (우리 데이터에 어떻게 매핑되나)
   - **Sample-scale risk** (§10.6 위반 여부)
   - **Quote** (paraphrased, ≤ 2 sentences)
   - **Link** (arXiv/venue)
   - **Carry-forward rules** (transfer 시 R1, R2, ... 형태로)
5. `_summary.md`에 한 줄 takeaway 추가.

## 현재 선반영된 항목

- `unified_backbones/_summary.md` + 3 entries (OneTrans, InterFormer, PCVRHyFormer) — tencent-cc/papers에서 카피, 본 프로젝트의 P1 method-transfer 1순위 백본 후보.
- `feature_engineering/_summary.md` + 2 entries (DLRM, FwFM) — H011 cold-start (2026-04-30). input-stage `<id, weight>` binding family. CLAUDE.md §3 / §4.8 mandate 직접 cover. FwFM 1저자 Junwei Pan 은 TAAC 2026 organizer (§0).
