# Papers — 카테고리별 장기기억 (former §8)

> **트리거**: literature-scout 호출 / 새 외부 논문 수집 / `papers/` 디렉토리 작업 시 본 파일 먼저 읽기.

`papers/README.md`가 라우팅 맵. 카테고리 디렉토리마다 `_summary.md`는 **3–5줄 takeaway + 상세파일 목록**.

## 검색 루틴 (literature-scout)

1. 먼저 `papers/README.md` + 관련 카테고리 `_summary.md` 읽기.
2. 이미 덮이는 주제면 새 검색 금지, 기존 요약 재활용.
3. 새 검색은 WebSearch + arXiv WebFetch.
4. 신규 항목은 `{category}/{source_slug}.md`로: claim / method / what-it-guarantees / applicability-to-ours / quote / link.
5. `_summary.md`에 한 줄 takeaway 추가.

## 카테고리

- `unified_backbones/` — **OneTrans, InterFormer, PCVRHyFormer** (P1 method-transfer 1순위).
- `long_seq_retrieval/` — SIM, ETA, TWIN, HSTU (P2).
- `multi_domain_fusion/` — MMoE, PLE, STAR, MiNet.
- `semantic_id/` — TIGER, OnePiece, RQ-VAE (P3).
- `target_attention/` — DIN, DIEN, DSIN.
- `sparse_feature_cross/` — DCN-V2, CAN, FwFM, AutoDis.
- `loss_calibration/` — Focal, class-balanced, calibrated CTR.
- `external_inspirations/` — 타 도메인 아이디어 (`method-transfer` 1순위 재료).
