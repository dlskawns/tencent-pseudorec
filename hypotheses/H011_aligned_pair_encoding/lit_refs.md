# H011 — Literature References

## Primary references

- [papers/feature_engineering/dlrm_naumov2019.md](../../papers/feature_engineering/dlrm_naumov2019.md)
  — sparse-dense input fusion at embedding stage. canonical pattern.
- [papers/feature_engineering/fwfm_pan2018.md](../../papers/feature_engineering/fwfm_pan2018.md)
  — field-weighted FM, weight semantics 정수.

## Secondary references

- [papers/unified_backbones/pcvrhyformer_baseline.md](../../papers/unified_backbones/pcvrhyformer_baseline.md)
  — organizer baseline. 현재 user_int / user_dense 분리 처리 (코드 audit
  결과 model.py:1760-1788).
- DIN (Zhou et al. KDD 2018, arXiv:1706.06978) — element-wise weighted
  history attention. 본 H 는 hard alignment 의 단순 form.
- Wide & Deep (Cheng et al. RecSys 2016, arXiv:1606.07792) — dense + cross
  feature 동일 tower input.
- AutoDis (Guo et al. CIKM 2021) — automatic discretization of dense features.
  scale handling sub-H 후보.
- DCN-V2 ([papers/sparse_feature_cross/](../../papers/sparse_feature_cross/)
  — H008 anchor, post-stage cross). H011 은 input-stage cross 로 차이.
- OneTrans NS→S xattn (H010 anchor) — NS-token level cross. H011 은 input-stage
  binding 으로 NS-token 입력 표현 enrichment.

## Counter-evidence references (Frame B 근거)

- DCN-V2 paper (Wang et al. WWW 2021) — interaction layer 가 충분히 deep
  하면 implicit binding 학습 가능 주장. NS-level cross-attention (H010)
  이 같은 역할.
- DIN attention 결과 — attention weight 학습 자체가 input weight 와 별개로
  유의미한 lift. soft alignment 이 hard alignment 보다 일반적으로 더 강함
  (sample-scale 한계 인정).

## Audit references (P0 — verification, 새 EDA 아님)

- **`competition/ns_groups.json`** (`_note_shared_fids`, `_note_user_dense`)
  — **검증된 aligned mapping 의 ground truth**. 2026-04-30 검증 완료.
  - shared fids: `{62, 63, 64, 65, 66, 89, 90, 91}` (8).
  - dense-only fids: `{61, 87}`.
  - user_dense total_dim = 918 (10 fids concat).
- **`competition/dataset.py`** — `_user_dense_plan` (lines 246-250, 293-297,
  565-571): per-fid (offset, dim) 매핑 산출 로직. P0 audit Quantity 1 의 source.
- **`demo_1000.parquet`** — per-row array length `n_k` 측정 source (P0
  audit Quantity 2). 새 EDA 가 아닌 verification 측정.
- HF README (`TAAC2026/data_sample_1000`, 2026-04-10) — flat layout 출처.
  단 aligned fid 후보 list 는 ns_groups.json 우선 (CLAUDE.md §4.9 chain).
- `experiments/H010_ns_to_s_xattn/upload/make_schema.py` — schema 생성 로직.
  코드 빌드 시 aligned offset/dim 매핑 보강 reference.
