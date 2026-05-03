# H015 — Literature References

## Primary references

- [papers/temporal_cohort/_summary.md](../../papers/temporal_cohort/_summary.md)
  — 신규 카테고리 cold-start, recency weighting / temporal embedding /
  domain adaptation family.
- **Pan & Yang 2010** ("A Survey on Transfer Learning") — sample weighting
  in domain adaptation 의 canonical reference.
- **Gama et al. 2014** ("A Survey on Concept Drift Adaptation") — online
  learning 의 recency-based weighting + concept drift handling.
- **Sugiyama et al. 2007** (arXiv:0710.1394, "Covariate Shift Adaptation
  by Importance Weighted Cross Validation") — importance weighting 의 이론적
  근거.

## Secondary references

- **Production CTR engineering** (Meta / Google / Tencent): time-decay loss
  weighting 이 standard practice (논문 부재, production know-how).
- DLRM (Naumov et al. 2019, arXiv:1906.00091) — temporal weighting 미적용
  하지만 production CTR 의 sample handling 표준.
- **CLAUDE.md §0** — Tencent UNI-REC challenge, organizer = Junwei Pan
  (Tencent Ads). production CTR cohort handling 표준 반영 가능성.

## Counter-evidence references (Frame B / C 근거)

- **9 H verdicts (H006~H014)** — OOF 0.857~0.860 stable / Platform 0.82~0.838
  변동 일관 패턴. cohort drift 가 mechanism 변경 대비 dominant variance
  source.
- **H011 F-5, H012 F-3, H013 F-4, H014 F-2/F-3** — cohort drift hard ceiling
  가설 누적 carry-forward.
- Keskar et al. 2017 (arXiv:1609.04836) — large-batch generalization gap.
  H013 F-2 적용. 단 H015 mean weight = 1.0 보존 → batch effect 영향 없음.
- **Frame C 후보**: OOF 재정의 (label_time future-only holdout) 가 더
  fundamental 일 가능성. H015 = loss weighting 이 더 conservative.

## Audit references

- **§3.4** (CLAUDE.md) — label_time / timestamp / label_minus_event_seconds
  관련 사실. H015 의 label_time 사용 정합성.
- **`tencent-cc/eda/out/semantics.json:label_minus_event_seconds`** — gap
  분포 (sibling 미카피, gap #1 unresolved). H015 결과 후 cohort drift
  fundamental 시 본 프로젝트 카피.
- **`experiments/INDEX.md`** — 9 H 누적 cost ~32h, OOF / Platform 표.
- **`competition/dataset.py:split_parquet_by_label_time`** — label_time
  기반 split 의 ground truth 코드.
- **CLAUDE.md §17.2** — single mutation rule. H015 적용 (loss weighting
  only).
- **CLAUDE.md §17.3** — binary success threshold.
