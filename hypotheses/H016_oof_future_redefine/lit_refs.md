# H016 — Literature References

## Primary references

- [papers/temporal_cohort/_summary.md](../../papers/temporal_cohort/_summary.md)
  — H015 first-touch + H016 sub-form.
- **Pan & Yang 2010** — transfer learning survey.
- **Sugiyama et al. 2007** — covariate shift importance weighting + measurement.
- **Production CTR engineering** — time-based holdout (train < cutoff <
  valid < oof) standard practice.

## Secondary references

- [hypotheses/H015_recency_loss_weighting/](../../hypotheses/H015_recency_loss_weighting/)
  — paired sibling H, train-side cohort drift attack. H016 = OOF-side.
- DLRM-v2 (production CTR, time-based eval split standard).
- Concept drift adaptation surveys (Gama 2014).

## Counter-evidence references (Frame B / C)

- **9 H verdicts (H006~H014)**: OOF stable / Platform 변동 패턴이 단순
  measurement 문제가 아닌 진짜 distribution shift 일 가능성.
- **H011 F-5 + H012 F-3 + H013 F-4 + H014 F-2/F-3**: cohort drift hard ceiling
  가설 누적.
- Backward compat 우려: H016 의 OOF 새 정의가 prior H 비교 base 깨짐 (Platform
  비교만 valid).

## Audit references

- **`competition/dataset.py:split_parquet_by_label_time`** — H010 baseline
  split 코드 (random_user 기본).
- **`tencent-cc/eda/out/semantics.json:label_minus_event_seconds`** — label-event
  gap 분포 (sibling cite, gap #1 unresolved). H016 결과 후 fundamental 시
  본 프로젝트 카피.
- **CLAUDE.md §3.4** — label_time / timestamp 사실.
- **CLAUDE.md §17.2** — single mutation (split definition only).
