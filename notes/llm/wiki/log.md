# LLM Wiki Log

> chronological, append-only log.

형식:

```md
## [YYYY-MM-DD HH:mm KST] ingest|query|lint|promote | short-title
- actor: user|assistant
- touched:
  - notes/llm/wiki/...
  - notes/llm/raw/...
- summary: 한 줄 요약
- promotion: none|to-papers|to-notes-refs|to-hypotheses|to-experiments
```

---

## [2026-05-01 00:00 KST] bootstrap | llm-wiki overlay scaffold
- actor: assistant
- touched:
  - notes/llm/README.md
  - notes/llm/authority.md
  - notes/llm/wiki/index.md
  - notes/llm/wiki/log.md
  - notes/llm/wiki/overview.md
  - notes/llm/wiki/page_template.md
- summary: LLM wiki overlay 기본 구조와 권한 규칙 추가
- promotion: none
- 2026-05-01 15:57 KST | edit | `hypotheses/H011_aligned_pair_encoding/transfer.md` | [[by_hypothesis/H011_aligned_pair_encoding]]
- 2026-05-01 15:57 KST | edit | `hypotheses/H011_aligned_pair_encoding/transfer.md` | [[by_hypothesis/H011_aligned_pair_encoding]]
- 2026-05-01 15:57 KST | edit | `hypotheses/H011_aligned_pair_encoding/predictions.md` | [[by_hypothesis/H011_aligned_pair_encoding]]
- 2026-05-01 15:58 KST | edit | `progress.txt` | [[by_governance/progress]]
- 2026-05-01 15:58 KST | write | `eda/out/dense_value_stats.json` | [[by_eda/dense_value_stats]]
- 2026-05-01 15:58 KST | edit | `hypotheses/H011_aligned_pair_encoding/transfer.md` | [[by_hypothesis/H011_aligned_pair_encoding]]
- 2026-05-01 15:58 KST | edit | `hypotheses/H011_aligned_pair_encoding/predictions.md` | [[by_hypothesis/H011_aligned_pair_encoding]]
- 2026-05-01 15:58 KST | edit | `hypotheses/H011_aligned_pair_encoding/verdict.md` | [[by_hypothesis/H011_aligned_pair_encoding]]
- 2026-05-01 15:59 KST | edit | `hypotheses/H011_aligned_pair_encoding/transfer.md` | [[by_hypothesis/H011_aligned_pair_encoding]]
- 2026-05-01 16:33 KST | edit | `hypotheses/H011_aligned_pair_encoding/transfer.md` | [[by_hypothesis/H011_aligned_pair_encoding]]
- 2026-05-01 16:33 KST | edit | `hypotheses/H011_aligned_pair_encoding/predictions.md` | [[by_hypothesis/H011_aligned_pair_encoding]]

## [2026-05-02 00:00 KST] query | sample1000 eda backlog bootstrap
- actor: assistant
- touched:
  - notes/llm/wiki/by_eda/eda_backlog_sample1000.md
  - notes/llm/wiki/index.md
  - CLAUDE.md
- summary: sample1000용 EDA 우선순위 백로그(P0/P1/P2) 추가 및 참조 경로 연결
- promotion: none
- 2026-05-03 14:29 KST | edit | `notes/refs/inference_lessons.md` | [[by_governance/inference_lessons]]
- 2026-05-03 14:29 KST | edit | `CLAUDE.md` | [[by_governance/CLAUDE]]
- 2026-05-03 14:35 KST | edit | `notes/refs/inference_lessons.md` | [[by_governance/inference_lessons]]
- 2026-05-03 15:20 KST | write | `.omc/prd.json` | (unmapped)
- 2026-05-03 15:21 KST | write | `experiments/H011_aligned_pair_encoding/training_request_remeasure.md` | [[by_experiment/H011_aligned_pair_encoding]]
- 2026-05-03 15:21 KST | write | `experiments/H012_multi_domain_fusion/training_request_remeasure.md` | [[by_experiment/H012_multi_domain_fusion]]
- 2026-05-03 15:22 KST | write | `experiments/H013_hyperparameter_calibration/training_request_remeasure.md` | [[by_experiment/H013_hyperparameter_calibration]]
- 2026-05-03 15:22 KST | edit | `experiments/_TEMPLATE/training_result.md` | (unmapped)
- 2026-05-03 15:22 KST | edit | `experiments/_TEMPLATE/training_result.md` | (unmapped)
- 2026-05-03 15:24 KST | write | `hypotheses/H018_per_user_recency_weighting/problem.md` | [[by_hypothesis/H018_per_user_recency_weighting]]
- 2026-05-03 15:25 KST | write | `hypotheses/H018_per_user_recency_weighting/challengers.md` | [[by_hypothesis/H018_per_user_recency_weighting]]
- 2026-05-03 15:26 KST | write | `hypotheses/H018_per_user_recency_weighting/transfer.md` | [[by_hypothesis/H018_per_user_recency_weighting]]
- 2026-05-03 15:26 KST | write | `hypotheses/H018_per_user_recency_weighting/predictions.md` | [[by_hypothesis/H018_per_user_recency_weighting]]
- 2026-05-03 15:27 KST | write | `hypotheses/H018_per_user_recency_weighting/lit_refs.md` | [[by_hypothesis/H018_per_user_recency_weighting]]
- 2026-05-03 15:27 KST | write | `hypotheses/H018_per_user_recency_weighting/verdict.md` | [[by_hypothesis/H018_per_user_recency_weighting]]
- 2026-05-03 15:29 KST | write | `experiments/H018_per_user_recency_weighting/upload_patch.md` | [[by_experiment/H018_per_user_recency_weighting]]
- 2026-05-03 15:29 KST | write | `experiments/H018_per_user_recency_weighting/card.yaml` | [[by_experiment/H018_per_user_recency_weighting]]
- 2026-05-03 15:37 KST | edit | `experiments/INDEX.md` | [[by_experiment/INDEX]]
- 2026-05-03 15:37 KST | edit | `experiments/INDEX.md` | [[by_experiment/INDEX]]
- 2026-05-03 15:37 KST | edit | `hypotheses/INDEX.md` | [[by_hypothesis/INDEX]]
- 2026-05-03 15:38 KST | edit | `hypotheses/INDEX.md` | [[by_hypothesis/INDEX]]
- 2026-05-03 15:39 KST | write | `.omc/prd.json` | (unmapped)
- 2026-05-03 16:03 KST | write | `.omc/prd.json` | (unmapped)
- 2026-05-03 16:04 KST | edit | `experiments/H011_aligned_pair_encoding/training_result.md` | [[by_experiment/H011_aligned_pair_encoding]]
- 2026-05-03 16:04 KST | edit | `experiments/H012_multi_domain_fusion/training_result.md` | [[by_experiment/H012_multi_domain_fusion]]
- 2026-05-03 16:05 KST | write | `experiments/H013_hyperparameter_calibration/training_result.md` | [[by_experiment/H013_hyperparameter_calibration]]
- 2026-05-03 16:05 KST | edit | `experiments/INDEX.md` | [[by_experiment/INDEX]]
- 2026-05-03 16:05 KST | edit | `experiments/INDEX.md` | [[by_experiment/INDEX]]
- 2026-05-03 16:05 KST | edit | `hypotheses/INDEX.md` | [[by_hypothesis/INDEX]]
- 2026-05-03 16:05 KST | edit | `hypotheses/INDEX.md` | [[by_hypothesis/INDEX]]
- 2026-05-03 16:06 KST | edit | `hypotheses/H018_per_user_recency_weighting/predictions.md` | [[by_hypothesis/H018_per_user_recency_weighting]]
- 2026-05-03 16:07 KST | write | `hypotheses/H019_twin_long_seq_retrieval/problem.md` | [[by_hypothesis/H019_twin_long_seq_retrieval]]
- 2026-05-03 16:08 KST | write | `hypotheses/H019_twin_long_seq_retrieval/challengers.md` | [[by_hypothesis/H019_twin_long_seq_retrieval]]
- 2026-05-03 16:08 KST | write | `hypotheses/H019_twin_long_seq_retrieval/transfer.md` | [[by_hypothesis/H019_twin_long_seq_retrieval]]
- 2026-05-03 16:09 KST | write | `hypotheses/H019_twin_long_seq_retrieval/predictions.md` | [[by_hypothesis/H019_twin_long_seq_retrieval]]
- 2026-05-03 16:09 KST | write | `hypotheses/H019_twin_long_seq_retrieval/lit_refs.md` | [[by_hypothesis/H019_twin_long_seq_retrieval]]
- 2026-05-03 16:10 KST | write | `hypotheses/H019_twin_long_seq_retrieval/verdict.md` | [[by_hypothesis/H019_twin_long_seq_retrieval]]
- 2026-05-03 16:10 KST | write | `experiments/H019_twin_long_seq_retrieval/card.yaml` | [[by_experiment/H019_twin_long_seq_retrieval]]
- 2026-05-03 16:11 KST | edit | `hypotheses/INDEX.md` | [[by_hypothesis/INDEX]]
- 2026-05-03 16:11 KST | edit | `experiments/INDEX.md` | [[by_experiment/INDEX]]
- 2026-05-03 16:14 KST | edit | `experiments/INDEX.md` | [[by_experiment/INDEX]]
- 2026-05-03 16:32 KST | write | `hypotheses/H022_h010_multi_seed_variance/problem.md` | [[by_hypothesis/H022_h010_multi_seed_variance]]
- 2026-05-03 16:33 KST | write | `hypotheses/H022_h010_multi_seed_variance/challengers.md` | [[by_hypothesis/H022_h010_multi_seed_variance]]
- 2026-05-03 16:34 KST | write | `hypotheses/H022_h010_multi_seed_variance/transfer.md` | [[by_hypothesis/H022_h010_multi_seed_variance]]
- 2026-05-03 16:34 KST | write | `hypotheses/H022_h010_multi_seed_variance/predictions.md` | [[by_hypothesis/H022_h010_multi_seed_variance]]
- 2026-05-03 16:35 KST | write | `hypotheses/H022_h010_multi_seed_variance/lit_refs.md` | [[by_hypothesis/H022_h010_multi_seed_variance]]
- 2026-05-03 16:35 KST | write | `hypotheses/H022_h010_multi_seed_variance/verdict.md` | [[by_hypothesis/H022_h010_multi_seed_variance]]
- 2026-05-03 16:36 KST | write | `experiments/H022_h010_multi_seed_variance/card.yaml` | [[by_experiment/H022_h010_multi_seed_variance]]
- 2026-05-03 16:36 KST | write | `experiments/H019_twin_long_seq_retrieval/upload_patch.md` | [[by_experiment/H019_twin_long_seq_retrieval]]
