# H014 — Literature References

## Primary references

- **§3.5 데이터 사실** (verified, sibling cite from `tencent-cc/eda/out/semantics.json:per_feature_seq_length`)
  — domain p50/p90/max/frac_empty per domain. **본 H 의 가장 강한 motivation
  source** (모델링 paper 가 아닌 데이터 자체).
- **TWIN** (Pan et al. RecSys 2024 / KDD 2024, "Two-stage Interest Network
  for Lifelong User Behavior Modeling in CTR Prediction") — Tencent paper.
  Long-seq lifelong behavior modeling 의 canonical. **H014 PASS 시 H015
  후보**.
- **SIM** (Pi et al. CIKM 2020, "Search-based User Interest Modeling with
  Lifelong Sequential Behavior Data") — Alibaba paper. target-aware top-K
  retrieval.
- **HSTU** (Zhai et al. arXiv 2024, "Actions Speak Louder than Words: Trillion-
  Parameter Sequential Transducers for Generative Recommendations") — Meta
  paper. long-seq generative trunk.

## Secondary references

- [papers/unified_backbones/onetrans_tencent.md](../../papers/unified_backbones/onetrans_tencent.md)
  — H010 anchor 의 source. NS xattn 의 K=V S concat length 가 본 H 에서
  4× 증가.
- [papers/unified_backbones/pcvrhyformer_baseline.md](../../papers/unified_backbones/pcvrhyformer_baseline.md)
  — organizer baseline 의 truncate envelope 출처.
- ETA (Chen et al. SIGIR 2022, "Efficient Long Sequential User Data Modeling
  for Click-Through Rate Prediction") — long-seq efficient retrieval.
- Sun et al. 2019 (BERT4Rec) — bidirectional attention on full seq.
- Linformer (Wang et al. 2020) — linear attention for long seq.

## Counter-evidence references (Frame B / C 근거)

- **H011 verdict.md F-5 + H012 F-3 + H013 F-2** — 8 H 의 OOF-Platform gap
  1.88~2.42pt 일관 패턴. cohort drift hard ceiling 가설 (Frame B).
- **H010 verdict.md F-3** — NS xattn entropy 0.81 sparse routing. mechanism
  ceiling 가설 (단 envelope expansion 시 routing 풍부해질 가능성).
- Keskar et al. 2017 (large-batch generalization gap) — batch 2048 + seq
  512 = 더 큰 effective batch volume → generalization 어려움 가능.
- Vaswani et al. 2017 (Attention Is All You Need) — dense self-attention
  의 O(L²) compute 한계. long-seq 효율성 = retrieval 또는 sparse attention.

## Audit references

- **§3.5** (CLAUDE.md) — domain seq length 분포 verified.
- **`tencent-cc/eda/out/semantics.json`** — `per_feature_seq_length` 원본
  source (sibling).
- **H013 verdict.md** — Frame A REFUTED, hyperparameter calibration retire.
- **H012 verdict.md F-3** — 4-layer ceiling diagnosis 명시.
- **`experiments/INDEX.md`** — 8 H 누적 ceiling table.

## Memory / OOM references

- attention O(L²) 표준 분석.
- Taiji GPU memory 미공개 — 사용자 spec 확인 필요. likely V100 16GB or
  A100 40GB.
- batch 2048 × seq 512 × d_model 64 × 2 hyformer blocks × 4 heads = ...
  rough estimate ~10-20GB activation. medium risk.
