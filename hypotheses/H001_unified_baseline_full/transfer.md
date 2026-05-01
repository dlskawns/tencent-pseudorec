# H001 — Method Transfer

## ① Source
`papers/unified_backbones/pcvrhyformer_baseline.md` (organizer baseline) — 코드는 `competition/{model.py, dataset.py, trainer.py, train.py}`.

## ② Original mechanism
PCVRHyFormer는 4-stage pipeline: (1) per-domain sequence encoder (transformer/swiglu/longer 중 택일) 가 각 도메인의 토큰 시퀀스를 self-attention으로 처리, (2) per-domain `MultiSeqQueryGenerator` 가 NS-tokens + seq tokens 를 cross-attention으로 압축해 Nq 개 query token 생성, (3) `MultiSeqHyFormerBlock` 안에서 decoded queries + NS tokens 를 한 RankMixerBlock 으로 token-mix (= 두 축이 같은 블록에서 gradient 공유), (4) 최종 query 토큰을 concat해 classifier head로. 학습은 dual optimizer (Adagrad sparse / AdamW dense), focal-or-BCE loss.

## ③ What we adopt
- 전체 architecture (1)-(4) 그대로.
- Hyperparameters: `--ns_tokenizer_type rankmixer --user_ns_tokens 5 --item_ns_tokens 2 --num_queries 2 --emb_skip_threshold 1000000` (organizer run.sh 디폴트).
- Dual optimizer + Pre-LN + focal_alpha=0.1, gamma=2.0 디폴트 (BCE도 옵션).

## ④ What we modify (NOT a clone)
- **결함 A 패치**: `dataset.py`의 row-group 순서 split을 `label_time` 기준 split으로 교체. 신규 함수 `split_parquet_by_label_time` + `get_pcvr_data_v2`.
- **결함 B 패치**: 10% user_id OOF 홀드아웃 추가 (seed=42 고정). train/valid/oof 3개 분리 parquet 생성 → 동일 PCVRParquetDataset으로 read.
- **결함 C 패치**: `train.py`의 path 디폴트 fallback (None이면 `${ROOT}/experiments/E000_unified_baseline_demo/{ckpt,logs,tf_events,work}`).
- **결함 D 패치**: `competition/make_schema.py` 신규 — parquet 메타에서 `schema.json` 자동 생성 (vocab=max+2, dim=max_len capped 1024, ts_fid via unix-timestamp range heuristic).
- **결함 E 패치**: `submission/infer.py` 가 PCVRHyFormer ckpt 로드 + 실패 시 prior fallback (계약 위반 없이 graceful degradation).
- **재현성 메타**: `metrics.json` 에 `{seed, git_sha, config_sha256, host, compute_tier, oof_user_ids, label_time_cutoff, best_val_AUC, best_oof_AUC}` 명시 — organizer baseline엔 없음.
- 모델 자체는 mutation 없음 (CLAUDE.md §17.2 — one-mutation-per-experiment, baseline은 zero-mutation).

## ⑤ UNI-REC alignment
- **Sequential reference**: SASRec / DIN — `TransformerEncoder` (model.py:544) 가 self-attention + FFN, time bucket embedding 추가.
- **Interaction reference**: DCN / FwFM — `RankMixerNSTokenizer` (model.py:1070) 가 cat-and-split chunking으로 NS feature를 N개 토큰으로 분할 후 token-mix.
- **Bridging mechanism**: `MultiSeqHyFormerBlock` (model.py:850) — decoded queries + NS tokens 가 같은 `RankMixerBlock` 의 token-mixing 단계 통과 → 두 축이 한 블록 안에서 gradient 공유. P1 진입 조건 (CLAUDE.md §0) 충족.
- **primary_category**: `unified_backbones`
- **Innovation axis**: Anchor measurement only — 본 H는 baseline 자체에 혁신축을 두지 않고, **결함 A–E 패치가 baseline을 demo_1000에서 동작 가능하게 만든다**는 운영적 혁신만 주장.

## ⑥ Sample-scale viability (Rule UB-1, §10.6)
- 예상 trainable params: ~250k–500k (d_model=64, num_hyformer_blocks=2, 4 domains × seq encoders + 2 NS tokenizers + classifier).
- §10.6 soft cap (≤ 2146) 100x+ 초과. 그러나 본 H의 목적은 **anchor 측정**이므로 cap 면제. demo_1000에서 학습 결과의 generalization은 주장 안 함 (`claim_scope: "demo-only"`).
- Full-data 도착 시 동일 config로 재실행 → E000.full.

## ⑦ Carry-forward rules to honor
- §10.5 LayerNorm on x0: PCVRHyFormer 는 `ln_mode='pre'` (model.py:298) — 자동 충족.
- §10.10 InterFormer bridge gating σ(−2): 본 H는 새 bridge 추가 없음 — 미적용.
- §10.9 OneTrans softmax-attention entropy abort: TransformerEncoder의 self-attention 1000-row에서 uniform collapse 위험. **diagnostic 추가**: 1 epoch 후 random batch에서 attention probability mean entropy 측정해 `metrics.json` 에 `attn_entropy_per_layer` 필드로 기록. 만약 ≥ 0.95·log(L) 이면 verdict.md에 plus-flag.
- §17.2 one-mutation-per-experiment: 본 H는 zero-mutation anchor.
- §17.5 sample-scale = code-path verification only: 본 H의 OOF AUC는 anchor 자격 검증 용도이지 leaderboard 점수 예측 아님.
