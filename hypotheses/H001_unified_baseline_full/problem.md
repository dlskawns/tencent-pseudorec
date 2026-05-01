# H001 — Unified-block baseline anchor (E000)

## What we're trying to explain
사용자가 명시적으로 "unified block에서 baseline을 먼저 테스트"하길 요청. 즉 organizer가 제공한 `PCVRHyFormer` (per-domain seq encoders → per-domain query decoders → joint token fusion via RankMixer) 가 우리 데이터에서 실제로 학습 가능한지, 그리고 §13 contract을 통과하는 end-to-end submission을 생성할 수 있는지를 측정한다. 이 H는 **최적화 목적이 아니라 anchor 생성**이 목적이다 — 모든 후속 H는 본 anchor의 OOF AUC 대비 paired Δ로 평가된다.

## Why now
- iter-00에서 baseline 코드 audit 완료. 6개 결함 (A=row-group split, B=OOF 부재, C=path None crash, D=schema.json 미존재, E=infer 미구현, F=local-minima 정책) 식별. **A, B, C, D는 baseline이 demo_1000에서 한 줄도 안 도는 hard blocker.**
- 사용자가 "원래의 실험방식이 너무 로컬미니마에 빠지는 경향" 라고 명시 회고. 새 방식의 1번 원칙 = baseline-first cloud anchor.
- iter-00의 모든 후속 가설 (H002–H005)이 본 H의 OOF AUC를 control로 가짐. 이 anchor 없이는 효과 크기 측정 불가.

## Scope
- In:
  - 결함 A/B/C/D 패치 (label_time-aware split + 10% user OOF + sane path defaults + auto schema.json).
  - Organizer baseline AS-IS (RankMixer NS tokenizer, num_queries=2, transformer seq encoder, BCE loss): zero hyperparameter mutation.
  - Sample-scale (demo_1000.parquet, 1000 rows) 학습 1회, full-data로 옮길 때 재실행.
  - 결함 E 패치: `submission/infer.py`가 학습된 ckpt를 로드해 §13 round-trip 1회 통과.
- Out:
  - 모델 architecture mutation (다음 H로).
  - Loss / focal / lr / schedule 튜닝 (CLAUDE.md §17.2 — P1+ 까지 금지).
  - num_workers, batch_size 등 throughput 튜닝 (별도 cloud H).

## UNI-REC axes
- Sequential: per-domain `TransformerEncoder` (model.py:544) — 4 도메인 a/b/c/d 각각 self-attention + FFN.
- Interaction: NS tokens (RankMixerNSTokenizer, model.py:1070) → `RankMixerBlock` (model.py:315) 안에서 token-mixing.
- Bridging mechanism: `MultiSeqHyFormerBlock` (model.py:850) — decoded queries + NS tokens가 같은 RankMixerBlock 통과 → token fusion 단계에서 gradient 공유. **✓ §0 P1 정의 ("seq + interaction이 한 블록에서 gradient 공유") 충족.**

## Success / Failure conditions
- **Success (anchor 자격)**: (i) train.py이 1 epoch 끝까지 NaN 없이 완주. (ii) OOF AUC ≥ 0.5 (random보다 나쁜 결과 아님). (iii) `infer.py`가 `local_validate.py` 5/5 통과 + 해당 ckpt를 로드한 결과가 prior fallback과 다른 분포 (즉 모델이 실제로 사용됨을 검증). (iv) `metrics.json`에 `{seed, git_sha, config_sha256, host, compute_tier, oof_user_ids, label_time_cutoff}` 필드 모두 채워짐.
- **Failure**: 위 4개 중 하나라도 미달 → REFUTED. 결함 분류 (A–F 중 어디 영향) + carry-forward.

## Frozen facts referenced
- `data/demo_1000.parquet` 1000 rows × 120 cols, label_type 1:2 = 876:124 (prior=0.124).
- label_time 범위 [1772725027, 1772725910] = 13분 윈도우.
- 단일 row group → organizer의 row-group split에서 train_rgs=0 (즉 학습 데이터 없음). **결함 A patch 없이는 실험 불가.**
- `papers/unified_backbones/pcvrhyformer_baseline.md` — baseline 구조 audit 결과.

## Inheritance from prior H
없음 (iter-00 anchor — 첫 가설).
