# H031 — Problem (item_int_feats_13 explicit head + small-vocab user_int outer-product cross)

## Background

이번 세션 (2026-05-04) 직접 EDA 측정 (`data/demo_1000.parquet`, n=1000, positive=124 → 12.4%):

| 신호 | 값 | 비고 |
|---|---|---|
| `item_int_feats_13` 단일 GBDT 5-fold AUC | **0.6561** | vs=10 (schema), nunique=9 (sample) |
| 13 item_int 합쳐서 GBDT 5-fold AUC | 0.7039 | head of importance |
| 모든 66 scalar GBDT 5-fold AUC | 0.6462 | user_int 가 noise 추가 |

**item_13 per-value lift**:

| value | count | label rate | lift vs base 0.124 |
|---|---|---|---|
| **4** | 25 (2.5%) | **0.640** | **+0.516** |
| 7 | 44 (4.4%) | 0.364 | +0.240 |
| 8 | 202 (20.2%) | 0.173 | +0.049 |
| 2 | 460 (46.0%) | 0.072 | −0.052 |
| 6 | 235 (23.5%) | 0.077 | −0.047 |

→ vs=10 짜리 단일 categorical feature 가 7% items 에서 30~64% conversion 만들어냄. 9 H 모두 `RankMixerNSTokenizer` 가 14 item_int feature 를 **하나의 group embedding 으로 합산** (`competition/model.py:1070-1190`) → item_13 의 strong cell signal 이 12 weak cell 과 평균화되어 dilute.

**item_13 의 hot category {4,7,8} 와 differential 한 user_int features** (hot vs cold 분류 univariate AUC 기준):

| user_int | schema_vs | sample_nunique | hot_AUC |
|---|---|---|---|
| user_int_1 | 6 | 3 | 0.5617 |
| user_int_97 | 5 | 4 | 0.5552 |
| user_int_58 | 4 | 3 | 0.4546 |
| user_int_49 | 4 | 3 | 0.5425 |
| user_int_95 | 5 | 4 | 0.5346 |

→ 5개 모두 vs ≤ 6 → **explicit outer-product cross 비용 작음** (5 × 6 × d_cross ≈ 600 cells).

## Why now

F-G (val 0.832~0.836 ceiling 9 H 누적): mechanism class 변경 (input-stage / MoE / hyperparam / recency / DCN-V2 cap / OneTrans backbone / per-user) 무관. **9 H 전부 fid-level explicit cross 안 함** (모두 NS-token block-level). 가장 강한 univariate signal (item_13 0.66) 이 block embedding 안 묻혀 있을 가능성 직접 확인 가치.

## Falsifiable claim

> item_int_feats_13 을 14-feat block 에서 분리해 dedicated 32-dim embedding + user_int{1,97,58,49,95} 와 outer-product cross → DCN-V2 입력 1 NS-token 추가 시 platform AUC Δ vs control ≥ +0.005pt (strong PASS) 또는 ≥ +0.001pt (measurable).

## Scope

- **In**:
  - item_int_feats_13 dedicated embedding (vocab=10 → 32-dim, ~320 params)
  - user_int{1, 97, 58, 49, 95} dedicated embeddings (avg vs=4.8, 16-dim each → ~384 params)
  - outer-product cross: item_13_emb ⊗ each user_int_emb → 5 cross matrices → mean-pool → 1 NS-token (32-dim)
  - 추가 NS-token concat 으로 DCN-V2 입력에 합류 (`PCVRHyFormer` forward path)
- **Out**:
  - 다른 item_int feature 분리 (item_9 AUC 0.55, item_16 0.53 — 2순위 후보, 본 H 검증 후 stack)
  - high-cardinality user_int (user_int_54 vs=2845 등) — 비용 크고 H 결과 후 결정
  - timestamp / recency input — H032 별도

## UNI-REC axes

- **Sequential**: 변경 없음 (H010 backbone 그대로 — TransformerEncoder + MultiSeqHyFormer + RankMixer NS tokenizer + DCN-V2)
- **Interaction**: **fid-level explicit cross 신규 추가** — block-level (H008 DCN-V2) 위에 fid-level 추가
- **Bridging**: 추가 NS-token 이 DCN-V2 입력 합류 → seq encoder 출력 (NS-tokens) 과 같은 cross block 공유 (P1 룰 ✅)

## Success / Failure conditions

- **Success (strong)**: Δ vs control ≥ +0.005pt → fid-level cross 추출 mechanism 작동, 9 H 의 0.84 ceiling 이 dilute 였음 confirm.
- **Success (measurable)**: Δ ∈ [+0.001, +0.005pt] → 작동 marginal, sub-H (다른 user_int 추가, item_9/16 분리) 정당화.
- **Noise**: Δ ∈ (−0.001, +0.001pt) → block-level NS tokenizer 가 이미 item_13 implicit 학습 → fid-level cross redundant.
- **Failure (degraded)**: Δ < −0.001pt → 추가 NS-token 이 DCN-V2 noise 증폭 → mechanism class retire 검토.

## Frozen facts referenced

- CLAUDE.md §3.1 (item_int scalar fid 목록 `{5–10, 12–13, 16, 81, 83–85}`, vs 별도)
- CLAUDE.md §3.4 (label_type=2 12.4% conversion rate)
- 본 세션 EDA (item_13 GBDT 5-fold 0.6561, 본 problem.md 표)
- §3 검증 chain: schema.json `item_int` (fid=13: vs=10, dim=1)

## Inheritance from prior H

- F-G (val ceiling 9 H) → mechanism axis 다른 측면 시도
- F-1 (H011, input-stage REFUTED) → input-stage modify 회피, **NS-token 추가 단계에서 cross**
- H008 F-1 (DCN-V2 block-level fusion PASS, +0.0035pt) → DCN-V2 입력에 추가 token 합류 정당화
- H010 carry-forward (NS-token enrichment safe stacking pattern, anchor 입력 byte-identical) → H031 도 같은 패턴 (기존 NS-token 변경 없이 1개 추가)
