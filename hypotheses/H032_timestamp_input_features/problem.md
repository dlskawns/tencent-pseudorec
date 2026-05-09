# H032 — Problem (timestamp/recency input features — never tried as model input)

## Background

이번 세션 (2026-05-04) 직접 코드 audit:

- `competition/dataset.py:510` — `timestamps` arrow column → result['timestamp'] 로 batch dict 에 포함 ✅
- `competition/model.py` 전체 (1714 lines) — **`timestamp`/`label_time` 키워드 zero hits**. `ModelInput` NamedTuple 에 timestamp field 없음 (line 11-15: user_int_feats, item_int_feats, user_dense_feats, item_dense_feats, sequence dict 만).
- timestamp 의 유일한 사용: `_convert_batch` line 637-648 의 sequence 내부 time bucketing (relative position 인코딩) — 절대 시간 / per-sample feature 로는 미사용.

→ **9 H 전부 timestamp 를 input feature 로 사용 안 함**. H015~H018 의 recency 시도는 모두 _loss reweighting_ axis (per-batch / per-user weight 곱셈), input feature 가 아님.

## EDA signal (이번 세션 직접 측정, `data/demo_1000.parquet` n=1000)

| Signal | univariate AUC | 비고 |
|---|---|---|
| `label_time - timestamp` | **0.6049** | **strong**, 단 inference 시 label_time null 가능 → leakage risk, 직접 사용 불가 |
| `timestamp - max(seq_ts)` (recency) | 0.5201 | weak alone |
| `log1p(domain_a_seq_len)` | 0.5276 | per-domain seq length 약 신호 |
| `log1p(domain_b_seq_len)` | 0.5175 | |
| `log1p(domain_c_seq_len)` | 0.5356 | |
| `log1p(domain_d_seq_len)` | 0.5092 | |

demo_1000 의 timestamp range = 1772725000 ~ 1772725910 (910초 ≈ 15분). **single short window** → hour-of-day / day-of-week 효과 local 검증 불가. **full data trust 필요** (full data 는 §3.5 의 long seq p90 1393~2215 events 로 추정 시 days 단위 span).

## Why now

F-G (val 0.832~0.836 ceiling 9 H 누적): mechanism class 변경 무관. **9 H 의 input feature space 는 user_int + item_int + user_dense + 4 domain seq 만** — temporal context 는 seq 내부 relative position 으로만 학습. CTR 분야 표준 (DIN, DIEN, BST, SIM/TWIN) 에서 absolute time / hour / dow / recency 는 standard input 인데 본 프로젝트는 미사용. **새 signal class 추가**.

H015~H018 차별화:
- H015 = per-batch linear recency loss reweight [0.5, 1.5] → REFUTED (Δ +0.00024pt marginal)
- H016 = OOF future-only redefine → REFUTED model lift (Δ −0.0059)
- H017 = per-batch exp decay → INVALID (submission lost)
- H018 = per-user exp time-decay loss weight (tau=14) → SCAFFOLDED 미실행

→ **모두 loss-axis**. H032 = **input-axis** (model 이 per-sample 시간 정보 직접 학습, loss 가 아닌 prediction logit 에 영향).

## Falsifiable claim

> timestamp 에서 derived 한 3개 categorical features (hour_of_day, day_of_week, recency_log_bucket from `timestamp - max(seq_ts)`) 를 별도 NS-token (16-dim embedding 합산) 으로 추가 시 platform AUC Δ vs control ≥ +0.005pt (strong PASS) 또는 ≥ +0.001pt (measurable).

## Scope

- **In**:
  - `hour_of_day` = `(timestamp // 3600) % 24` → `nn.Embedding(24, 16)`
  - `day_of_week` = `(timestamp // 86400) % 7` → `nn.Embedding(7, 16)`
  - `recency_bucket` = `min(7, log2(max(1, timestamp - max_seq_ts)))` → 8 buckets → `nn.Embedding(8, 16)` (max_seq_ts = max across 4 domains 의 ts column 의 max)
  - 3 embeddings sum → 1 NS-token (16-dim → projected to d_model)
- **Out**:
  - `label_time - timestamp` (leak risk — inference 시 label_time null 가능)
  - per-seq-token timestamp encoding (이미 sequence 내부 time bucketing 으로 처리 중, line 637-648)
  - 다른 derived (week_of_year, time_since_signup) — 본 H 는 minimal 3-feature 시도

## UNI-REC axes

- **Sequential axis**: 변경 없음 (seq encoder 내부 time bucketing 그대로). 본 H 는 per-sample 의 _absolute_ timestamp signal 추가 — sequence 의 _relative_ 시간 인코딩 보완.
- **Interaction axis**: 변경 없음 (DCN-V2 그대로). 단 새 NS-token 이 DCN-V2 입력 합류 → time × user × item cross 가 자연스럽게 학습.
- **Bridging mechanism**: 새 NS-token 이 NS xattn → DCN-V2 같은 trunk 에서 gradient 공유 (P1 룰 ✅).

## Success / Failure conditions

- **Success (strong)**: Δ ≥ +0.005pt → temporal input signal 기존 mechanism 으로 못 잡고 있던 정보 capture confirm.
- **Success (measurable)**: Δ ∈ [+0.001, +0.005pt] → 작동 marginal, sub-H (week_of_year, multi-resolution time bucket) 정당화.
- **Noise**: Δ ∈ (−0.001, +0.001pt) → seq 내부 time bucketing + per-batch optimizer 로 implicit 학습 충분.
- **Failure**: Δ < −0.001pt → temporal feature 가 cohort fit 만 키우고 platform 악화 (H011 패턴).

## Frozen facts referenced

- `competition/model.py` line 1-15 (ModelInput NamedTuple — timestamp field 없음)
- `competition/dataset.py` line 510, 579 (timestamps batch dict 진입)
- `competition/dataset.py` line 637-648 (seq 내부 time bucketing 만 사용)
- 본 세션 EDA (label_time-timestamp 0.6049, recency 0.5201)
- CLAUDE.md §3.4 (label_type=2 12.4%)
- CLAUDE.md §3.5 (seq p90 1393~2215, multi-day span 함의)

## Inheritance from prior H

- F-G (val ceiling 9 H) → input space 다른 측면 시도
- F-1 (H011 input-stage REFUTED for cohort overfit) → input-stage 위험 알고 있음 → minimal 3-feature 로 시작 (H011 의 8-fid 고용량 mutation 과 다름)
- H015 F-1 (recency loss-axis Δ +0.00024 marginal) → loss-axis 가 weak → input-axis 로 mechanism class 전환 정당화
- H010 carry-forward (NS-token enrichment safe stacking) → H032 도 같은 패턴
