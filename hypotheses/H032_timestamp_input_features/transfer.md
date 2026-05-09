# H032 — Method Transfer

## Sources

- **Cheng et al. 2016** "Wide & Deep Learning for Recommender Systems" — hour/day-of-week 이 wide path 의 standard CTR features.
- **Beutel et al. 2018** "Latent Cross: Making Use of Context in Recurrent Recommender Systems" — context features (time, location) 를 RNN trunk 와 explicit cross.
- **Pi et al. 2020** "Search-based User Interest Modeling with Lifelong Sequential Behavior Data for Click-Through Rate Prediction (SIM)" — temporal recency 가 GSU/ESU retrieval 의 주요 signal.
- **Reference (anti-clone)**: HSTU / OneTrans / InterFormer 도 timestamp 활용 — 단 본 H032 는 minimal additive form (3 categorical embeddings → 1 NS-token), trunk 변경 없음.

## Mechanism — single mutation

H010 anchor (NS xattn + DCN-V2 + RankMixer NS tokenizer) byte-identical EXCEPT:

1. **Dataset side** (`dataset.py:_convert_batch`):
   - 이미 result['timestamp'] 존재 ✅
   - 추가 추출: `max_seq_ts` per row = max across 4 domains 의 ts arrow column 의 row-wise max. (이미 있는 4개 ts column 활용.)
   - result['hour_of_day'] = `(timestamps // 3600) % 24`
   - result['day_of_week'] = `(timestamps // 86400) % 7`
   - result['recency_bucket'] = `clip(0, 7, log2(max(1, timestamps - max_seq_ts)))`.astype(int64)
2. **ModelInput** (`model.py:11-15`) NamedTuple 에 새 field 3개 추가:
   - `hour_of_day: torch.Tensor` (B,)
   - `day_of_week: torch.Tensor` (B,)
   - `recency_bucket: torch.Tensor` (B,)
3. **Time embedding module** (new):
   ```
   self.time_emb_hour = nn.Embedding(24, 16)
   self.time_emb_dow = nn.Embedding(7, 16)
   self.time_emb_recency = nn.Embedding(8, 16)
   self.time_proj = nn.Linear(16, d_model)
   ```
   forward: `time_token = time_proj(LayerNorm(emb_hour + emb_dow + emb_recency))` → shape (B, d_model)
4. **NS-token concat**: user_ns + item_ns + time_token (1개 추가) → DCN-V2 input.

## §17.2 single mutation compliance

1 mutation = "timestamp 에서 derived 3 categorical features 를 1 NS-token 으로 합류". 다른 mechanism 변경 없음.

## §⑤ UNI-REC alignment (P1+ 의무)

- **Sequential**: 변경 없음 (seq encoder 내부 time bucketing 그대로).
- **Interaction**: 변경 없음 (DCN-V2). 단 새 time NS-token 이 cross 입력 합류 → time × user × item × seq implicit cross 학습.
- **Bridging**: 새 NS-token 이 H010 의 NS xattn / DCN-V2 / 분류 head 의 같은 trunk gradient 공유 (P1 룰 ✅).
- **What's not a clone**: Wide&Deep 의 wide path 가 아님 (linear cross 아닌 explicit deep cross 의 일부). Latent Cross 의 RNN multiplication 아님. **minimal additive form** — 새 NS-token 1 개만 추가.

## §10 Anti-bias rules audit

- **§10.5 LayerNorm on x_0**: time_emb sum 후 LayerNorm 적용 → time_proj. ✅
- **§10.6 sample-scale param budget ≤ N/10 = ~200 params**: 추가 params = 24×16 + 7×16 + 8×16 + 16×d_model + d_model (LN). d_model=128 이면 ~2.6K params. **위반** — sample-scale param budget 초과. **Action**: T0/T1 sample 학습 skip, T2 cloud (1.5M+ rows) 만 학습.
- **§10.7 category rotation**: 직전 2 H primary_category — H030 (measurement), H029 (measurement). H032 = `temporal_input` (NEW first-touch — §10.7 FREE).
- **§10.10 InterFormer bridge gating init = sigmoid(−2)**: time_token 합류 시 별도 gate 미사용 (additive concat) → §10.10 N/A.

## Differentiation from H015~H018 (loss-axis)

| H | Axis | Signal location | Result |
|---|---|---|---|
| H015 | Loss reweight | per-batch linear recency [0.5, 1.5] | REFUTED Δ +0.00024 marginal |
| H016 | Eval framework | OOF redefine future-only | REFUTED model, PASS infra |
| H017 | Loss reweight | per-batch exp decay | INVALID (submission lost) |
| H018 | Loss reweight | per-user exp τ=14 | SCAFFOLDED 미실행 |
| **H032** | **Input feature** | **per-sample 3 derived categoricals → NS-token** | **TBD** |

H032 의 핵심 차별: model 이 forward path 에서 per-sample temporal context 직접 학습. loss-axis 는 gradient magnitude 만 조절 → mechanism 차이 큼.

## Implementation risk

- **Medium**: dataset.py 의 `_convert_batch` 마지막에 4 cells 추가 + ModelInput NamedTuple 확장 + 새 Embedding 3개 + Linear 1개 + forward path 수정. 1714-line model.py 의 forward path 변경 → side-effect 가능.
- **Risk: max_seq_ts 추출**: 4 도메인의 ts column 의 row-wise max → arrow ListArray 처리 필요. dataset.py line 633-648 의 ts 추출 로직 재사용 가능.
- **Risk: inference data 의 timestamp null**: §18.7 룰 (label_time fill_null) 처럼 timestamp.fill_null(0) 후 dummy bucket 처리 필요. dataset-inference-auditor 서브에이전트 invoke mandatory.
- **Mitigation**: H019 / H018 처럼 upload/ build 는 별도 세션에서 careful local sanity (T0 1000-row forward + ModelInput shape + dummy timestamp inference) 후 진행.

## Local sanity test plan (upload build 시 mandatory)

1. T0: `_convert_batch` 4 cells 추가 → batch dict 의 hour/dow/recency shape (B,) int64
2. T0: max_seq_ts 추출 로직 — 4 domain ts column 의 row-wise max 계산 결과가 모든 row 에 대해 finite + ≤ timestamp
3. T1: 1-batch forward NaN-free + ModelInput shape consistent
4. T1: backward → time embedding gradient non-zero
5. T1: inference dummy: timestamp 는 모든 row 0 으로 강제 → forward 에서 hour=0, dow=0, recency=0 (default bucket) → predictions 정상
6. **dataset-inference-auditor 서브에이전트 invoke** (CLAUDE.md §19, §18.7 timestamp.fill_null(0) 추가 검증)
