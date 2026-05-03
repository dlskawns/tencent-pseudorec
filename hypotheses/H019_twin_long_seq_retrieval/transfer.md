# H019 — Method Transfer

## ① Source

- **Chang, J. et al. 2024 (Tencent)** — "TWIN: TWo-stage Interest Network
  for Long-term User Behavior Modeling." RecSys. **본 H 의 1차 source**.
  GSU (general search unit) lightweight scoring → top-K → ESU (exact
  search unit) full attention. 100M+ user 환경 검증.
- **Pi, Q. et al. 2020 (Alibaba)** — "Search-based User Interest Modeling
  with Lifelong Sequential Behavior Data" (SIM). TWIN 의 prior — hard
  search (category match) → soft search (embedding sim). H019 의 fallback
  reference.
- **Zhai, J. et al. 2024 (Meta)** — "Actions Speak Louder than Words: Trillion-
  Parameter Sequential Transducers for Generative Recommendations" (HSTU).
  trunk replacement form, H019 REFUTED 시 H020 후보 reference.
- 카테고리 family (`retrieval_long_seq/`): TWIN / SIM / HSTU / ETA. 신규
  카테고리 first-touch.

## ② Original mechanism

**TWIN GSU+ESU** (1단락 재서술):

User 의 long history (L=10K+ events) 가 dense full-attention 으로
처리 불가능 (O(L²)). TWIN 의 two-stage:
1. **GSU (General Search Unit)**: lightweight scoring (linear projection +
   inner product) — 모든 history event 의 candidate-relevance score 계산.
   O(L) cost, no parameter sharing with ESU.
2. **Top-K filter**: GSU score 의 top-K=128 (paper) event 만 추출.
3. **ESU (Exact Search Unit)**: top-K event 의 full multi-head attention
   with candidate. O(K²) cost, K ≪ L.

**핵심 가정**: long history 의 대부분은 candidate 와 무관. top-K 만으로
full-attention 효과 회복 가능.

**우리 적용**:
- 도메인별 seq (a/b/c/d) 각각 GSU+ESU 적용.
- envelope seq_max_lens 64-128 → 512 expand (cap), GSU 가 score → ESU 가
  top-K=64 attend.
- candidate (item_id) embedding 으로 candidate-relevance score.

## ③ What we adopt

- **Mechanism class**: TWIN GSU+ESU minimum viable form (per-domain).
- **변경 내용 (5 files + run.sh)**:
  1. `dataset.py`: seq_max_lens 64-128 → 512 (cap). retrieval form 의 의미
     있게 하는 정보 capacity.
  2. `model.py`: 4개 도메인 encoder 의 self-attention block 앞에
     `TWINBlock(GSU+ESU)` 모듈 추가. GSU = `Linear(d_model, d_model//4) +
     candidate inner product`. ESU = MultiHeadAttention(num_heads=4,
     d_model).
  3. `model.py`: candidate embedding 추출 (item_id encoder 출력) → GSU
     scoring input.
  4. `train.py`: argparse `--use_twin_retrieval --twin_top_k 64
     --twin_seq_cap 512`.
  5. `run.sh`: 3 H019 flags + 2 H010 default 명시 bake.
  6. `make_schema.py`: seq_max_lens 변경에 schema 재생성.
- **CLI**: `--use_twin_retrieval --twin_top_k 64 --twin_seq_cap 512`.

## ④ What we modify (NOT a clone)

- **Per-domain (not per-history)**: TWIN paper = single long history.
  본 H = 4 도메인 분할 history. domain-specific GSU+ESU.
- **top-K=64 (not paper 128)**: sample-scale viability (Frame B 우려).
  cap=512 환경에서 64=12% 비율. paper 의 K=128 / L=10K=1.3% 보다
  conservative-aggressive.
- **GSU = simple inner product (not paper learnable scorer)**:
  parameter-free GSU 로 §10.6 sample budget 친화 (paper 의 GSU 는 별도
  embedding table). PASS 시 sub-H = learnable GSU.
- **ESU = standard MultiHeadAttention (not paper customized)**: 기존
  PCVRHyFormer encoder 재사용 가능.
- **§17.2 single mutation**: TWIN module 추가가 1 mechanism class 변경.
  NS xattn + DCN-V2 stack byte-identical.

## ⑤ UNI-REC alignment (HARD)

- **Sequential reference**: NEW axis 강화 — TWIN GSU+ESU 가 sequence axis
  의 long-seq form. SASRec / BERT4Rec 의 dense form 대체.
- **Interaction reference**: 변경 없음 (DCN-V2 fusion 그대로).
- **Bridging mechanism**: 변경 없음.
- **Training procedure**: 변경 없음.
- **primary_category**: `retrieval_long_seq` (NEW first-touch — §17.4
  rotation auto-justified).
- **Innovation axis**: §3.5 quantitative motivation (p90 1393~2215 vs
  envelope 64-128) 의 직접 attack. UNI-REC 의 sequence axis lever 강화.
- **OneTrans / InterFormer / PCVRHyFormer 와의 관계**:
  - OneTrans: NS-token mixed-causal — 변경 없음 (별도 mechanism class).
  - InterFormer: bridge gating — 변경 없음.
  - PCVRHyFormer: per-domain encoder backbone 유지, 그 안 self-attention
    block 앞에 TWIN module insert.

## ⑥ Sample-scale viability (Rule UB-1, §10.6)

- 추가 trainable params: **TWIN module** = 4 도메인 × (GSU=0 inner product
  + ESU=MultiHeadAttention(4, d_model)). d_model=64 시 ESU per-domain ≈ 16K
  params × 4 = 64K. **§10.6 sample budget cap 200** 위반 가능 (4× 초과).
  
  → **mitigation**: ESU shared across domains (parameter sharing) → 16K
  total params (사실상 cap 초과). sample-scale 만 fp16 + 1 epoch sanity →
  cloud full-data 측정.

- §10.6 sample-scale soft cap 위반 명시 — Frame B 우려. **risk acceptance**:
  paradigm shift first entry, sample-scale 측정 무효화 인지.

- Sample-scale viability hard test: **local sanity 1 epoch + 1000-row →
  loss finite + GSU/ESU shape mismatch 없음 + top-K filter 작동**. NaN
  free 시 cloud upload.

## ⑦ Carry-forward rules to honor

- **§10.5 LayerNorm on x₀**: 변경 없음.
- **§10.6 sample budget cap**: **위반 인지** — paradigm shift 첫 entry,
  cloud full-data 측정 의존. sample-scale = code-path P1 만.
- **§10.7 카테고리 rotation**: `retrieval_long_seq` first-touch
  auto-justified.
- **§10.9 OneTrans softmax-attention entropy**: ESU attention 에 적용
  필요. 측정 기록 의무. threshold 0.95 × log(top-K=64) = 0.95 × 4.16 =
  3.95 (upper). lower bound 0.5 (highly selective).
- **§10.10 InterFormer bridge gating σ(−2)**: 미적용.
- **§17.2 one-mutation**: TWIN GSU+ESU 모듈 추가. mechanism stack 의 다른
  부분 byte-identical.
- **§17.3 binary success**: Δ vs H010 corrected ≥ +0.005pt → PASS strong.
  Δ ∈ [+0.001, +0.005pt] → measurable. Δ < +0.001pt → REFUTED.
- **§17.4 rotation**: auto-justified (new category first-touch).
- **§17.5 sample-scale = code-path verification only**.
- **§17.6 cost cap**: T3 ~$15/job. cost cap 위협 (per-campaign ≤ $100).
  H019 REFUTED 시 H020 paradigm shift 추가 시도 어려움 → H019 가
  paradigm shift class 의 critical test.
- **§18.6 dataset-inference-auditor**: H019 upload/ ready 직전 PASS
  의무. seq_max_lens 변경 → make_schema.py + dataset.py 영향 → 재 audit.
- **§18.7 nullable to_numpy**: 영향 없음 (carry-forward H015 패치).
- **§18.8 emit_train_summary**: H019 의 train.py 끝에 SUMMARY 블록 의무.
  H018 와 함께 §18.8 second user.
