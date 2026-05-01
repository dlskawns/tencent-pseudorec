# H011 — Method Transfer

## ① Source

- **DLRM** (Deep Learning Recommendation Model, Naumov et al. 2019,
  arXiv:1906.00091) — sparse embedding lookup × dense feature 를 input
  stage 에서 결합 후 interaction layer 로. dense-sparse fusion at embedding
  stage 의 canonical form.
- **FwFM** (Field-weighted Factorization Machine, Pan et al. WWW 2018,
  arXiv:1806.03514) — field-level weight 를 cross 항에 곱함. aligned pair
  의 weight semantics 에 가장 가까운 family.
- **DIN** (Deep Interest Network, Zhou et al. KDD 2018, arXiv:1706.06978)
  — historical behavior 의 attention weight 를 candidate 와의 relevance 로
  계산 (soft alignment). 본 H 는 hard alignment (fid pair) 의 단순 form.
- **Wide & Deep** (Cheng et al. RecSys 2016, arXiv:1606.07792) — dense
  feature 와 cross feature 를 같은 deep tower 에 input.
- 카테고리 family (`feature_engineering` / `interaction_encoding`):
  field-weighted embedding, weighted feature lookup, dense-sparse fusion at
  input stage. 신규 카테고리 first-touch.

## ② Original mechanism

**DLRM input fusion** (1단락 재서술):

DLRM 은 sparse categorical features 를 embedding lookup 으로 dense vector
로 변환 (`E[id]`), dense numerical features 를 MLP 로 같은 차원 vector 로
변환 한 후, 두 source 를 element-wise 또는 dot-product 로 조합해서
interaction layer 의 입력으로 만든다. 핵심: **dense feature 가 sparse
embedding 과 같은 stage 에서 통합** — interaction layer 가 둘 사이 cross
를 학습할 때 explicit alignment 제공.

**FwFM weight semantics** (1단락 재서술):

FwFM 은 field i 와 field j 의 cross 항 `<v_i, v_j>` 에 field-pair-specific
scalar weight `r_{F(i),F(j)}` 를 곱한다. 본 H 의 aligned pair 는 변형 —
**같은 fid 안에서** ID embedding 과 weight 가 element-wise 결합. field
간 cross 가 아니라 field 내 ID-weight binding.

## ③ What we adopt

- **Mechanism class**: aligned fid pair `{62, 63, 64, 65, 66, 89, 90, 91}`
  (8 fids, verified 출처: `competition/ns_groups.json`) 의 input-stage
  weighted embedding. DLRM 의 dense-sparse 결합 + FwFM 의 weight semantics
  + DIN 의 element-wise weighted lookup 의 minimum viable form.
- **데이터 layout 사실**:
  - `user_int_feats_k` (k ∈ aligned set): list<int64>, per-row array length
    = `n_k` (variable, padding 처리됨).
  - `user_dense_feats`: 10 fids 의 multi-dim list 가 concat 되어 per-row
    flat tensor (B, total_dim=918). per-fid k 의 slice = `[offset_k,
    offset_k + dim_k)` (`competition/dataset.py:_user_dense_plan`).
  - **Critical question (P0 audit)**: `n_k` (user_int 측 array length) 와
    `dim_k` (user_dense 측 slice dim) 가 같은가? 같으면 position-wise binding
    직접; 다르면 binding semantics 명확화 필요 (예: dense 가 fixed 통계
    summary 라면 broadcast).
- **신규 module / 변경**:
  1. `RankMixerNSTokenizer.__init__` 에 `aligned_pair_fids: Optional[List[int]]
     = None`, `aligned_dense_offsets: Optional[Dict[int, Tuple[int, int]]] =
     None` (per-fid offset/dim 매핑) 인자.
  2. `RankMixerNSTokenizer.forward` 에서 aligned fid k 의 embedding lookup
     직후 weighted multiply:
     - `n_k == dim_k` 시: `E_k = E_k * w_k.unsqueeze(-1)` where `E_k` shape
       `(B, n_k, D)`, `w_k` shape `(B, n_k)`.
     - `n_k != dim_k` 시: P0 audit fail → INVALID 분류, retract.
  3. non-aligned fids (38 user_int + dense-only `{61, 87}`) 는 변경 없음.
     `user_dense_proj` 그대로 (단 aligned fid slices 는 user_dense_proj 입력
     에서 제외할지 — 이중 사용 (weighted embedding + dense_proj concat) 도
     option, sub-form 비교).
  4. dense feature scale handling: raw weight [범위 검증 P0] / sigmoid 게이팅
     / LayerNorm — minimum viable form = raw multiply.
- **CLI flag**: `--use_aligned_pair_encoding`, `--aligned_pair_form
  {multiply, gated_multiply}`, `--aligned_pair_dense_dual {exclude, include}`.
- **PCVRHyFormer constructor 인자**: `use_aligned_pair_encoding: bool =
  False`, `aligned_pair_form: str = 'multiply'`, `aligned_pair_fids: List[int]
  = [62, 63, 64, 65, 66, 89, 90, 91]`, `aligned_dense_offsets: Dict[int,
  Tuple[int, int]] = {…}` (P0 단계에서 산출).
- **Audit step 첫 (P0)**: `competition/dataset.py` 의 `_user_dense_plan`
  에서 aligned fid k 의 (offset_k, dim_k) 추출 → eda/out/aligned_offsets.json
  산출. `n_k` 측정은 `demo_1000.parquet` 에서 fid k 별 array length 분포
  추출 (per-row / mean / max). `n_k == dim_k` 검증 필수.

## ④ What we modify (NOT a clone)

- **DLRM 의 모든 dense feature 를 결합 stage 에 안 보냄**: DLRM 은 dense
  numerical features 모두를 MLP → interaction layer. 우리는 aligned pair
  (8 fids = `{62, 63, 64, 65, 66, 89, 90, 91}`) 만 weighted embedding,
  dense-only fids (`{61, 87}`) 는 기존 baseline 의 `user_dense_proj` 그대로.
  **단일 mutation 정신** 유지.
- **FwFM 의 field-pair cross 미적용**: FwFM 은 field 간 cross. 우리는 같은
  fid 안 ID-weight binding (intra-fid). field-pair cross 는 H008 DCN-V2 의
  영역.
- **DIN 의 attention weight 미적용**: DIN 은 candidate-history attention
  으로 weight 계산. 우리는 raw `user_dense_feats[k]` 를 그대로 사용 (학습
  된 attention 아닌 직접 데이터 weight). DIN-style attention 적용은 sub-H.
- **multi-form 한 번에 시도 안 함**: gated_multiply / log_weight / quantile-
  bucketize 형태는 sub-H. minimum viable form (raw multiply) 부터.
- **Layer 추가 0**: parameter-free element-wise 곱셈만. 구조 변경 최소.
- **§17.2 one-mutation**: input embedding lookup 직후 element-wise multiply
  추가. 다른 모든 component (NS tokenizer, NS xattn, DCN-V2 fusion, query
  decoder) byte-identical.

## ⑤ UNI-REC alignment (HARD)

- **Sequential reference**: 변경 없음 — transformer encoder + NS xattn
  (H010) 그대로. 단 NS tokenizer 입력 embedding 이 weighted form 으로
  들어가서 NS 표현 enrichment.
- **Interaction reference**: DCN-V2 (H008 anchor) 그대로 + input-stage
  explicit `<id, weight>` cross 추가. interaction axis 의 새 form (intra-fid
  ID-weight binding).
- **Bridging mechanism**: input-stage `id × weight` → NS tokenizer (5 user
  NS tokens 표현 변경) → seq encoder + NS xattn (output 표현 변경) → DCN-V2
  fusion (input 표현 변경) → final ranking. **input stage** 변경이 모든
  downstream gradient 공유. §0 P1 ("seq + interaction 한 블록 gradient
  공유") 의 strongest form (input = 모든 block 의 root).
- **primary_category**: `feature_engineering` (신규 카테고리). 또는 기존
  `interaction_encoding` 카테고리 신규 생성.
- **Innovation axis**: §4.8 mandate 의 직접 구현 + DLRM/FwFM/DIN 의
  element-wise binding 의 minimum viable form 추출. 1:1 복제 아닌 부분 = aligned
  pair 만 선택 적용 (전체 dense 가 아닌 8 verified shared fids),
  parameter-free, single mutation.

## ⑥ Sample-scale viability (Rule UB-1, §10.6)

- 추가 trainable params: **0** (parameter-free element-wise multiplication).
- audit step 의 추가 지표 buffer (dense feat index map): 0 trainable param
  (constant lookup).
- Total params: ~198M (H010 동일). §10.6 cap 면제 (anchor envelope 동일).
- Sample-scale 부담 없음 — params 추가 0, 학습 cost 동일.
- 단 audit 실패 시 (fid 매핑 inconsistent) INVALID 분류 — 학습 시작 전
  audit 통과 mandatory.

## ⑦ Carry-forward rules to honor

- **§10.5 LayerNorm on x₀**: input embedding lookup 후 NS tokenizer 입력
  의 LayerNorm 보존. weighted embedding 적용은 LayerNorm 전에 — `E[k] *
  weight[k] → LayerNorm` 순서. (LayerNorm 후 곱하면 normalize 효과 깨짐.)
- **§10.6 sample budget cap**: anchor envelope 동일, params 추가 0. PASS.
- **§10.7 카테고리 rotation**: `feature_engineering` 신규 first-touch.
  재진입 정당화 불필요.
- **§10.9 OneTrans softmax-attention entropy**: H010 NS xattn 그대로 →
  본 H 가 weight-modified embedding 입력 시 entropy 변화 측정 (P2 mechanism
  check). threshold 5.65 violation 모니터.
- **§10.10 InterFormer bridge gating**: 본 H 는 새 bridge/gate 추가 없음
  (parameter-free multiply). 미적용.
- **§17.2 one-mutation**: input embedding lookup 직후 element-wise multiply
  추가. flag 분기. PASS.
- **§17.3 binary success**: Δ ≥ +0.5pt platform AUC vs anchor (H010 0.8408).
  단 sample-scale extended 한계 인정 → seed×3 paired bootstrap CI > 0 으로
  완화 가능 (seed 1회 시 점추정 + warning).
- **§17.4 카테고리 rotation 재진입 정당화**: 미발동 (FREE first-touch).
- **§17.5 sample-scale = code-path verification only**: extended 결과는
  mechanism 효과 measurement.
- **§17.6 cost cap**: extended ~3-3.5시간, T2 cap 안. 누적 cost ~21시간
  (H006~H011). cap 압박 → fp16/batch=512 또는 train_ratio=20% (H010 의
  30% 보다 절약) 검토.
- **§17.7 falsification-first**: predictions.md 에 audit fail 시 INVALID,
  Δ < +0.001pt 시 noise 로 분류 명시.
- **§17.8 cloud handoff discipline**: training_request.md + flat upload
  + git_sha pin.
- **§18 inference 인프라 룰**: H010 패키지에서 inherit + H011 의 새 cfg
  keys (use_aligned_pair_encoding, aligned_pair_form, aligned_pair_fids)
  read-back 추가.
- **H010 F-1 (NS-only enrichment safe pattern)** → H011 input-stage enrichment
  도 같은 원리 (downstream 텐서 byte-identical). 안전 stacking 보장.
- **§4.8 mandate** → audit step 으로 baseline 룰 위반 여부 확인. 위반 confirmed
  시 본 H 자체가 룰 통과 패치.
