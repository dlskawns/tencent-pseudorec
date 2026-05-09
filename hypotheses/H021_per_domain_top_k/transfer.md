# H021 — Method Transfer

## ① Source

- **Chang, J. et al. 2024 (Tencent)** — "TWIN" RecSys 2024. paper top-K=128 (single history). 본 H 의 multi-history (4 도메인) per-domain K = paper-uncovered extension.
- **Pi, Q. et al. 2020 (Alibaba)** — "SIM" KDD 2020. soft search + hard search 의 K 가 search type 별 다름 — per-search-policy K 의 reference.
- **H019** — sub-H base. uniform top_k=64 의 per-domain dict 변환.
- 카테고리 (`retrieval_long_seq/`): re-entry from H019/H020. quantity axis 검증.

## ② Original mechanism (TWIN paper K policy)

TWIN paper 의 K = 128 (single user history L=10K+). 본 H 데이터:
- 4 도메인 분할 history (a/b/c/d), 각 도메인 p90 887~2215 (§3.5).
- H019 의 uniform K=64 = paper K/L 1.3% 보다 conservative-aggressive (4.1~7.2%).
- per-domain K assignment = 본 데이터 specific (paper-uncovered).

**§3.5 정량 motivation**:

| 도메인 | p90 seq len | K=64 의 vs p90 비율 | K=96 의 vs p90 비율 |
|---|---|---|---|
| a | 1562 | 4.1% | 6.1% |
| b | 1393 | 4.6% | 6.9% |
| c | 887 | 7.2% | (변경 안함) |
| d | 2215 | **2.9%** | **4.3%** |

→ domain d 만 K=96 (paper K/L 1.3% 보다 여전히 3× 높지만, K=64 보다 50% 확장).

## ③ What we adopt (H021 mutation)

- **Mechanism**: TWINBlock 의 top_k 를 도메인별로 다르게 instantiate. **TWINBlock class 자체는 byte-identical** (이미 top_k 를 init param 으로 받음).

- **변경 내용 (3 files)**:
  - `model.py` `PCVRHyFormer.__init__` — `twin_top_k` 의 type 변경 (int → int | dict):
    ```python
    # 기존 (H019):
    twin_top_k: int = 64,
    ...
    self.twin_blocks = nn.ModuleDict({
        f'twin_{domain}': TWINBlock(d_model=d_model, num_heads=twin_num_heads, top_k=twin_top_k)
        for domain in ('a', 'b', 'c', 'd')
    })

    # H021:
    twin_top_k: Union[int, Dict[str, int]] = 64,
    ...
    if isinstance(twin_top_k, int):
        top_k_per_domain = {d: twin_top_k for d in ('a', 'b', 'c', 'd')}
    else:
        top_k_per_domain = twin_top_k
    self.twin_blocks = nn.ModuleDict({
        f'twin_{domain}': TWINBlock(d_model=d_model, num_heads=twin_num_heads, top_k=top_k_per_domain[domain])
        for domain in ('a', 'b', 'c', 'd')
    })
    ```
  - `train.py` argparse:
    ```python
    parser.add_argument('--twin_top_k_per_domain', type=str, default=None,
                        help='H021: comma-separated per-domain K, e.g. "64,64,64,96" for a,b,c,d')
    ```
    parsing:
    ```python
    if args.twin_top_k_per_domain:
        ks = [int(x) for x in args.twin_top_k_per_domain.split(',')]
        assert len(ks) == 4, "twin_top_k_per_domain must be 4 ints (a,b,c,d)"
        twin_top_k_arg = {d: k for d, k in zip(('a', 'b', 'c', 'd'), ks)}
    else:
        twin_top_k_arg = args.twin_top_k   # backward compat
    ```
  - `run.sh`:
    ```bash
    --twin_top_k_per_domain "64,64,64,96"   # H021: domain d만 96
    ```

- **TWINBlock class**: **byte-identical to H019**. 이미 top_k 를 init param 으로 받기 때문에 class 변경 불필요.

- **다른 모든 부분 byte-identical**: ESU / aggregator / gate / num_heads / seq_max_lens / batch / NS xattn / DCN-V2 stack 전부 H019 그대로.

## ④ What we modify (NOT a clone of paper)

- **per-domain K (not paper single K)**: paper single history → 본 데이터 multi-history. K=96 for d 는 paper top_k=128 의 75% (paper-faithful 영역).
- **단일 도메인 변경 (d only)**: 보수적 single-axis test. multi-domain 동시 변경 시 효과 분리 어려움.
- **K=96 (not 128)**: K=128 sweep 결과 flat — 모든 도메인에서 추가 lift 없음. domain d 만 K=96 = sweep flat zone (96~128) 의 conservative end.
- **§17.2 single mutation**: top_k policy 의 uniform → domain-aware. TWINBlock 내부 변경 0. PCVRHyFormer wiring 만.

## ⑤ UNI-REC alignment (HARD)

- **Sequential reference**: H019 carry-forward — TWIN GSU+ESU per-domain. quantity axis 정밀화 (per-domain K).
- **Interaction reference**: 변경 없음 (DCN-V2 fusion + H010 NS→S xattn 그대로).
- **Bridging mechanism**: 변경 없음.
- **Training procedure**: 변경 없음.
- **primary_category**: `retrieval_long_seq` (re-entry, sub-H justified, 3회 연속).
- **Innovation axis**: H019 의 retrieval form 안 quantity axis (per-domain K). §3.5 quantitative motivation 직접 활용 — UNI-REC sequence axis 의 도메인-인지 정밀화.
- **OneTrans / InterFormer / PCVRHyFormer 와의 관계**: H020 과 동일 (변경 없음).

## ⑥ Sample-scale viability (Rule UB-1, §10.6)

- 추가 trainable params: **0** (top_k 는 hyperparam, weight 아님). TWINBlock class byte-identical — 추가 weight 없음.
- TWIN module 합산: H019 71K = H021 71K. total 161M 의 0.044% (H019 동일).
- §10.6 sample budget 영향 없음 (parameter-free 변경).
- Sample-scale viability hard test: **local sanity 1 epoch + 1000-row → loss finite + 4 도메인 TWINBlock instantiation 정상 (top_k 64/64/64/96 확인) + domain d 의 K=96 forward shape 정상**. NaN free 시 cloud upload.

## ⑦ Carry-forward rules to honor

- **§10.5 LayerNorm on x₀**: 변경 없음.
- **§10.6 sample budget cap**: H019 동일 (parameter-free 변경).
- **§10.7 카테고리 rotation**: `retrieval_long_seq` 3회 연속 — challengers.md 에 5 사유 명시 (RE_ENTRY_JUSTIFIED).
- **§10.9 OneTrans softmax-attention entropy**: ESU attention 측정 carry-forward (H019/H020 동일). domain d 의 K=96 → threshold 0.95 × log(96) = 4.34 upper.
- **§10.10 InterFormer bridge gating σ(−2)**: H019 의 twin_gate 그대로 (sigmoid(-2)≈0.12).
- **§17.2 one-mutation**: top_k policy 의 uniform → domain-aware. TWINBlock 내부 변경 없음.
- **§17.3 binary success**: Δ vs H019 ≥ +0.003pt → PASS strong (H020 동일 임계, sub-H 보수적 cut).
- **§17.4 rotation**: re-entry justified (challengers.md §17.4 블록).
- **§17.5 sample-scale = code-path verification only**.
- **§17.6 cost cap**: T2.4 ~3.5h × $5-7. H019/H020 동급.
- **§18.6 dataset-inference-auditor**: H021 upload/ ready 직전 PASS 의무. dataset.py / infer.py / make_schema.py / model.py 의 TWINBlock class 부분 모두 byte-identical → audit 범위 좁음 (PCVRHyFormer wiring + train.py argparse + run.sh).
- **§18.7 nullable to_numpy**: H015 carry-forward (변경 없음).
- **§18.8 emit_train_summary**: H019 의 train.py SUMMARY 블록 carry, exp_id 만 H021 로 변경.
