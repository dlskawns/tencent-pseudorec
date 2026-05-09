# H020 — Method Transfer

## ① Source

- **Chang, J. et al. 2024 (Tencent)** — "TWIN: TWo-stage Interest Network for Long-term User Behavior Modeling." RecSys 2024. **본 H 의 1차 source — paper-faithful GSU 직접 검증**.
- **Pi, Q. et al. 2020 (Alibaba)** — "Search-based User Interest Modeling with Lifelong Sequential Behavior Data" (SIM). soft search = projection + inner product 형태 reference.
- **H019** — sub-H base. parameter-free inner product GSU 의 learnable 변환.
- 카테고리 (`retrieval_long_seq/`): TWIN / SIM / HSTU / ETA. re-entry from H019, scoring axis 직접 검증.

## ② Original mechanism (TWIN paper-faithful GSU)

TWIN paper 의 GSU 는 lightweight learnable scorer:
```
score_i = <Q · candidate_emb, K · history_emb_i>
```
- Q, K = nn.Linear projection (d_model → d_proj). paper d_proj = d_model 또는 d_model//2.
- candidate 와 history 가 *별도 metric space* 에서 비교됨 (backbone embedding space 와 분리).
- O(L · d_proj) cost — H019 inner product (O(L · d_model)) 보다 약간 비싸지만 여전히 lightweight.

**H019 simplified form** (parameter-free):
```
score_i = <candidate_emb, history_emb_i>   # backbone embedding space 직접 사용
```

## ③ What we adopt (H020 mutation)

- **Mechanism**: TWINBlock 의 GSU scoring 에 `nn.Linear(d_model, d_model // 4)` 1쌍 추가 (W_q on candidate, W_k on history). parameter-free → learnable projection 1단계 깊이.

- **변경 내용 (1 file)**:
  - `model.py` `TWINBlock`:
    - `__init__`:
      ```python
      self.gsu_q = nn.Linear(d_model, d_model // 4, bias=False)
      self.gsu_k = nn.Linear(d_model, d_model // 4, bias=False)
      ```
    - `forward` GSU score 계산부:
      ```python
      # H020: learnable projection before inner product (paper-faithful GSU)
      q = self.gsu_q(candidate)  # (B, d_model // 4)
      k = self.gsu_k(history)    # (B, L, d_model // 4)
      scores = (k * q.unsqueeze(1)).sum(-1)  # (B, L)
      ```
  - 다른 모든 부분 (top_k, ESU MultiheadAttention, residual, LayerNorm, gate, aggregator) **byte-identical to H019**.

- **CLI**: 새 flag 없음 (H019 대비 변경 없는 hyperparam). `--use_twin_retrieval --twin_top_k 64 --twin_gate_init -2.0` 등 H019 과 동일.

- **Argparse 추가**: `--twin_learnable_gsu` (bool flag) — default False 로 H019 호환 유지. `run.sh` 에서 `--twin_learnable_gsu` 추가.

## ④ What we modify (NOT a clone of paper)

- **projection dim d_model // 4 (= 16)**: paper d_proj = d_model 또는 d_model//2. 본 H = d_model//4 — §10.6 sample budget 친화 + Frame B 우려 인지. PASS strong → sub-H = d_model//2 확장.
- **bias=False**: paper 명시 안 함. inner product 의 magnitude 는 candidate/history embedding norm 으로 충분, bias 추가 안 정당화.
- **per-domain (not per-history)**: H019 carry-forward — 4 도메인 별 W_q/W_k 4쌍. paper single history 와 다름.
- **§17.2 single mutation**: GSU scoring function 의 parameter-free → learnable projection 1단계. ESU / top_k / aggregator / gate 전부 H019 byte-identical.

## ⑤ UNI-REC alignment (HARD)

- **Sequential reference**: H019 carry-forward — TWIN GSU+ESU per-domain. scoring axis 정밀화.
- **Interaction reference**: 변경 없음 (DCN-V2 fusion + H010 NS→S xattn 그대로).
- **Bridging mechanism**: 변경 없음.
- **Training procedure**: 변경 없음.
- **primary_category**: `retrieval_long_seq` (re-entry, sub-H justified).
- **Innovation axis**: H019 의 retrieval form 안 selection policy axis. backbone embedding space 와 retrieval-specific space 분리 — UNI-REC sequence axis 의 정밀도 향상.
- **OneTrans / InterFormer / PCVRHyFormer 와의 관계**:
  - OneTrans / InterFormer: 변경 없음 (별도 mechanism class).
  - PCVRHyFormer: per-domain encoder backbone 유지, TWINBlock 안의 GSU 만 learnable 화.

## ⑥ Sample-scale viability (Rule UB-1, §10.6)

- 추가 trainable params: **W_q + W_k** = 2 × (64 × 16) = 2048 per domain × 4 domains = **8,192 params**. (bias=False).
- TWIN module 합산: H019 71K + H020 8K = ~79K. total 161M 의 0.049% (vs H019 0.044%).
- **§10.6 sample budget cap (200 hard, 2146 soft) 위반 인지** — H019 carry-forward (paradigm shift class exempt). cloud full-data 측정 의존.
- Sample-scale viability hard test: **local sanity 1 epoch + 1000-row → loss finite + projection forward NaN-free + W_q/W_k grad flow + ablation diff (H019 vs H020 forward output > 0.001)**. NaN free 시 cloud upload.

## ⑦ Carry-forward rules to honor

- **§10.5 LayerNorm on x₀**: 변경 없음.
- **§10.6 sample budget cap**: 위반 인지 (H019 carry, paradigm shift class).
- **§10.7 카테고리 rotation**: `retrieval_long_seq` re-entry — challengers.md 에 4 사유 명시 (RE_ENTRY_JUSTIFIED).
- **§10.9 OneTrans softmax-attention entropy**: ESU attention 측정 carry-forward (H019 와 동일 기준 — threshold 0.95 × log(top_k=64) = 3.95 upper, 0.5 lower).
- **§10.10 InterFormer bridge gating σ(−2)**: H019 의 twin_gate 그대로 유지 (sigmoid(-2)≈0.12 init).
- **§17.2 one-mutation**: GSU scoring function 의 parameter-free → learnable projection. 다른 모든 부분 byte-identical.
- **§17.3 binary success**: Δ vs H019 ≥ +0.003pt → PASS strong (sub-H 깊이 들어가는 변경의 보수적 임계). [+0.001, +0.003pt] measurable. < +0.001pt REFUTED.
- **§17.4 rotation**: re-entry justified (challengers.md §17.4 블록).
- **§17.5 sample-scale = code-path verification only**.
- **§17.6 cost cap**: T2.4 ~3.5h × $5-7. H019 동급, campaign cap 친화.
- **§18.6 dataset-inference-auditor**: H020 upload/ ready 직전 PASS 의무. 단 H019 와 dataset.py / make_schema.py / infer.py 동일 → schema 변경 없음 (model.py 만 변경).
- **§18.7 nullable to_numpy**: H015 carry-forward (영향 없음, dataset.py 변경 없음).
- **§18.8 emit_train_summary**: H019 의 train.py 의 SUMMARY 블록 그대로 carry. exp_id 만 H020_learnable_gsu 로 변경.
