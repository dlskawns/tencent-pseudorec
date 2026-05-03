# H014 — Method Transfer

> **Envelope mutation** — no method transfer in mechanism sense. 본 §는
> long-seq processing 의 background literature + UNI-REC alignment.

## ① Source

- **§3.5 데이터 사실** (verified, sibling cite) — 가장 강한 motivation
  source. 모델링 paper 가 아닌 **데이터 자체**.
- **TWIN** (Two-stage Interest Network for Lifelong User Behavior Modeling
  in CTR Prediction, Pan et al. RecSys 2024 / KDD 2024) — Tencent paper,
  long-seq retrieval canonical. **H014 PASS 시 H015 후보**.
- **SIM** (Search-based User Interest Modeling, Pi et al. CIKM 2020) —
  Alibaba paper, target-aware long-seq retrieval.
- **HSTU** (Hierarchical Sequential Transducer Unit, Zhai et al. arXiv 2024) —
  Meta paper, long-seq trunk specialist.
- 카테고리 family (`long_seq_retrieval/`): TWIN / SIM / ETA / HSTU. 신규
  카테고리 cold-start 후보 (H014 PASS 시).

## ② Original mechanism

**Long-seq processing 의 두 패러다임**:

1. **Dense self-attention (H014 가 채택)**: seq 전체에 transformer self-
   attention 적용. O(L²) compute, simple. seq length 직접 확장.
2. **Retrieval/compression (H015 후보)**: target-aware top-K retrieval (TWIN/
   SIM) 또는 hierarchical compression (HSTU) 으로 dominant subseq 만 dense
   처리. O(K²) where K << L.

H014 = (1) 의 minimum viable form. **현재 baseline 이 이미 dense self-attention
(transformer encoder) 사용 중** — 단순히 seq length 만 확장.

## ③ What we adopt

- **Mechanism class**: 없음 (model 변경 0).
- **변경 내용**: run.sh 의 `--seq_max_lens` 만:
  - `"seq_a:64,seq_b:64,seq_c:128,seq_d:128"` → `"seq_a:256,seq_b:256,
    seq_c:512,seq_d:512"`.
- **Code**: train.py / model.py / dataset.py / trainer.py / infer.py 모두
  byte-identical with H010.

## ④ What we modify (NOT a clone)

- **TWIN/SIM/HSTU 의 retrieval/compression 미적용**: H014 가 dense self-
  attention 만 (단순 expansion). retrieval 은 H015 후보. **단일 mutation
  정신** 유지.
- **Per-domain length asymmetric**: domain a/b 256 vs c/d 512 (현재
  asymmetric 64/128 의 같은 4× scaling). uniform 256 도 가능 (sub-H 후보).
- **§17.2 정밀**: envelope mutation = "structural" 인정 (data input shape
  변경). mechanism 변경 0 → confound 작음.

## ⑤ UNI-REC alignment (HARD)

- **Sequential reference**: per-domain encoder + NS xattn (H010). 본 H 가
  sequence length 확장 → encoder 가 더 풍부한 history 처리. **sequential
  axis 의 가장 직접적 강화**.
- **Interaction reference**: H008 DCN-V2 fusion 그대로. enriched seq tokens
  → NS xattn → DCN-V2 (interaction). 변경 없음.
- **Bridging mechanism**: H010 NS xattn 의 K=V (concat 4 도메인) length
  4× 증가 (256+256+512+512=1536, 이전 64+64+128+128=384). attention 후보
  4× 증가, NS-token 7개 의 selective routing 더 풍부한 후보 위에서 작동.
- **primary_category**: 없음 (envelope mutation, no mechanism). 또는 신규
  `long_seq_retrieval/` first-touch 후보 (H015 retrieval 추가 시 본격).
- **Innovation axis**: 8 H 누적 미시도 axis. data motivation 가장 강함
  (95%+ 정보 손실).

## ⑥ Sample-scale viability (Rule UB-1, §10.6)

- 추가 trainable params: **0** (model byte-identical).
- §10.6 cap 위반: 없음.
- Sample-scale risk:
  - **Memory**: O(L²) attention. seq 4× → compute 16×. batch 2048 + seq
    512 OOM risk medium. mitigation = sub-H seq 256 uniform 또는 batch 줄임.
  - **Wall**: 학습 시간 ~2-4× 증가 예상. cost cap (§17.6) 압박.
  - **Long-seq overfit**: 더 많은 정보 = 더 많은 noise. cohort drift 가
    dominant 이면 OOF lift 만 발생, Platform stagnant.

## ⑦ Carry-forward rules to honor

- **§10.5 LayerNorm on x₀**: 변경 없음.
- **§10.6 sample budget cap**: 변경 없음 (params 추가 0).
- **§10.7 카테고리 rotation**: envelope mutation, mechanism category 충돌
  없음.
- **§10.9 OneTrans softmax-attention entropy**: H010 NS xattn entropy
  threshold 0.95 × log(L_total) — L_total 384 → 1536 으로 변경. threshold
  = 0.95 × log(1536) ≈ **6.97** (이전 5.65). H010 baseline entropy 0.81
  유지하면 violation 더 안 함.
- **§10.10 InterFormer bridge gating σ(−2)**: 미적용.
- **§17.2 one-mutation**: envelope mutation, "structural" 인정.
- **§17.3 binary success**: Δ ≥ +0.001pt (sample-scale relaxed) 또는
  +0.005pt (strong).
- **§17.4 카테고리 rotation 재진입 정당화**: 미발동.
- **§17.5 sample-scale = code-path verification only**: 본 H 는 mechanism
  변경 0 → sample-scale code-path 변경 없음. cloud full-data 결과로만 결정.
- **§17.6 cost cap**: extended ~3-5h 예상 (long-seq overhead). 누적 ~30h.
  cap 임박.
- **§17.7 falsification-first**: predictions.md 에 strong / measurable /
  noise / degraded / OOM 분기 명시.
- **§17.8 cloud handoff discipline**: training_request.md + flat upload.
- **§18 inference 인프라 룰**: 변경 없음.
- **H010 F-1**: NS-only enrichment safe pattern → H014 mechanism 변경 0
  → 안전.
- **H013 F-1 (Frame A REFUTED)**: L1 retire → L4 우선순위 강화.
- **H012 F-3 (4-layer ceiling diagnosis)** → H014 = L4 검증.
- **§3.5 데이터 사실** → H014 motivation 직접 출처.
