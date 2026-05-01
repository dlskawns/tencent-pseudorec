# H010 — Method Transfer (envelope-only)

> 본 H 는 mechanism mutation 0. envelope mutation only. paper 인용 없음.

## ① Source

- **Internal source**:
  - `experiments/original_baseline/upload/` (12 파일) — PCVRHyFormer pure
    baseline + label_time split + 10% OOF + §18 인프라 룰. 코드 byte-identical
    재사용.
  - `hypotheses/H007_candidate_aware_xattn/verdict.md` F-3 — anchor extended
    measurement 미래 H 명시.
  - `hypotheses/H009_combined_xattn_dcn_v2/verdict.md` F-3 — anchor 정확값
    의존성 정량 노출.
  - `hypotheses/H008_dcn_v2_block_fusion/verdict.md` F-4 — patience=3 + early
    stop aggressive carry-forward.

## ② Original mechanism

본 H 는 mechanism mutation 0. measurement objective.

original_baseline mechanism 그대로:
- **Backbone**: PCVRHyFormer.
- **Per-domain seq encoder**: TransformerEncoder (4 도메인 a/b/c/d 각각).
- **Per-domain query decoder**: cross-attention from Nq=2 queries to seq.
- **Token fusion**: `RankMixerBlock` (decoded_q + NS tokens token-mixing).
- **NS tokenizer**: rankmixer parameter-free chunking, user_ns=5 + item_ns=2 = 7
  NS tokens.
- **Loss**: BCE.
- **Sparse**: high-cardinality embedding via Adagrad.
- **Dense**: AdamW.

## ③ What we adopt

- **코드 byte-identical**: original_baseline/upload/ 의 12 파일 (run.sh 제외)
  그대로. mechanism 변경 0.
- **Envelope 변경 (단일 mutation)**:
  | flag | original_baseline (smoke) | H010 (extended) |
  |---|---|---|
  | `--num_epochs` | 1 | **10** |
  | `--train_ratio` | 0.05 | **0.3** |
  | `--patience` | 5 | **3** (H008 F-4 carry-forward) |
  | 그 외 모든 args | (byte-identical) | (byte-identical) |
- **§18 인프라 룰**: original_baseline 패키지에 이미 포함 → byte-identical 재사용.
- **infer.py**: byte-identical (새 cfg key 없음).

## ④ What we modify (NOT a clone)

- **mechanism 추가 0**: H006~H009 의 model.py 변경 (longer encoder, candidate
  xattn, DCN-V2, OneTrans router 등) 어떤 것도 포함 안 함. **pure baseline**.
- **CLI flags 추가 0**: original_baseline run.sh 의 args 에 extended envelope
  값만 변경, 새 flag 추가 0.
- **infer.py cfg.get 추가 0**: byte-identical.

## ⑤ UNI-REC alignment

- **Sequential axis**: 변경 없음 (original_baseline transformer encoder + query
  decoder 그대로).
- **Interaction axis**: 변경 없음 (RankMixerBlock fusion 그대로).
- **Bridging mechanism**: 변경 없음.
- **primary_category**: n/a (envelope mutation, mechanism category 부여 안 함).
- **Innovation axis**: n/a (measurement objective).

§17.4 카테고리 rotation 룰 적용 안 됨 (mechanism category 부여 없음). H011+ 부터
mechanism category rotation 다시 적용.

## ⑥ Sample-scale viability (Rule UB-1, §10.6)

- 추가 trainable params: **0**.
- Total params: ~198M (original_baseline 동일).
- §10.6 cap 면제 (anchor 와 동일).
- Extended envelope (train_ratio=0.3 × 10 epoch ≈ 51M sample steps): 충분.
- 본 H smoke envelope 우선 미적용 — extended 에서만 측정 (envelope 효과 정확
  isolation 위해).

## ⑦ Carry-forward rules to honor

- **§10.5 LayerNorm on x₀ MANDATORY**: original_baseline 는 DCN-V2 미사용 →
  적용 대상 아님.
- **§10.6 sample budget cap**: anchor 면제.
- **§10.7 카테고리 rotation**: mechanism category 부여 안 함 → 적용 안 됨.
- **§10.9 OneTrans softmax-attention entropy abort**: original_baseline 는
  OneTrans 미사용 → 적용 안 됨.
- **§10.10 InterFormer bridge gating σ(−2)**: 본 H 는 새 bridge 추가 없음 →
  적용 안 됨.
- **§17.2 one-mutation**: envelope 변경 단일 mutation. ✓ (mechanism mutation 0,
  envelope mutation 1).
- **§17.3 binary success**: ≥ +0.5pt 임계 적용 안 됨 (measurement objective).
- **§17.4 카테고리 rotation**: 적용 안 됨.
- **§17.5 sample-scale**: extended envelope 결과는 anchor 의 ground truth 측정.
- **§17.6 cost cap**: extended ~3-4시간, T2 cap 안. 누적 cost 압박 지속.
- **§17.7 falsification-first**: predictions.md 에 시나리오 분기 (A/B/C) +
  부수 게이트 falsification.
- **§17.8 cloud handoff discipline**: training_request.md + flat upload (12
  files) + git_sha pin.
- **§18 inference 인프라 룰**: original_baseline 패키지에 이미 포함.
