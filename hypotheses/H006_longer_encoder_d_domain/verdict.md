# H006 — Verdict (REFUTED on P2)

> 클라우드 학습 + inference 완료 (2026-04-28~29). §17.3 binary 임계 미달. long_seq_retrieval 카테고리 일시 archive.

## Status
`done` — REFUTED. Platform AUC 0.82 < anchor (original_baseline) 0.83X by ~1pt. §18 인프라 룰 검증 PASS (batch=256 fix 후 inference 정상 통과, 180초).

## P1 — Code-path success
- Measured: 1 epoch + 0.3 ratio × 10 epoch 학습 4시간 10분 NaN-free 완주, finite val/OOF AUC.
- Verdict: **PASS**.

## P2 — Primary lift (§17.3 binary)
- Measured: **Platform AUC = 0.82**.
- Anchor (original_baseline): Platform AUC ~**0.83X**.
- Δ vs anchor: **−0.01 ~ −0.02 pt** (음수).
- Predicted: Δ ≥ +0.5pt.
- **Verdict: REFUTED**.

## P3 — D vs A/B/C 비대칭 (메커니즘 검증)
- Measured: per-domain attention pattern 직접 측정 안 함 (instrumentation 부재).
- Verdict: UNVERIFIED. P2 REFUTED 자체가 메커니즘 작동 여부와 무관하게 lift 부재 확정.

## P4 — §18 인프라
- Measured: 첫 inference 시 `INFER_BATCH_SIZE=1024` default 로 vGPU OOM (line 1174 sum-pool 단계). default 256 fix + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 추가 후 통과. wall 180초.
- Verdict: **PASS** (after fix). §18.1–§18.5 룰 검증.

## P5 — val ↔ platform AUC alignment
- val_AUC: ~0.82.
- OOF AUC: 0.8562 (10% user holdout).
- Platform AUC: 0.82.
- |val − platform| ≈ 0 (정합).
- |OOF − platform| ≈ 3.5pt (큰 갭).
- Verdict: val 은 platform 과 정합, OOF 는 다른 distribution. **OOF 는 supplementary signal, platform 이 ground truth**.

## Findings (F-N carry-forward)

- **F-1 (P2 REFUTED 본질)**: LongerEncoder 의 top-K=50 selection 이 우리 envelope 의 seq_max_lens=128 에서 단순 truncation 효과. K (50) < L (128) 일 때 정보 손실. 게다가 **selection 이 candidate-unaware** (self-attention probability mass 기반) → random 에 가까움. paper 의 long-seq retrieval claim 은 (a) seq_max_lens >> K 환경 또는 (b) candidate-aware retrieval (DIN/SIM/TWIN/CAN family) 일 때만 의미.
- **F-2 (long_seq_retrieval archive)**: 본 카테고리 우리 smoke envelope 에선 retire. seq_max_lens 확장 (별도 H, compute 큼) 또는 candidate-aware retrieval (H007) 로 카테고리 reformulate.
- **F-3 (OOF ≠ platform 큰 갭)**: 0.8562 vs 0.82 = 3.5pt. OOF holdout user pool 가 platform test 와 다른 distribution (cold-start 또는 cohort effect). **future H 의 paired Δ 는 platform AUC 으로만 결정**, OOF 는 supplementary.
- **F-4 (extended envelope cost)**: train_ratio=0.3 + num_epochs=10 wall 4시간. T2 cap (per-job ≤ $5) 위험. cost-aware planning: anchor 는 smoke 으로 측정, mutation H 는 가능한 smoke 우선, extended 는 검증된 mutation 만.
- **F-5 (§18 인프라 fix 검증)**: batch=256 default + PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True → vGPU partition 메모리 변동에 robust. 180초 inference. **새 carry-forward**: §18.4 default batch_size 1024 → 256 갱신, original_baseline + H005 + H006 + H007+ 모두 적용 (이미 patched).

## Surprises
- val_AUC ≈ platform AUC (둘 다 0.82) — **leakage 가설 무효화**. label_time split + OOF 활성 상태에서 val 이 platform 과 정합. 이전 anchor recalibration 의 unknown 답 = val 신뢰 가능.
- OOF AUC 가 platform 보다 3.5pt 높음 — 사용자가 의도한 "honest measurement" 가 oversight: OOF holdout cohort 가 더 separable. cold-start 또는 user pool distribution 차이.
- LongerEncoder top-K compression 이 우리 envelope 에선 단순 truncation 같다는 거 paper claim 의 scale-dependent nature 직접 확인.

## Update to CLAUDE.md?
- §18.4 default batch_size 1024 → 256 갱신 (이미 코드 적용). CLAUDE.md 본문 갱신 필요.
- 신규 carry-forward 후보: "**Long-seq retrieval mutation 은 seq_max_lens > top_K 환경 또는 candidate-aware retrieval 와 paired 일 때만 의미**". F-1 의 generalization. 본 H 결과 + H007 결과 누적 후 본문 추가 결정.

## Carry-forward to H007

- F-1 → H007 = candidate-aware cross-attention. random selection (H006) 의 직접 후속.
- F-3 → 미래 H 의 paired Δ 측정은 **platform AUC 으로만**. OOF supplementary.
- F-4 → H007 smoke envelope 우선, extended envelope 은 mutation 검증 후.
- F-5 → §18 인프라 룰 H007 패키지에 그대로 inherit. infer.py 에 batch=256 default + PYTORCH_CUDA_ALLOC_CONF.
