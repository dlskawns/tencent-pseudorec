# E_H014 — Experiment Verdict (REFUTED — L4 retire, paradigm shift inevitable)

> Mirror of `hypotheses/H014_long_seq_envelope/verdict.md`.

## Status
`done` — **REFUTED — L4 retire**. 4-layer ceiling diagnosis 의 L1 + L3 + L4
모두 retire. **L2 (cohort drift) 만 남음**. Frame B 강한 confirm.

## Headline numbers
- **Platform AUC: 0.833587** (vs H010 corrected 0.837806 = **−0.0042pt**).
- OOF AUC: 0.8595 (vs H010 0.8596 거의 동일).
- attn_entropy: [0.9915, 0.9920] (H010 [0.81, 0.81] +0.18, 단 비율 더 sparse).
- Wall: 학습 3:30:49 (−6%) / inference 229.87s (+148%).
- OOF-Platform gap: **2.59pt** = 9 H 중 **가장 큼** (cohort drift 강한 신호).

## Decision applied
predictions.md table → "noise / degraded" 분기 → **REFUTED, L4 retire,
paradigm shift inevitable**. H015 = L2 (cohort drift) 우선.

## Critical context (eval data correction)
- Organizer eval data 수정 → H010 재측정 = 0.837806 (이전 0.8408 −0.003 shift).
- prior H011/H012/H013 corrected 재측정 안 됨 → ranking 비교 정밀도 낮음.
- 사용자 직관 "H012 가 가장 높다" = invalid comparison (H010 corrected vs
  H012 prior).

## Setup (OOM mitigation iter-4)
- batch=1024 (사용자 prior regime 2048 절반).
- lr=1e-4, num_workers=2, buffer_batches=4.
- seq_max_lens "192-192-192-192" uniform (a/b 3×, c/d 1.5× from H010).
- PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.
- Memory ~12.5 GiB ≤ 19 GiB partition. 안전.

## Next actions
1. `experiments/INDEX.md` H014 row 갱신 + corrected eval note.
2. `hypotheses/INDEX.md` Active Phase + Active Pipeline → Archive, Recent
   Findings prepend, Backlog reorder.
3. **paradigm shift 결정 turn**: H015 후보 분석 + 사용자 결정.
4. (mandatory) H011/H012/H013 corrected eval 재측정 권장 (verify-claim sanity
   gate 정합성).
