# H014_long_seq_envelope — Technical Report

> Single mutation: **`seq_max_lens 64-128 → 256-512` (4× per domain)**.
> H010 mechanism + 모든 코드 byte-identical. **L4 (truncate 정보 손실)**
> ceiling diagnosis. 4-layer ceiling diagnosis 의 마지막 unexplored axis.

## Why now

8 H 누적 (H006~H013) ceiling 0.82~0.8408. 4-layer diagnosis:
- L1 (hyperparameter regime): H013 REFUTED — Frame A retire.
- L3 (NS xattn sparse routing): H011/H012/H013 marginal — retire.
- L2 (cohort drift): 8 H OOF-Platform gap 1.88~2.42pt 일관, **남은 가설**.
- **L4 (truncate)**: 단 한 번도 측정 안 됨. **가장 강한 데이터 motivation**.

## Data motivation (§3.5)

| Domain | p50 | p90 | max | frac_empty |
|---|---|---|---|---|
| a | 577.5 | 1562.1 | 1888 | 0.5% |
| b | 405.0 | 1393.0 | 1952 | 1.2% |
| c | 322.0 | 887.3 | 3894 | 0.2% |
| d | 1035.5 | 2215.3 | 3951 | 8.0% |

→ 모든 도메인 p90 ≫ 100. 현재 truncate 64-128 = **95%+ 정보 손실**.
H014 expansion 64-128 → 256-512 = 4×. 여전히 b/d 의 p90=1393~2215 미달
(단계적 1차 시도).

## Mutation

```diff
- --seq_max_lens "seq_a:64,seq_b:64,seq_c:128,seq_d:128"
+ --seq_max_lens "seq_a:256,seq_b:256,seq_c:512,seq_d:512"
```

기타 H010 byte-identical.

## Memory risk

Attention O(L²). seq 4× → compute 16×.
- domain a/b: O(256²) per layer × 2 blocks.
- domain c/d: O(512²) per layer × 2 blocks → 4× of a/b.
- NS xattn (B, 7, L_total=1536) cross-attention. manageable.

**OOM risk medium** (batch 2048 + seq 512). 사용자 GPU spec 확인 필요.

## Sub-H 후보 (if OOM 또는 degraded)

1. seq 256/256/256/256 (uniform 4×).
2. seq 128/128/256/256 (2× conservative).
3. batch 256 복귀.

## Falsification

| Result | Implication | Next H |
|---|---|---|
| Δ vs H010 ≥ +0.005pt (strong) | **L4 confirmed**. P2 phase entry. | H015 = TWIN/SIM (target-aware retrieval) 또는 더 큰 expansion (1024). |
| Δ ∈ [+0.001, +0.005pt] (measurable) | L4 partial confirmed. | H015 = expansion + retrieval combo. |
| Δ ≤ +0.001pt (noise) | L4 도 ceiling. **L2 (cohort) 만 남음 + paradigm shift**. | H015 = cohort H 또는 backbone replacement. |
| Δ < −0.001pt (degraded) | seq 늘어 학습 어려움 (긴 seq noise). | sub-H 1: 256 uniform. |
| OOM | sub-H 1/2/3. | seq 256 uniform 또는 batch 줄임. |

## Files

| File | Diff vs H010/upload/ | Role |
|---|---|---|
| `run.sh` | 변경 (seq_max_lens 1줄) | Entry point |
| `model.py` | byte-identical | Model |
| `train.py` | byte-identical | CLI |
| `infer.py` | byte-identical | Inference |
| `trainer.py` | byte-identical | Train loop |
| `dataset.py` | byte-identical | Data |
| `local_validate.py` | byte-identical | G1–G6 |
| `make_schema.py` | byte-identical | §18.5 |
| `utils.py` | byte-identical | helpers |
| `ns_groups.json` | byte-identical | NS ref |
| `requirements.txt` | byte-identical | deps |
| `README.md` | 변경 | H014 정체성 |

총 12 files (1 changed code: run.sh + this README).

## Repro pin (post-launch)
- git_sha: TBD.
- config_sha256: TBD.
