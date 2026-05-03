# H016_oof_future_redefine — Technical Report

> Single mutation — **OOF 재정의**: random user holdout → label_time
> future-only. cohort drift 의 다른 form (Frame C). H010 mechanism + envelope
> byte-identical, dataset.py 의 split logic 만 변경.

## Mechanism (vs H010 baseline)

| | H010 (random_user) | H016 (future_label_time) |
|---|---|---|
| OOF 정의 | random 10% users 의 모든 rows | rows with `label_time >= quantile(label_time, 0.9)` |
| Train | non-OOF rows with `label_time < cutoff` | rows with `label_time < quantile(label_time, 0.85)` |
| Valid | non-OOF rows with `label_time >= cutoff` | rows with `0.85q <= label_time < 0.9q` |
| OOF user pool | random 10% (paired Δ rigor) | 모든 user (분포 미래만) |

## Why this variant (Frame C — OOF 재정의)

H015 (recency loss weighting) 와 다른 attack:
- **H015**: train 의 loss 함수 변경 → train 이 platform 에 closer 되도록.
- **H016**: OOF 의 정의 변경 → OOF 가 platform proxy 됨 → 측정 정합성 자체 회복.

9 H 의 OOF stable / Platform 변동 일관 패턴이 진짜 "OOF measure 가 platform
과 다른 것" 이라면, OOF 자체 재정의가 더 fundamental fix.

## Critical caveat — paired Δ baseline 깨짐

H016 의 OOF 정의가 H010~H015 와 다름:
- prior H 들 OOF AUC 비교 **invalid**.
- 단 **Platform AUC 비교는 valid** (eval data 동일).

해석:
- Platform AUC: H016 vs H010 corrected (0.837806) paired Δ 비교 가능.
- OOF AUC: H016 의 새 OOF 분포에서 측정값 — prior H 와 다른 base.
- **결과 해석 시**: Δ Platform 만 cohort gap 의 진짜 정체 검증.

## Diff vs H010/upload/

| File | Diff | Role |
|---|---|---|
| `run.sh` | 변경 (`--oof_split_type future_label_time` + 2 default 명시 bake) | Entry point |
| `dataset.py` | _RowFilter (oof_cutoff field) + split_parquet_by_label_time + get_pcvr_data_v2 (oof_split_type 분기) | Data |
| `train.py` | argparse 1 + get_pcvr_data_v2 1 key | CLI |
| `model.py` | byte-identical | Model |
| `infer.py` | byte-identical | Inference |
| `trainer.py` | byte-identical | Train loop |
| `utils.py` | byte-identical | helpers |
| `local_validate.py` | byte-identical | G1–G6 |
| `make_schema.py` | byte-identical | §18.5 |
| `ns_groups.json` | byte-identical | NS ref |
| `requirements.txt` | byte-identical | deps |
| `README.md` | 변경 | H016 정체성 |

총 12 files (3 changed code + run.sh + README).

## Falsification (H010 corrected anchor 0.837806 기준)

| Result | Implication |
|---|---|
| Δ Platform ≥ +0.005pt | OOF 재정의 가 cohort drift fundamental 해결 → train-eval distribution 정합성 회복 → mechanism 진짜 효과 가시화. anchor = H016. |
| Δ Platform ∈ [+0.001, +0.005pt] | partial. OOF measure 정합성 일부 회복. |
| Δ Platform ≤ +0.001pt (noise) | OOF 정의 변경도 cohort drift 못 풀림 → cohort 가 mechanism class hard ceiling, paradigm shift mandatory. |
| Δ Platform < −0.001pt | OOF 새 정의가 train cohort 줄여 (train_ratio 효과) → 학습 약화. |

## Triple-H setup (H015 + H016 + H017 동시)

L2 multi-form attack:
- H015 = recency linear weighting [0.5, 1.5] (train 에 recency emphasis).
- H016 = OOF 재정의 (label_time future-only — measure proxy 회복).
- H017 = recency exp decay (form 변경, sub-form of H015).

셋 다 noise → cohort drift = paradigm 안 hard ceiling 매우 강한 confirm
→ H018 = backbone replacement 정당화 강력.

## Reproducibility

- seed 42, batch 2048 + lr 1e-4 (H010 envelope, paired 정합성 maximize).
- mechanism (H010 + H008) byte-identical.
- single envelope-style mutation (split definition only).

## Repro pin (post-launch)
- git_sha: TBD.
- config_sha256: TBD.
