# H013_hyperparameter_calibration — Technical Report

> CLAUDE.md §17.2 parametric mutation **justified by H012 F-2** (hyperparameter
> measurement bias 노출). H010 mechanism (NS→S xattn) + H008 mechanism (DCN-V2
> fusion) + envelope **byte-identical** — model.py / train.py / infer.py /
> trainer.py 모두 변경 없음. **run.sh 만 수정**.

## Purpose — measurement diagnostic

7개 H 누적 ceiling Platform AUC 0.82~0.8408. 4-layer ceiling diagnosis 중
**Layer 1 (hyperparameter regime)** 검증. 사용자 batch_size=2048 override
+ default lr=1e-4 → linear scaling rule 미적용 → effective lr 1/8 underpowered
가설.

## Mutation (run.sh only)

| Hyperparameter | H010 (prior) | H013 (calibrated) | Reason |
|---|---|---|---|
| `--batch_size` | 256 (default, user override 2048) | **2048 (explicit bake)** | 사용자 override 명시화. metrics.json sanity gate 신뢰성. |
| `--lr` | 1e-4 (default) | **8e-4** | Linear scaling rule (Goyal et al. 2017): batch 8× → lr 8×. |
| `--num_workers` | 2 | **4** | IO bound 완화 (H012 F-5 wall 단축 신호). Taiji 안전 범위 (deadlock 위험 8 미만). |
| `--buffer_batches` | 4 | **8** | 큰 배치 (2048) 에서 IO 부하 완화. |

기타 모든 flags H010 byte-identical.

## Falsification (decision tree)

| Result | Implication | Next H |
|---|---|---|
| Δ vs H010 ≥ +0.005pt (strong) | **Ceiling = hyperparameter artifact**. 모든 prior H 재해석 의무. | H010+ 전체 ranking 재평가 + 다음 mechanism H |
| Δ ∈ [+0.001, +0.005pt] (measurable) | 부분적 hyperparameter, 부분적 mechanism. | mixed analysis + Track B |
| Δ ∈ (−0.001, +0.001pt] (noise) | Mechanism ceiling 진짜. lr 적정. | **Track B (long-seq P2)** 또는 cohort drift 처리 |
| Δ < −0.001pt (degraded) | lr 너무 높음 — divergence 또는 instability. | sub-H: lr 4e-4 (절반 scaling) |
| NaN abort | lr 8e-4 너무 큼 → divergence. | sub-H: lr 4e-4. |

## Files

| File | Diff vs H010/upload/ | Role |
|---|---|---|
| `run.sh` | 변경 (4 hyperparameter flags) | Entry point |
| `train.py` | byte-identical | CLI driver |
| `model.py` | byte-identical | Model architecture |
| `infer.py` | byte-identical | §18 인프라 |
| `trainer.py` | byte-identical | Train loop |
| `dataset.py` | byte-identical | Data |
| `local_validate.py` | byte-identical | G1–G6 |
| `make_schema.py` | byte-identical | §18.5 |
| `utils.py` | byte-identical | helpers |
| `ns_groups.json` | byte-identical | NS ref |
| `requirements.txt` | byte-identical | deps |
| `README.md` | 변경 | H013 정체성 |

총 12 files (1 changed: run.sh + this README).

## Why parametric mutation is OK here (§17.2 정당화)

- §17.2: "One-mutation-per-experiment, structural not parametric. 하이퍼
  파라미터 (focal γ, lr, dropout, init scale) 튜닝은 P2까지 명시 금지".
- 본 H 위반 가능성. 단:
  - **Justification = measurement integrity**. H012 F-2 carry-forward.
  - 4 changes 모두 single concern (training efficiency under batch 2048).
  - Linear scaling rule 은 standard practice (Goyal et al. 2017), arbitrary
    tuning 아님.
  - 결과 분기에서 mechanism limit 가 진짜로 입증되면 prior H 의 mechanism
    ranking 가 valid (mechanism 추가 H 정당).
- card.yaml `claim_scope` + `mutation_class.rotation` 에 명시.

## Reproducibility

- seed 42, label_time split + 10% OOF.
- H010 mechanism (`--use_ns_to_s_xattn --ns_xattn_num_heads 4`) byte-identical.
- H008 mechanism (`--fusion_type dcn_v2 --dcn_v2_num_layers 2 --dcn_v2_rank 8`)
  byte-identical.
- 4 hyperparameter changes baked in run.sh.

## Repro pin (post-launch)
- git_sha: TBD.
- config_sha256: TBD (run 후 metrics.json 의 batch_size=2048, lr=8e-4 확인).
