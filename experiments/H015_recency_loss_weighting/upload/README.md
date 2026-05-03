# H015_recency_loss_weighting — Technical Report

> Single mutation — **recency-aware loss weighting (per-batch linear by
> label_time)**. H010 mechanism (NS xattn) + H008 mechanism (DCN-V2 fusion)
> + envelope byte-identical. Cohort drift mitigation (4-layer ceiling
> diagnosis L2 직접 검증).

## Why now — L2 가 마지막 가설

9 H 누적 ceiling 0.82~0.838. 4-layer diagnosis 종료:
- L1 (hyperparameter): H013 REFUTED.
- L3 (NS xattn sparse): H011/H012/H013/H014 누적 retire.
- L4 (truncate): H014 REFUTED.
- **L2 (cohort drift): 9 H 의 OOF stable / Platform 변동 일관 패턴 — 마지막
  unexplored 가설**.

→ H015 = L2 직접 attack.

## Mechanism

```python
# In trainer.py _train_step:
if self.use_recency_loss_weighting:
    label_time = device_batch['label_time'].float()
    lt_min, lt_max = label_time.min(), label_time.max()
    pct = (label_time - lt_min) / (lt_max - lt_min)  # [0, 1]
    weights = 0.5 + 1.0 × pct                         # [0.5, 1.5], mean = 1.0
    loss_per = F.binary_cross_entropy_with_logits(logits, label, reduction='none')
    loss = (weights × loss_per).mean()
```

Per-batch linear: oldest sample weight 0.5, newest 1.5. **Mean weight = 1.0**
(loss scale 보존). Recent sample 이 더 큰 gradient 받음 → train cohort 가
platform eval distribution (미래 시점) 에 closer.

## Files

| File | Diff vs H010/upload/ | Role |
|---|---|---|
| `run.sh` | 변경 (5 flags 추가: 3 H015 + 2 H010 default 명시) | Entry point |
| `dataset.py` | 변경 (label_time batch dict 노출, 2 줄 추가) | Data |
| `trainer.py` | 변경 (__init__ 3 args + _train_step weighting branch ~20줄) | Train loop |
| `train.py` | 변경 (argparse 3 + Trainer 3 keys) | CLI |
| `model.py` | byte-identical | Model architecture |
| `infer.py` | byte-identical | Inference (loss 계산 안 함) |
| `utils.py` | byte-identical | helpers |
| `local_validate.py` | byte-identical | G1–G6 |
| `make_schema.py` | byte-identical | §18.5 |
| `ns_groups.json` | byte-identical | NS ref |
| `requirements.txt` | byte-identical | deps |
| `README.md` | 변경 | H015 정체성 |

총 12 files (4 changed code + README).

## Falsification

| Result | Implication | Next H |
|---|---|---|
| Δ vs H010 ≥ +0.005pt (strong) | **L2 confirmed (strong)**. Cohort drift = ceiling 의 진짜 정체. | H016 = recency variants (exp decay, larger range) 또는 OOF 재정의. |
| Δ ∈ [+0.001, +0.005pt] (measurable) | L2 partial. | H016 = recency + cohort embedding combo. |
| Δ ∈ (−0.001, +0.001pt] (noise) | Cohort drift 도 가설 약함. **마지막 layer 도 retire → paradigm shift mandatory** (backbone replacement 또는 retrieval). | H016 = backbone replacement 또는 TWIN/SIM. |
| Δ < −0.001pt (degraded) | Recency weighting 이 학습 disrupt. | sub-H = weight range 좁힘 [0.7, 1.3] 또는 quadratic decay. |

## Why per-batch (not per-dataset)

- per-batch: shuffle 영향 없음. Implementation 단순. Single mutation.
- per-dataset (전체 train min/max 미리 계산): batch 마다 같은 percentile,
  더 stable 단 implementation 복잡 (DataLoader sampler 변경 필요).
- per-batch 선택: minimum viable form. PASS 시 sub-H 로 per-dataset.

## Mean weight conservation

Linear `[0.5, 1.5]` symmetric around 1.0 → mean weight = 1.0 → loss scale
보존. lr / optimizer 영향 없음. paired Δ 비교 confound 작음.

## Reproducibility

- seed 42, label_time split + 10% OOF.
- batch 2048 + lr 1e-4 (사용자 prior H regime, paired 정합성).
- mechanism (H010 + H008) byte-identical.
- single mutation (loss weighting only).

## Repro pin (post-launch)
- git_sha: TBD.
- config_sha256: TBD.
