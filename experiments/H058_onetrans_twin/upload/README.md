# H058_onetrans_twin — Technical Report

> **PARADIGM SHIFT (backbone class)**: 14 H 모두 HyFormer 위 mutation. H058 =
> OneTrans backbone (single-stream attention) + TWIN retrieval.
> H026 (OneTrans 단독, TWIN 없이) val 0.8330 = HyFormer 0.8372 보다 underperformed.
> H058 = OneTrans + TWIN 으로 H026 underperform 의 진짜 원인이 backbone class
> 인지 missing TWIN 인지 분리 측정.

## 1. Hypothesis & Claim
- Hypothesis: H026 OneTrans underperform 의 원인이 missing TWIN 이라면, OneTrans+TWIN 이 H019 (HyFormer+TWIN) 와 비슷하거나 더 높음. backbone class 자체 문제라면 H026 처럼 underperformed.
- Falsifiable: Δ vs H019 platform (0.839674) ≥ +0.001pt → OneTrans+TWIN 가 HyFormer+TWIN 보다 강함, paradigm shift confirm.

## 2. Mutation
- run.sh: `--backbone hyformer (default)` → `--backbone onetrans`.
- 모든 .py file byte-identical to H019.
- TWIN, NS xattn, DCN-V2 fusion 모두 OneTrans 와 호환 (PCVRHyFormer 가 두 backbone 모두 지원).

## 3. Decision tree

| Δ vs H019 platform (0.839674) | 의미 | Action |
|---|---|---|
| ≥ +0.001pt | OneTrans+TWIN > HyFormer+TWIN — backbone shift PASS | H058 anchor, OneTrans family sub-H |
| (-0.001, +0.001pt] | OneTrans+TWIN ≈ HyFormer+TWIN — TWIN dominates | H026 underperform = missing TWIN, backbone class 무관 |
| > H026 (0.8330 val) but < H019 | OneTrans + TWIN 가 OneTrans 단독 보다 lift, but HyFormer+TWIN 미달 | TWIN add 가 큰 lift, backbone 부분도 일부 영향 |
| << H019 (≈ H026) | OneTrans backbone class 가 본질 문제 | retire OneTrans family |

## 4. Files
| File | H019 대비 | Purpose |
|---|---|---|
| `run.sh` | + `--backbone onetrans` | Entry |
| `README.md` | new | Doc |
| 모든 .py | byte-identical (md5 verify) | unchanged |

trainable params: H019 동일 (단 OneTrans backbone 의 layer 수에 따라 마이너 차이).

## 5. Carry-forward
- §17.2 CLI only mutation.
- §17.4: backbone_replacement re-entry (H026 first-touch + TWIN add justified).
- §0.5 paradigm shift: 14 H HyFormer-bound → OneTrans first valid TWIN combo.
- §18.7 + §18.8 H019 carry.
