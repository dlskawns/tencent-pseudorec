# H033_twin_combined — Technical Report

> CLAUDE.md §17.2 stacking sub-H — H020 (learnable GSU) + H021 (per-domain
> top_k) 동시 적용. H019 (TWIN paradigm shift, cloud measurable PASS 0.839674)
> 위 2 axis 동시 mutation. control = H019 paired Δ.
> §17.2 strict reading 위반 — challengers.md 에 "H020+H021 모두 PASS 가정 하의
> 직접 stack 검증 H" 정당화 명시. 결과 해석은 H020/H021 결과 회수 후 conditional.
> §17.4 retrieval_long_seq re-entry (4회 연속) RE_ENTRY_JUSTIFIED.

## 1. Hypothesis & Claim
- Hypothesis: **H033_twin_combined** = H020 ∘ H021.
- Stacking sub-H of H019:
  - H020 mechanism: GSU parameter-free → learnable projection (`twin_learnable_gsu=True`).
  - H021 mechanism: top_k uniform=64 → per-domain `{seq_a:64, seq_b:64, seq_c:64, seq_d:96}` (`--twin_top_k_per_domain "64,64,64,96"`).
  - 두 axis 직교 → stacking effect 측정 (H020 Δ + H021 Δ 의 합과 비교).
- Predicted (paired classifications vs H019 cloud actual 0.839674):
  - **super-additive** Δ ≥ +0.005pt → 두 axis 시너지 (각 sub-H 합 보다 큼).
  - **additive** Δ ∈ [+0.001 + max(H020,H021), +0.005pt] → 단순 합.
  - **interference** Δ < max(H020, H021) → axis 간 간섭 (H009 패턴).
  - **noise** Δ ∈ [−0.001, +0.001pt] → 둘 다 NOOP 한 confirm.
- Compute tier: **T2.4 (~3.5h, ~$5-7)**, H019/H020/H021 동급.

## 2. What this code does

H020 base (already has learnable_gsu) + H021 per-domain top_k logic 추가.
TWINBlock class = H020 (learnable_gsu support).
PCVRHyFormer wiring = H021 (int|dict twin_top_k 처리) + H020 (learnable_gsu pass).

`PCVRHyFormer` TWIN block construction:
```python
top_k_per_domain = parse(twin_top_k)  # int → uniform / dict → per-domain
TWINBlock(top_k=top_k_per_domain[d], learnable_gsu=twin_learnable_gsu) for d in (seq_a..d)
```

trainable params 추가: H020 (8K) + H021 (0) = **+8K total**.

## 3. Files
| File | H019 대비 | Purpose |
|---|---|---|
| `run.sh` | + `--twin_learnable_gsu` + `--twin_top_k_per_domain "64,64,64,96"` flags | Entry |
| `train.py` | + 2 argparse + helper `_parse_twin_top_k_arg` + model_args | CLI |
| `model.py` | TWINBlock learnable_gsu (H020) + PCVRHyFormer int|dict wiring (H021) | Model |
| 다른 모든 파일 | byte-identical | unchanged |

## 4. Stacking 정당화 (§17.2 strict reading 위반 인지)

**Strict reading**: 단일 mutation = 1 axis 변경. H033 = 2 axis 동시 → 위반.

**Re-justification**:
- H020 + H021 = H019 mechanism class 안 직교 axis 의 sub-H. paired 비교 framework 이미 challengers.md 정의.
- H020/H021 모두 PASS 시: stacking H 가 *반드시* 필요 (시너지/간섭 측정 위해). single-mutation 룰의 "한 번에 한 변경" 원칙은 *발견* 단계 — *통합* 단계에서는 stacking 정당.
- H020/H021 한 쪽 NOOP 시: H033 결과는 PASS axis + NOOP axis 의 *합산 효과* 로 해석 (NOOP axis 의 음의 기여 측정으로도 가치).
- H020/H021 둘 다 NOOP 시: H033 도 NOOP — retrieval class 전체 saturation 강한 confirm (4 H 누적 evidence).

**Pre-build = insurance**: H020/H021 결과 회수 전 H033 patch 미리 ready → 둘 다 PASS 확인 즉시 cloud submit (round-trip 절약).

## 5. Outputs
- `metrics.json` — `twin_learnable_gsu=true` + `twin_top_k_per_domain` 둘 다 기록.
- `train.log` — `H033 TWIN combined enabled: per-domain K = ... learnable_gsu=True ...` 메시지.

## 6. Carry-forward
- H019/H020/H021 모든 carry-forward inherit (NS xattn, DCN-V2, gate=-2.0, seq 256/256/256/256, batch=1024).
- §17.4 rotation: retrieval_long_seq 4회 연속, RE_ENTRY_JUSTIFIED (challengers.md).
- §18.7/§18.8 H019 carry.
