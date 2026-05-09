# H033 — Predictions

> Stacking sub-H — H020 ∘ H021 동시 mutation. control = H019.
> Conditional cloud submit (H020/H021 결과 의존).

## Outcome distribution

| Outcome | Δ vs H019 | probability | mechanism implication |
|---|---|---|---|
| super-additive | ≥ max(H020,H021) + 0.003pt | ~15% | 두 axis 시너지, anchor = H033 |
| additive | ≈ H020 Δ + H021 Δ | ~25% | axis 독립, 단순 합 |
| sub-additive (interference) | < max(H020,H021) | ~25% | H009 패턴 — axis 충돌 |
| noise | (-0.001, +0.001pt] | ~30% | 둘 다 NOOP, retrieval class saturation |
| degraded | < -0.001pt | ~5% | over-capacity / instability |

(H020/H021 둘 다 PASS 가정 conditional 분포. H020/H021 결과 회수 후 재계산 필요.)

## P1 — Code-path success

- Risk: H020 + H021 stacking 의 unforeseen interaction. 특히 PCVRHyFormer 의 twin_top_k dict logic 과 TWINBlock learnable_gsu 의 init order 충돌 가능.
- Failure mode: dict iteration order (Python 3.7+ insertion order 보장이라 OK) 또는 ESU input 의 norm scale mismatch.
- Mitigation: T0 sanity 에서 TWINBlock(K=96, learnable_gsu=True) forward NaN-free 검증 완료.

## P2 — Primary lift (vs H019, §17.3 binary)

| 비교 | classification | Δ 임계 |
|---|---|---|
| Δ vs H019 (0.839674) | super-additive | ≥ max(H020, H021) + 0.003pt |
| Δ vs H019 | additive | ≈ H020 Δ + H021 Δ (±0.001) |
| Δ vs H019 | sub-additive | < max(H020, H021) |
| Δ vs H019 | noise | (-0.001, +0.001pt] |
| Δ vs H019 | degraded | < -0.001pt |

**Secondary**:
- Δ vs H010 corrected (0.837806) = +0.001868 (H019) + H033 Δ.
- Triple-paired: H020 + H021 + H033 → axis 독립성/시너지 정량.

## P3 — Stacking mechanism 검증

- **Synergy check**: H020 Δ + H021 Δ 의 합 vs H033 Δ.
  - 합 ≈ H033 Δ → axis 독립.
  - 합 < H033 Δ → 시너지 (super-additive).
  - 합 > H033 Δ → 간섭 (sub-additive).
- **GSU score distribution**: H020 (learnable) + per-domain K (H021) 시 domain seq_d 의 score 분포 measurement.
- **ESU attention entropy per-domain**: domain seq_d 의 entropy 가 H019/H020/H021 와 어떻게 다른지.

## P4 — §18 인프라 PASS

- §18.5: schema 변경 없음.
- §18.6 dataset-inference-auditor: H020/H021 와 동일 변경 범위 (model.py wiring + train.py + run.sh). audit 범위 좁음.
- §18.7/§18.8: H019 carry.

## P5/P6 — val/OOF ↔ Platform gap

- F-A baseline: −0.003pt. H016 redefined OOF baseline: −0.0036pt. H033 expected: similar.

## P7 — Cost cap audit

- T2.4 ~3.5h × $5-7. campaign cap $100 친화.
- **Conditional submit**: H020/H021 결과 회수 후 결정 (둘 다 PASS 시만 권장).

## Decision tree (post-result)

| H020 결과 | H021 결과 | H033 권장 | H033 결과별 다음 |
|---|---|---|---|
| PASS | PASS | **즉시 cloud submit** | super-additive → anchor = H033, additive → 둘 axis 독립 confirm |
| PASS | noise | submit 보류 (interpretation ambiguous) | — |
| noise | PASS | submit 보류 | — |
| noise | noise | submit 불필요 (NOOP 확정) | H034 (capacity) 또는 cohort pivot |
| degraded | * | submit 불필요 | H020 의 issue 격리 |
| * | degraded | submit 불필요 | H021 의 issue 격리 |

## Falsification claim

> "H020 (learnable GSU) + H021 (per-domain top_k) stacking 이 H019 anchor 위 Δ ≥ max(H020 Δ, H021 Δ) + 0.003pt super-additive lift 만든다."

거짓 → axis 독립 (additive) 또는 간섭 (sub-additive). axis 통합 가치 작음.

## Cost cap escape valve

H020/H021 cloud actual cost 회수 후 결정. 누적 + H033 estimate > $80 면 사용자 confirm.
