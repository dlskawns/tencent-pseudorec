# STATUS — TAAC 2026 UNI-REC (last updated 2026-05-10)

> 매 H 진행 상황의 단일 view. INDEX.md (full) / progress.txt (chronological) 보다 빠른 snapshot.

## 1. Champion / Anchor

| | Platform AUC | Notes |
|---|---|---|
| **H057 (TWIN attn aggregator)** | **0.841364** | **현 champion** (H019 위 +0.0017pt, 14 H 만의 첫 PASS) |
| H019 (TWIN baseline) | 0.839674 | prior anchor, multi-seed (43): Val −0.0004 |
| H010 corrected (NS xattn + DCN-V2) | 0.837806 | mid-tier baseline |
| Vanilla H039 (mechanism-stripped) | 0.835426 | floor |
| 0.85 target | 0.85 | gap from H057 = +0.0086pt |

## 2. Current Slot Status

### Cloud training/recently-finished
| H | Mutation | Status |
|---|---|---|
| H019 seed=44 | multi-seed robustness | running |
| H059 | per-user TWIN gate | running (re-submitted after OOM) |
| H060 | TWIN ESU pos emb | running |
| H061 | TWIN candidate Q time enrichment | ready to submit |
| **H062** | **TWIN learnable GSU (low-rank W_q/W_k)** | **BUILT 2026-05-10, ready** |

### Recent platform results
| H | Val | OOF | Platform | Verdict |
|---|---|---|---|---|
| **H057** | 0.83777 | 0.8615 | **0.841364 ✅** | **PASS +0.0017** |
| H056 (concat clsfier) | 0.83773 | 0.8611 | not submitted | Val noise (skipped) |
| H058 (OneTrans+TWIN) | OOM | — | — | OneTrans single-stream too large |
| H055 (cross-domain SSL) | 0.83226 | 0.8576 | val OOM | archived (lambda 0.1 too high) |
| H054 (LambdaRank) | 0.83677 | 0.8611 | not submitted | Val noise |
| H053 (SimCSE) | training | — | — | running |
| H052 (User-Item InfoNCE) | 0.83520 | — | — | likely Val degraded |
| H050 (train_ratio 0.6) | 0.83672 | — | not submitted | Val NOOP, archived |
| H051 (per-pattern dense) | 0.83659 | — | not submitted | Val NOOP, archived |
| H048 (user×item bilinear) | 0.83771 | 0.8614 | **0.835477 ❌** | F-A 극단 사례 (Platform −0.0042) |
| H043 (item-side DCN-V2) | 0.83666 | 0.8607 | infer fail (fixed) | retry pending |
| H042 (KLD prior matching) | 0.83719 | 0.8616 | 0.839388 | NOOP |
| H041 (cold-start dual) | 0.83496 | 0.8601 | not submitted | REFUTED Val/OOF |
| H039 (vanilla baseline) | 0.83677 | 0.8615 | 0.835426 | floor measurement |
| H038 (aux MSE timestamp) | 0.83735 | 0.8623 | 0.839071 | NOOP (axis dead) |

### Built but unsubmitted (Val unknown)
- H044 (DANN v1, ABORTED loss explosion)
- H045 (cross-domain bridge, training)
- H046 (proper DANN v2)
- H047 (per-domain aux heads)
- H049 (NS architecture refinement, item_ns 6 + slot type emb)

## 3. Key Findings (F-N)

### F-A: Val/OOF ↔ Platform divergence (decisive)
- H019 multi-seed: Val noise ±0.0005pt, OOF noise ±0.0006pt
- 모든 H 의 Val Δ < 0.001pt = noise band, ranking signal 무용
- H048 = 가장 극단 사례: Val +0.0005 / OOF +0.0003 → Platform −0.0042pt
- **Platform 만 신뢰 가능 — Val/OOF 단독 verdict 금지**

### F-G: Mechanism stacking lever pattern
- PASS = "새 정보 흐름" 추가 (H010 NS xattn, H019 TWIN, H057 attn aggregator)
- NOOP = 변조/regularization (H011~H018, H038, H041, H042, H050, H051)
- **Single residual rule (from H048 fail)**: TWIN 위 다른 residual ADD 동시 사용 = transfer 깨짐
- **TWIN-internal mutation 1/1 PASS** (H057), 다른 layers 4 H 진행 중

### F-H: TWIN-internal 5 layer 분석
| Layer | H | Direction |
|---|---|---|
| candidate Q (upstream) | H061 | enrichment with time |
| GSU scoring | H062 | param-free → learnable W |
| ESU MHA input | H060 | + positional emb |
| aggregator | H057 ✅ | mean → attention |
| gate | H059 | scalar → per-user |

## 4. Design Rules (lessons-learned)

1. **Single residual rule** (H048 lesson): TWIN 위 다른 residual ADD 금지 — output magnitude 증가 → cohort transfer 깨짐
2. **infer.py flag parity** (H043 lesson): model.py 새 flag 추가 시 infer.py 도 cfg.get() 추가 필수
3. **Val/OOF 단독 verdict 금지** (F-A lesson): noise ±0.0005pt 안의 measurement 의미 없음
4. **TWIN-internal preferred** (H057 lesson): single residual 보존 + TWIN 내부 mutation
5. **uniform → adaptive pattern** (H057 mechanism): per-user / per-domain attention 이 unifrom mean 보다 우월

## 5. Slot 결정 history

| Slot | Action | Reason |
|---|---|---|
| H050 (orig) | train_ratio 0.6 → archived | Val NOOP |
| H051 (orig) | per-pattern dense → archived | Val NOOP |
| H055 | cross-domain SSL → val OOM, archived | lambda too high + axis fail |
| H056 | concat clsfier → not submitted | Val noise after H057 success change priority |
| H058 | OneTrans+TWIN → OOM, archived | OneTrans single-stream too memory-heavy |

## 6. 다음 결정 매트릭스 (post-result)

| H057+H059+H060+H061+H062 결과 | Action |
|---|---|
| 모두 PASS | TWIN-internal 모든 layer lever, stacking sub-H 거대 잠재 |
| 일부 PASS | 그 layer 가 critical, others redundant |
| H057 만 PASS | aggregator unique, retire others |
| 모두 NOOP | TWIN-internal saturated, paradigm shift 필수 |

## 7. Reference

- **Full registry**: `experiments/INDEX.md`, `hypotheses/INDEX.md`
- **Chronological journal**: `progress.txt`
- **Per-H docs**: `hypotheses/H{XXX}/{problem,transfer,predictions,verdict}.md`
- **Code**: `experiments/H{XXX}/upload/`
