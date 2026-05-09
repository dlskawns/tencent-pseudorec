# H039_no_history_baseline — Technical Report

> **Diagnostic H — NO new mechanism**. CLAUDE.md §0.5 data-signal-driven 진단.
> 14 H 동안 mechanism axis (TWIN/DCN-V2/NS xattn/HSTU/multi-task) 모두 같은
> 영역 (val 0.834~0.837) 으로 수렴 → mechanism 의 absolute floor 측정 필요.
> Mutation: H019 의 모든 mechanism flag 제거 (DCN-V2, NS xattn, TWIN).
> 코드 변경 0 — `run.sh` flag 만 변경. model.py / dataset.py / trainer.py /
> infer.py 모두 H019 byte-identical (md5 verified).

## 1. Hypothesis & Claim

- **Hypothesis**: H039 = "absolute floor" of PCVRHyFormer + RankMixer NS tokenizer + standard transformer encoder, *without* any of the 14 H's mechanism additions.
- **Falsifiable claim**: H039 platform AUC > 0.835 → 14 H mechanism work is essentially NOOP, paradigm pivot mandatory.
- **Compute tier**: T2.4 (~3.5h, ~$5-7).

## 2. Decision tree (post-result)

| H039 platform AUC | Interpretation | Action |
|---|---|---|
| ≥ 0.838 | mechanism work essentially NOOP | abandon mechanism axis, pivot to capacity / loss / distillation / problem reformulation |
| [0.825, 0.838) | mechanism contributed marginally | continue but recalibrate expectations |
| < 0.825 | mechanism contributed meaningfully | mechanism axis valid, but within saturated band — need bigger lever (capacity, distillation) |

H019 platform = 0.83967. Δ vs H039 quantifies **total mechanism contribution** of 14 H worth of work.

## 3. What this code does

H019 anchor 인프라 그대로 + **mechanism flags 모두 제거**:
- ❌ `--use_twin_retrieval` (no TWIN GSU+ESU)
- ❌ `--use_ns_to_s_xattn` (no NS→S xattn)
- ❌ `--fusion_type dcn_v2` (default 'rankmixer')
- ❌ `--log_attn_entropy` (mechanism diagnostic 불필요)

남은 것 = organizer baseline equivalent + 우리 harness 개선:
- ✅ PCVRHyFormer + RankMixer NS tokenizer + per-domain transformer encoder (organizer-default)
- ✅ label_time split + OOF holdout (우리 harness)
- ✅ §18.7 label_time fill_null + §18.8 emit_train_summary (우리 harness)
- ✅ batch=1024, lr=1e-4 (H019 envelope, organizer default 와 다름)

## 4. Files
| File | H019 대비 | Purpose |
|---|---|---|
| `run.sh` | 변경 (mechanism flags 제거 + 주석 + EXP_ID) | Entry point |
| 다른 모든 .py 파일 | byte-identical (md5 verified) | unchanged |

## 5. Carry-forward
- §18.7 label_time fill_null: H015 carry.
- §18.8 emit_train_summary: H019 carry, exp_id 만 H039 로.
- §17.2 single mutation: "remove all mechanism flags". model code 변경 0.
- §17.4 rotation: diagnostic H, NEW first-touch (AUTO_JUSTIFIED).
