# H021 — Upload Package Patch Spec

> Patch spec only — full upload/ construction deferred until H019 cloud
> result 검토 + H020 동시 build. Generated 2026-05-06.

## 1. Base & sequence

- **Base**: `experiments/H019_twin_long_seq_retrieval/upload/` byte-by-byte copy.
- **Mutation**: TWINBlock 의 top_k 를 도메인별로 다르게 instantiate. **TWINBlock class 자체 byte-identical** (이미 top_k 를 init param 으로 받음).
- **Carry-forward**: H019 의 모든 코드 byte-identical except 4 files (model.py + train.py + run.sh + README.md).
- **§17.2 single mutation**: top_k policy 의 uniform → domain-aware. TWINBlock 내부 / GSU / ESU / aggregator / gate / num_heads / seq_max_lens / batch / NS xattn / DCN-V2 stack 전부 H019 byte-identical. trainable params 추가 0.

## 2. Files to modify (3 .py + run.sh + README)

| File | Δ vs H019 | Role |
|---|---|---|
| `model.py` | + ~10 lines (PCVRHyFormer __init__ wiring 변경, TWINBlock class 변경 없음) | Model |
| `train.py` | + ~7 lines (argparse + parsing logic) | CLI |
| `run.sh` | + 1 line (`--twin_top_k_per_domain "64,64,64,96"`) | Entry |
| `README.md` | rewrite — H021 identity | Doc |
| `dataset.py` / `infer.py` / `make_schema.py` / `utils.py` / `local_validate.py` / `trainer.py` / `ns_groups.json` / `requirements.txt` | byte-identical | unchanged |

**중요**: H020 와의 차이 = H020 은 TWINBlock class 자체에 W_q/W_k Linear 추가 (graph 변경, +8K params). H021 은 TWINBlock class **byte-identical**, PCVRHyFormer wiring 만 변경 (graph 변경 0, params 추가 0).

## 3. model.py patch — PCVRHyFormer wiring (TWINBlock class 변경 없음)

**Modify** `class PCVRHyFormer.__init__` (around line 1789-1793 in H019 model.py — argument signature):

```python
# 기존 (H019):
twin_top_k: int = 64,

# H021:
twin_top_k: Union[int, Dict[str, int]] = 64,
```

**Modify** `PCVRHyFormer.__init__` (around line 2038-2049 — instantiation loop):

```python
# H021 — per-domain top_k 지원
if use_twin_retrieval:
    if isinstance(twin_top_k, int):
        # H019 backward compat: uniform K
        top_k_per_domain = {d: twin_top_k for d in ('a', 'b', 'c', 'd')}
    else:
        # H021: per-domain dict
        top_k_per_domain = twin_top_k
        assert set(top_k_per_domain.keys()) == {'a', 'b', 'c', 'd'}, \
            f"twin_top_k dict keys must be {{a,b,c,d}}, got {set(top_k_per_domain.keys())}"

    self.twin_blocks = nn.ModuleDict({
        f'twin_{domain}': TWINBlock(
            d_model=d_model,
            num_heads=twin_num_heads,
            top_k=top_k_per_domain[domain],     # H021 per-domain
        )
        for domain in ('a', 'b', 'c', 'd')
    })
    self.twin_aggregator = TwinRetrievalAggregator(
        d_model=d_model, gate_init=twin_gate_init
    )
    print(
        f"H021 TWIN per-domain K: "
        f"a={top_k_per_domain['a']} b={top_k_per_domain['b']} "
        f"c={top_k_per_domain['c']} d={top_k_per_domain['d']} "
        f"num_heads={twin_num_heads} gate_init={twin_gate_init}"
    )
```

**Required import** (top of file):
```python
from typing import Union, Dict
```

**TWINBlock class**: **byte-identical to H019** (line 1323-1369). 변경 없음.

## 4. train.py patch

**Argparse addition** (after `--twin_top_k`):
```python
parser.add_argument('--twin_top_k_per_domain', type=str, default=None,
                    help='H021: comma-separated per-domain K, e.g. "64,64,64,96" for a,b,c,d. '
                         'If set, overrides --twin_top_k.')
```

**Parsing logic** (before model construction):
```python
# H021 — per-domain top_k 파싱
if args.twin_top_k_per_domain:
    ks = [int(x.strip()) for x in args.twin_top_k_per_domain.split(',')]
    assert len(ks) == 4, \
        f"--twin_top_k_per_domain must be 4 ints (a,b,c,d), got {len(ks)}"
    twin_top_k_arg = {d: k for d, k in zip(('a', 'b', 'c', 'd'), ks)}
    print(f"H021 enabled: per-domain K = {twin_top_k_arg}")
else:
    twin_top_k_arg = args.twin_top_k       # H019 backward compat (uniform int)
```

**Model construction** (pass parsed arg):
```python
model = PCVRHyFormer(
    ...,
    use_twin_retrieval=args.use_twin_retrieval,
    twin_top_k=twin_top_k_arg,             # H021 — int 또는 dict
    twin_num_heads=args.twin_num_heads,
    twin_gate_init=args.twin_gate_init,
)
```

**§18.8 SUMMARY block** (carry-forward from H019):
- exp_id 변경: `H019_twin_long_seq_retrieval` → `H021_per_domain_top_k`.
- 다른 fields 그대로.

## 5. run.sh patch

```bash
# H021 baked args (vs H019: + --twin_top_k_per_domain):
--use_twin_retrieval
--twin_top_k_per_domain "64,64,64,96"        # H021 — domain d만 96
--twin_num_heads 4
--twin_gate_init -2.0
--oof_redefine future_only                   # H016 carry
--seq_max_lens 256 256 256 256               # H019 carry (sweep saturation 영역)
--batch_size 1024                            # H019 carry
+ all H010 mechanism flags 그대로 (NS xattn, DCN-V2)
```

**중요**: `--twin_top_k 64` 와 같이 쓰면 `--twin_top_k_per_domain` 이 우선 (parsing logic 명시).

## 6. README.md (rewrite)

```markdown
# H021 — per_domain_top_k

> TWIN sub-H (Tencent 2024) — top_k policy 의 uniform → domain-aware.
> per-domain K = {a:64, b:64, c:64, d:96}. domain d 만 50% 확장.
> §3.5 quantitative motivation: domain d p90=2215 = K=64 의 top 2.9%
> (4 도메인 중 가장 under-served).
>
> Mutation: top_k policy 의 single mutation (uniform → domain-aware).
> TWINBlock class 변경 없음 (이미 top_k 를 init param 으로 받음).
> PCVRHyFormer wiring + train.py argparse + run.sh 만 변경.
> trainable params 추가 0.
>
> ESU / GSU / aggregator / gate=-2.0 / num_heads=4 / seq_max_lens
> 256/256/256/256 / batch 1024 byte-identical to H019.
>
> Control: H019 (champion, cloud actual 0.839674). Paired Δ vs H019.
> Parallel sub-H: H020 (scoring axis, learnable GSU) — paired 비교 framework.
>
> Carry-forward: H016 redefined OOF (future-only) default. §18.8 SUMMARY
> block emit (exp_id 만 H019 → H021 변경).
```

## 7. Local sanity check (§17.5 code-path verification only)

```bash
.venv-arm64/bin/python train.py \
  --num_epochs 1 \
  --train_ratio 0.05 \
  --use_twin_retrieval \
  --twin_top_k_per_domain "64,64,64,96" \   # H021 mutation
  --twin_gate_init -2.0 \
  --oof_redefine future_only \
  --seq_max_lens 256 256 256 256 \
  --batch_size 256                          # local downscale
```

Expected:
- Log line (parsing): `H021 enabled: per-domain K = {'a': 64, 'b': 64, 'c': 64, 'd': 96}`.
- Log line (model): `H021 TWIN per-domain K: a=64 b=64 c=64 d=96 num_heads=4 gate_init=-2.0`.
- 4 TWINBlock instantiation: `self.twin_blocks['twin_a'].top_k == 64`, ..., `self.twin_blocks['twin_d'].top_k == 96` 확인.
- domain d forward (history_len ≤ 256, K=96) shape (B, D), NaN-free.
- top-K filter: `min(self.top_k, L)` — L=128 시 K=min(96,128)=96, L=64 시 K=min(96,64)=64 (자동 cap).
- §10.6 sample budget: H019 동일 (parameter-free).
- §18.8 SUMMARY block printed at end (1 epoch row).
- `local_validate.py` G1–G6 5/5 PASS.

**Ablation diff sanity** (H019 vs H021 forward output):
```python
import torch
from model import PCVRHyFormer

torch.manual_seed(0)
# ... config setup ...

# H019 (uniform K=64)
model_h019 = PCVRHyFormer(
    ..., use_twin_retrieval=True, twin_top_k=64, ...
).eval()

# H021 (per-domain K)
model_h021 = PCVRHyFormer(
    ..., use_twin_retrieval=True, twin_top_k={'a':64, 'b':64, 'c':64, 'd':96}, ...
).eval()

# weight init RNG offset → 출력 차이 비교 가능
out_h019 = model_h019(batch)
out_h021 = model_h021(batch)
diff = (out_h019 - out_h021).abs().max().item()
print(f"ablation diff: {diff:.4f}")   # > 0.001 expected (domain d 경로 다름)
print(f"NaN check: {torch.isnan(out_h021).any().item()}")  # False expected
```

**Per-domain K 확인 sanity**:
```python
for domain in ('a', 'b', 'c', 'd'):
    block = model_h021.twin_blocks[f'twin_{domain}']
    expected_k = {'a':64, 'b':64, 'c':64, 'd':96}[domain]
    assert block.top_k == expected_k, f"domain {domain}: expected K={expected_k}, got {block.top_k}"
    print(f"domain {domain}: K={block.top_k} ✓")
```

## 8. dataset-inference-auditor invocation (§18.6)

After upload/ package built:

```
Agent(subagent_type="dataset-inference-auditor",
      prompt="Audit experiments/H021_per_domain_top_k/upload/. prior_h=H019. Confirm dataset.py / infer.py / make_schema.py / TWINBlock class 부분 byte-identical to H019 (no schema change, no model graph change). model.py change scope = PCVRHyFormer __init__ 의 twin_top_k arg type (int → int|dict) + instantiation loop wiring 만. train.py: --twin_top_k_per_domain argparse + parsing logic. §18.7 carry (label_time fill_null) preserved. §18.8 SUMMARY block exp_id = H021_per_domain_top_k.")
```

Expected: PASS — H019/H020 보다 변경 범위 더 좁음 (TWINBlock class 자체 변경 없음).

BLOCK 시 fix → re-audit. PASS 받기 전 cloud upload 금지.

## 9. config_sha256 + git_sha (§4 reproducibility)

After patch applied + local sanity PASS:
- `git rev-parse --short=7 HEAD` → save to card.yaml.
- `sha256sum train_config.json` → save to card.yaml `config_sha256`.
- both written to SUMMARY block automatically via env vars.

## 10. Cost cap pre-launch audit (§17.6)

Before T2.4 launch:
- H019/H020 cloud actual cost 회수 (T2.4 ~$5-7 each expected).
- 누적 cost = (Subset A H022/H028~H031 actual) + H019 actual + H020 actual + H021 estimate.
- per-job cap T2.4 ~$10 / per-campaign cap $100 audit.
- 초과 시 사용자 confirm 필수.

## 11. Paired 비교 framework (H020 와 동시 submit 권장)

H021 과 H020 은 직교 axis sub-H — 동시 cloud submit 시 paired 비교 framework 효과 극대화:

| H020 결과 | H021 결과 | 다음 H 결정 |
|---|---|---|
| strong | strong | H022 = stack (H020 + H021 동시 mutation, challengers.md 단일성 정당화 필요) |
| strong | measurable+ | anchor = H020 + H021, H022 = stack 검증 |
| strong | noise/degraded | anchor = H020, scoring axis dominant, H022 = H020 sub-H (projection dim 확장) |
| measurable | measurable | H022 = stack 검증, 둘 다 약 effect 누적 |
| measurable | noise | anchor = H020 약 effect, H022 = scoring axis 더 깊이 |
| noise | strong | anchor = H021, quantity axis dominant, H022 = 정량안 (a:96 d:128) |
| noise | measurable | anchor = H021 약 effect, H022 = quantity axis 더 깊이 |
| noise | noise | retrieval class 전체 saturation → ESU(A3) 또는 cohort(C) pivot |
| degraded | * | H020 의 projection 자체 issue 격리 → H021 결과 단독 해석 |
| * | degraded | K=96 too aggressive → H021 sub-H (K=80) 또는 다른 도메인 |

**동시 submit 안 가능 시**: H020 먼저 → 결과 회수 → H021 launch. 단 paired 비교의 정합성 위해 같은 git SHA + config 사용 권장.
