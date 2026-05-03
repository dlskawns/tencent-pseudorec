# H012_multi_domain_fusion — Technical Report

> CLAUDE.md §17.2 stacking sub-H on H010 champion (Platform 0.8408): H010
> envelope + H010 mechanism (NS→S xattn) + H008 mechanism (DCN-V2 fusion)
> byte-identical PLUS `--use_multi_domain_moe` 추가. **NS-token level**
> mutation, post NS xattn / pre query decoder. NS dimension preserved →
> DCN-V2 fusion input token stack unchanged → H009 위치 충돌 회피, H011 의
> input-stage cohort drift 회피.

## Mechanism

```python
class MultiDomainMoEBlock(nn.Module):
    """4 expert FFNs + softmax gate per NS-token. Source: MMoE (KDD 2018)."""
    def __init__(d_model=64, num_experts=4, ffn_hidden=128):
        self.experts = [
            Sequential(Linear(d_model, ffn_hidden), SiLU(), Linear(ffn_hidden, d_model))
            for _ in range(num_experts)
        ]
        self.gate = Linear(d_model, num_experts)
        self.input_ln = LayerNorm(d_model)

    def forward(ns_tokens):                                  # (B, N_NS, D)
        x = self.input_ln(ns_tokens)
        gate_w = softmax(self.gate(x), dim=-1)               # (B, N_NS, E)
        expert_out = stack([e(x) for e in self.experts], -2) # (B, N_NS, E, D)
        weighted = (expert_out * gate_w.unsqueeze(-1)).sum(-2)  # (B, N_NS, D)
        return ns_tokens + weighted                          # residual
```

통합 위치: `MultiSeqHyFormerBlock.forward`, NS xattn 직후 (line 1068
바로 뒤). NS-token (B, 7, D) → MultiDomainMoEBlock → enriched (B, 7, D).
shape 보존, downstream 텐서 byte-identical.

## Why this stacking is safe (vs H009 / H011 lessons)

- H009 (REFUTED interference): candidate token prepend → seq encoder 입력
  변경 → DCN-V2 입력 변경. **여러 텐서 동시 변경**.
- H011 (REFUTED degraded): input embedding lookup → user NS tokens 표현
  변경 + cohort drift 악화. **input-stage modify**.
- **H012**: NS-token level enrichment, anchor 의 NS xattn 출력 텐서만 변경.
  S tokens / decoded queries / DCN-V2 입력 모두 byte-identical. H010 F-1
  안전 stacking 패턴 직접 따름.

## Why MMoE on multi-domain (data motivation)

`eda/out/domain_facts.json`:
- **Domain Jaccard overlap**: a_vs_c=0.7%, a_vs_b=7.9%, max a_vs_d=10%.
  4 도메인 거의 disjoint vocab → expert specialization 자연.
- **Domain seq length p50** (§3.5): a=577, b=405, c=322, d=1035. d 가
  3× 길이.
- **frac_empty**: a=0.5%, d=8%. d 의 양극 분포 (heavy-tail + inactive).

→ 단일 fusion block 이 4 도메인 균일 처리. MMoE 로 explicit specialization.

## Sample-scale collapse risk (§10.9 룰 적용)

gate softmax routing 의 collapse 위험 (1-2 expert 만 활성).
- threshold (collapse): 0.5 × log(4) ≈ **0.69** 미만 시 §10.9 abort.
- threshold (uniform): log(4) ≈ **1.39** = uniform routing (no specialization).
- specialized 범위: entropy ∈ [0.69, 1.30] 권장.
- log_attn_entropy active 시 metrics.json 의 `moe_gate_entropy_per_block`
  로 기록.

## Files

| File | Diff vs H010/upload/ | Role |
|---|---|---|
| `run.sh` | +3 flags (`--use_multi_domain_moe --num_experts 4 --moe_ffn_hidden 128`) | Entry point |
| `train.py` | +argparse 3 + model_args 3 + MoE entropy diagnostic block | CLI driver |
| `model.py` | +MultiDomainMoEBlock 클래스 (~60 lines) + MultiSeqHyFormerBlock __init__/forward + PCVRHyFormer __init__ + collect_moe_gate_entropies() | PCVRHyFormer + DCN-V2 + NS xattn + MMoE |
| `infer.py` | +cfg.get 3 keys | §18 인프라 + new cfg |
| `trainer.py` | byte-identical | Train loop |
| `dataset.py` | byte-identical | §18.2 universal handler |
| `local_validate.py` | byte-identical | G1–G6 |
| `make_schema.py` | byte-identical | §18.5 |
| `utils.py` | byte-identical | helpers |
| `ns_groups.json` | byte-identical | NS ref |
| `requirements.txt` | byte-identical | deps |
| `README.md` | 변경 | H012 정체성 |

총 12 files.

## Reproducibility

- seed 42, label_time split + 10% OOF.
- `--use_multi_domain_moe --num_experts 4 --moe_ffn_hidden 128` baked.
- 모든 H010/H008 flags 동시 baked.
- ~33K params 추가 (4 × FFN 64→128→64 + gate 64→4).

## Repro pin (post-launch)
- git_sha: TBD (launch 직전 캡쳐).
- config_sha256: TBD (run 후 metrics.json).
