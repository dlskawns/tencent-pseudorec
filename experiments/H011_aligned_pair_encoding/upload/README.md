# H011_aligned_pair_encoding — Technical Report

> CLAUDE.md §17.2 stacking sub-H on H010 champion + H008 anchor: H010 envelope
> + H010 mechanism (NS→S xattn) + H008 mechanism (DCN-V2 fusion) byte-identical
> PLUS `--use_aligned_pair_encoding` 추가. **input-stage** mutation으로 H010 의
> NS xattn / DCN-V2 fusion 입력 텐서 byte-identical → H009 위치 충돌 회피.

## Mechanism

For each verified aligned fid k ∈ `{62, 63, 64, 65, 66, 89, 90, 91}` (출처:
`competition/ns_groups.json _note_shared_fids` + `eda/out/aligned_audit.json`),
inside `RankMixerNSTokenizer.forward`'s per-fid embedding loop:

```python
# Baseline (non-aligned 또는 dense_feats=None):
fid_emb = (emb_all * mask).sum(dim=1) / count

# H011 (aligned + dense_feats provided):
w = dense_feats[:, d_offset:d_offset + d_dim]            # (B, n_k)
abs_w_masked = w.abs() * mask.squeeze(-1)
norm = abs_w_masked.sum(dim=1, keepdim=True).clamp(min=1e-8)
w_norm = w / norm                                        # row-L1 norm, sign 보존
fid_emb = (emb_all * w_norm.unsqueeze(-1) * mask).sum(dim=1)
```

Per-row L1-normalized weighted mean (Option α) — parameter-free, sign-preserved,
NaN-safe, baseline mean-pool 의 자연스러운 weighted generalization (uniform
weights → 정확히 baseline 으로 reduce).

## Why this scale handling (not raw multiply)

`eda/out/dense_value_stats.json` 에서 user_dense 가 **두 분포** 혼재:
- **Pattern X (fid 62-66)**: unbounded count/duration, max=4.8M~18.4M.
  raw multiply 시 gradient explode 확정.
- **Pattern Y (fid 89-91)**: ~[-1, +1], std≈0.32, ~60% negative.
  raw multiply 가능하지만 X 와 통일 처리 위해 같은 normalize.

per-row L1 norm 이 두 pattern 통일 + sign 보존 + parameter-free.

## Stacking on champion

- H008 anchor (Platform 0.8387) — DCN-V2 block fusion 그대로.
- H010 anchor (Platform 0.8408, current champion) — NS→S bidirectional xattn
  그대로.
- H011 추가 = `RankMixerNSTokenizer.forward` 안의 mean-pool 분기. **NS xattn
  출력 텐서 byte-identical, DCN-V2 입력 텐서 byte-identical** — H009 의
  block-level 위치 충돌 패턴 회피 by 설계.

## Files

| File | Diff vs H010/upload/ | Role |
|---|---|---|
| `run.sh` | +2 flags (`--use_aligned_pair_encoding`, `--aligned_pair_fids ...`) | Entry point |
| `train.py` | +argparse 2 + aligned_user_dense_specs 산출 + model_args 2 keys | CLI driver |
| `model.py` | RankMixerNSTokenizer __init__ + forward (weighted-mean branch); PCVRHyFormer __init__ 2 args + _build_token_streams 분기 | PCVRHyFormer + DCN-V2 + NS xattn + aligned encoding |
| `infer.py` | aligned_user_dense_specs 재구성 + cfg.get 2 keys | §18 인프라 + new cfg |
| `trainer.py` | byte-identical | Train loop |
| `dataset.py` | byte-identical | §18.2 universal handler |
| `local_validate.py` | byte-identical | G1–G6 |
| `make_schema.py` | byte-identical | §18.5 |
| `utils.py` | byte-identical | helpers |
| `ns_groups.json` | byte-identical | NS ref |
| `requirements.txt` | byte-identical | deps |
| `README.md` | 변경 | H011 정체성 |

총 12 files.

## Reproducibility

- seed 42, label_time split + 10% OOF.
- `--use_aligned_pair_encoding --aligned_pair_fids 62 63 64 65 66 89 90 91` baked.
- 모든 H010/H008 flags 동시에 baked.

## Repro pin (post-launch)
- git_sha: TBD (launch 직전 캡쳐).
- config_sha256: TBD (run 후 metrics.json).
