"""H011 sanity — dummy forward of RankMixerNSTokenizer with aligned_dense_specs.

Verifies:
- weighted-mean branch dispatches when aligned_dense_specs + dense_feats provided.
- baseline mean-pool when aligned_dense_specs={} or dense_feats=None.
- Output shape (B, num_ns_tokens, d_model) preserved.
- No NaN/inf with realistic dense value scale (Pattern X 18M, Pattern Y [-1, +1]).
- Equivalence: when weights are uniform (all=1), H011 weighted-mean reduces to baseline mean-pool.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "upload"))

import torch
from model import RankMixerNSTokenizer

torch.manual_seed(42)

# Match aligned fid 64 schema (vocab=52, dim=18). 1 fid, 1 group for simplicity.
feature_specs = [(52, 0, 18)]
groups = [[0]]
B = 4
emb_dim = 64
d_model = 64
num_ns_tokens = 1

aligned_dense_specs = {0: (0, 18)}  # fid_idx 0 → dense slice [0, 18)

# Dummy inputs.
int_feats = torch.randint(0, 52, (B, 18))
int_feats[0, 5:] = 0  # row 0: half-padding
int_feats[1, :] = 0   # row 1: all-padding (zero-row case)

# Pattern X scale (unbounded count up to 18M).
dense_feats_X = torch.rand(B, 18) * 18e6
# Pattern Y scale ([-1, +1] signed).
dense_feats_Y = torch.randn(B, 18) * 0.32

# === Test 1: weighted-mean branch with Pattern X ===
mod = RankMixerNSTokenizer(
    feature_specs=feature_specs, groups=groups,
    emb_dim=emb_dim, d_model=d_model, num_ns_tokens=num_ns_tokens,
    aligned_dense_specs=aligned_dense_specs,
)
out_X = mod(int_feats, dense_feats=dense_feats_X)
assert out_X.shape == (B, num_ns_tokens, d_model), f"Shape: {out_X.shape}"
assert torch.isfinite(out_X).all(), "NaN/inf in Pattern X output"
print(f"Test 1 (Pattern X, max=18M): out shape {out_X.shape}, finite OK, "
      f"|out|.max={out_X.abs().max().item():.4f}")

# === Test 2: weighted-mean branch with Pattern Y ===
out_Y = mod(int_feats, dense_feats=dense_feats_Y)
assert out_Y.shape == (B, num_ns_tokens, d_model)
assert torch.isfinite(out_Y).all(), "NaN/inf in Pattern Y output"
print(f"Test 2 (Pattern Y, [-1,+1]):  out shape {out_Y.shape}, finite OK, "
      f"|out|.max={out_Y.abs().max().item():.4f}")

# === Test 3: baseline branch (dense_feats=None) ===
out_base = mod(int_feats)
assert out_base.shape == (B, num_ns_tokens, d_model)
assert torch.isfinite(out_base).all()
print(f"Test 3 (baseline, dense=None): out shape {out_base.shape}, finite OK, "
      f"|out|.max={out_base.abs().max().item():.4f}")

# === Test 4: equivalence under uniform weights ===
# When all weights = 1.0, L1 norm divisor = n_active (count of non-zero), and
# weighted mean = sum(emb * 1/n_active * mask) = baseline mean pool.
uniform_w = torch.ones(B, 18)
out_uniform = mod(int_feats, dense_feats=uniform_w)
diff = (out_uniform - out_base).abs().max().item()
print(f"Test 4 (uniform weights == baseline): |diff|.max={diff:.6e} "
      f"({'PASS' if diff < 1e-5 else 'FAIL'})")

# === Test 5: zero-row safety ===
# row 1 has all-zero int_feats → mask all 0 → norm clamped to 1e-8 → no NaN.
zero_row_out = out_X[1]
assert torch.isfinite(zero_row_out).all(), "NaN/inf in zero-row"
print(f"Test 5 (zero-row safety): row 1 finite OK, |out[1]|.max={zero_row_out.abs().max().item():.4f}")

# === Test 6: no aligned specs (empty dict) ===
mod_empty = RankMixerNSTokenizer(
    feature_specs=feature_specs, groups=groups,
    emb_dim=emb_dim, d_model=d_model, num_ns_tokens=num_ns_tokens,
    aligned_dense_specs={},
)
out_empty = mod_empty(int_feats, dense_feats=dense_feats_X)
diff_empty = (out_empty - out_base).abs().max().item()
print(f"Test 6 (empty aligned_specs == baseline): |diff|.max={diff_empty:.6e} "
      f"({'PASS' if diff_empty < 1e-5 else 'FAIL'})")

n_params = sum(p.numel() for p in mod.parameters())
n_params_empty = sum(p.numel() for p in mod_empty.parameters())
print(f"\nParam count w/ aligned_specs: {n_params}")
print(f"Param count w/o aligned_specs: {n_params_empty}")
assert n_params == n_params_empty, "H011 should be parameter-free!"
print("Param-free verified ✓")
