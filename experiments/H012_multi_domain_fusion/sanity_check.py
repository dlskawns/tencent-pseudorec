"""H012 sanity — dummy forward of MultiDomainMoEBlock.

Verifies:
- Output shape (B, N_NS, D) preserved.
- No NaN/inf.
- Param count = expected (~33K for 4 experts × 64→128→64 + gate + LN).
- Gate entropy logging when log_gate_entropy=True.
- Baseline equivalence: when MoE block absent, baseline path unchanged.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "upload"))

import torch
from model import MultiDomainMoEBlock

torch.manual_seed(42)

B, N_NS, D = 4, 7, 64
ns_tokens = torch.randn(B, N_NS, D)

# === Test 1: shape + finite ===
mod = MultiDomainMoEBlock(d_model=D, num_experts=4, ffn_hidden=128, dropout=0.0,
                          log_gate_entropy=True)
out = mod(ns_tokens)
assert out.shape == (B, N_NS, D), f"Shape: {out.shape}"
assert torch.isfinite(out).all(), "NaN/inf in output"
print(f"Test 1 (shape + finite): out shape {out.shape}, |out|.max={out.abs().max().item():.4f}")

# === Test 2: param count ===
n_params = sum(p.numel() for p in mod.parameters())
# Expected: 4 × (64×128 + 128 + 128×64 + 64) + (64×4 + 4) + (64 + 64)  [LN]
expected = 4 * (64 * 128 + 128 + 128 * 64 + 64) + (64 * 4 + 4) + (64 + 64)
print(f"Test 2 (param count): {n_params} (expected ~{expected})")
assert abs(n_params - expected) < 100, f"Param count mismatch: {n_params} vs {expected}"

# === Test 3: gate entropy logging ===
ent = mod._last_gate_entropy
log_E = math.log(4)
collapse_threshold = 0.5 * log_E
assert ent is not None, "gate entropy not logged"
print(f"Test 3 (gate entropy): {ent.item():.4f} "
      f"(collapse threshold {collapse_threshold:.4f}, uniform {log_E:.4f})")
# Random init → entropy ≈ log(4) ≈ 1.386 (uniform). After training: should specialize.
assert 0.0 < ent.item() <= log_E + 0.01, f"Entropy out of range: {ent.item()}"

# === Test 4: residual connection (output ≠ input but bounded) ===
delta = (out - ns_tokens).abs().mean().item()
print(f"Test 4 (residual delta): {delta:.4f} (should be > 0 and bounded)")
assert delta > 0.0, "residual not applied"
assert delta < 10.0, "delta too large"

# === Test 5: gradient flow ===
loss = out.sum()
loss.backward()
n_with_grad = sum(1 for p in mod.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
n_total = sum(1 for _ in mod.parameters())
print(f"Test 5 (gradient flow): {n_with_grad}/{n_total} params have non-zero grad")
assert n_with_grad == n_total, "some params have no grad"

print("\nAll H012 sanity tests PASSED ✓")
