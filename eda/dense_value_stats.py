"""H011 gap #3 — measure user_dense weight value distribution per fid.

For each of the 10 user_dense fids, compute value stats over all non-padding
positions (= positions inside the per-row array, not 0-pad).

Decides H011 scale handling:
- bounded [0, 1] or near-zero std → raw multiply safe.
- unbounded / large variance → sigmoid gate or LayerNorm needed.
- negatives → sign-flip mechanism considered.
- NaN/inf → upstream data issue, INVALID.

Outputs eda/out/dense_value_stats.json.
"""

import json
import math
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
PARQUET = ROOT / "data" / "demo_1000.parquet"
OUT = ROOT / "eda" / "out" / "dense_value_stats.json"

ALL_DENSE_FIDS = [61, 62, 63, 64, 65, 66, 87, 89, 90, 91]
ALIGNED_FIDS = {62, 63, 64, 65, 66, 89, 90, 91}


def percentile(sorted_arr, p):
    if not sorted_arr:
        return None
    k = (len(sorted_arr) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_arr[int(k)]
    return sorted_arr[f] + (sorted_arr[c] - sorted_arr[f]) * (k - f)


def stats(values):
    if not values:
        return None
    n = len(values)
    n_nan = sum(1 for v in values if v is None or (isinstance(v, float) and math.isnan(v)))
    n_inf = sum(1 for v in values if isinstance(v, float) and math.isinf(v))
    finite = [v for v in values if v is not None and isinstance(v, (int, float))
              and not math.isnan(v) and not math.isinf(v)]
    if not finite:
        return {"n": n, "n_nan": n_nan, "n_inf": n_inf, "n_finite": 0}
    mn = min(finite)
    mx = max(finite)
    mean = sum(finite) / len(finite)
    var = sum((v - mean) ** 2 for v in finite) / len(finite)
    std = math.sqrt(var)
    sorted_finite = sorted(finite)
    return {
        "n": n,
        "n_nan": n_nan,
        "n_inf": n_inf,
        "n_finite": len(finite),
        "n_negative": sum(1 for v in finite if v < 0),
        "n_zero_value": sum(1 for v in finite if v == 0.0),
        "min": mn,
        "max": mx,
        "mean": mean,
        "std": std,
        "p01": percentile(sorted_finite, 0.01),
        "p50": percentile(sorted_finite, 0.50),
        "p99": percentile(sorted_finite, 0.99),
    }


cols = [f"user_dense_feats_{fid}" for fid in ALL_DENSE_FIDS]
table = pq.read_table(PARQUET, columns=cols)
n_rows = table.num_rows

results = {"n_rows": n_rows, "per_fid": {}}

print(f"{'fid':<5} {'role':<11} {'n_finite':<10} {'min':<10} {'max':<10} {'mean':<10} {'std':<10} {'p99':<10} {'neg':<5} {'nan':<5}")
print("-" * 120)

for fid in ALL_DENSE_FIDS:
    arr = table.column(f"user_dense_feats_{fid}").to_pylist()
    flat = []
    for r in arr:
        if r is None:
            continue
        flat.extend(r)
    s = stats(flat)
    role = "aligned" if fid in ALIGNED_FIDS else "dense-only"
    results["per_fid"][fid] = {"role": role, "stats": s}
    if s is None:
        print(f"{fid:<5} {role:<11} (no values)")
        continue
    print(
        f"{fid:<5} {role:<11} {s['n_finite']:<10} "
        f"{s['min']:<10.4g} {s['max']:<10.4g} "
        f"{s['mean']:<10.4g} {s['std']:<10.4g} "
        f"{s['p99']:<10.4g} {s['n_negative']:<5} {s['n_nan']:<5}"
    )

# H011 scale handling decision
def decide(s):
    if s is None or s.get("n_finite", 0) == 0:
        return "skip"
    if s["n_nan"] > 0 or s["n_inf"] > 0:
        return "INVALID — upstream NaN/inf"
    if s["n_negative"] > 0:
        return "negatives present — sign-flip risk, sigmoid 또는 abs"
    if s["max"] > 100 or s["std"] > 10:
        return "large variance — LayerNorm or sigmoid recommended"
    if s["max"] <= 1.0 and s["min"] >= 0.0:
        return "bounded [0,1] — raw multiply safe"
    return "moderate — raw multiply OK, monitor first batch"

print()
print("--- H011 scale handling decision per fid ---")
decisions = {}
for fid in ALL_DENSE_FIDS:
    s = results["per_fid"][fid]["stats"]
    decisions[fid] = decide(s)
    print(f"  fid {fid}: {decisions[fid]}")
results["h011_scale_decisions"] = decisions

OUT.write_text(json.dumps(results, indent=2, default=str))
print(f"\nSaved → {OUT}")
