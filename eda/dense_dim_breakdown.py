"""Check 918 vs 755 discrepancy — measure actual per-row array length
of all 10 user_dense fids in demo_1000.parquet, compare to schema.json
(auto-generated max observed length) and ns_groups.json claim of 918.
"""

import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
PARQUET = ROOT / "data" / "demo_1000.parquet"
SCHEMA = ROOT / "data" / "schema.json"
OUT = ROOT / "eda" / "out" / "dense_dim_breakdown.json"

ALL_DENSE_FIDS = [61, 62, 63, 64, 65, 66, 87, 89, 90, 91]
ALIGNED_FIDS = {62, 63, 64, 65, 66, 89, 90, 91}
DENSE_ONLY_FIDS = {61, 87}
NS_GROUPS_TOTAL = 918  # claim from competition/ns_groups.json _note_user_dense

schema = json.loads(SCHEMA.read_text())
schema_dims = {fid: dim for fid, dim in schema["user_dense"]}

cols = [f"user_dense_feats_{fid}" for fid in ALL_DENSE_FIDS]
table = pq.read_table(PARQUET, columns=cols)
n_rows = table.num_rows

results = {
    "n_rows": n_rows,
    "ns_groups_json_total_dim": NS_GROUPS_TOTAL,
    "schema_json_total_dim": sum(schema_dims.values()),
    "data_observed_max_total": 0,
    "gap_ns_groups_minus_schema": NS_GROUPS_TOTAL - sum(schema_dims.values()),
    "interpretation": (
        "Schema sum and data observed_max sum match perfectly within demo_1000 — "
        "make_schema.py auto-extraction is correct. ns_groups.json's 918 reflects "
        "a different snapshot (likely full-data observed_max). Gap of 163 cannot "
        "be attributed to specific fid(s) without access to that snapshot's per-fid "
        "dims; most likely candidates = fids 61/87 (long dense vectors, near-full "
        "in demo_1000) or fids 65/66 (highest-cardinality aligned, capped at "
        "observed max in 1000-row sample)."
    ),
    "per_fid": {},
}

print(f"{'fid':<5} {'role':<11} {'schema_dim':<11} {'data_min':<9} {'data_max':<9} {'mean':<8} {'top3_lengths'}")
print("-" * 110)
for fid in ALL_DENSE_FIDS:
    arr = table.column(f"user_dense_feats_{fid}").to_pylist()
    lens = [0 if r is None else len(r) for r in arr]
    if not lens:
        continue
    mn, mx, avg = min(lens), max(lens), sum(lens) / len(lens)
    top3 = dict(Counter(lens).most_common(3))
    sd = schema_dims[fid]
    role = "aligned" if fid in ALIGNED_FIDS else "dense-only"
    results["data_observed_max_total"] += mx
    results["per_fid"][fid] = {
        "role": role,
        "schema_dim": sd,
        "data_min": mn,
        "data_max": mx,
        "data_mean": avg,
        "variance_max_minus_min": mx - mn,
        "histogram_top3": top3,
        "schema_matches_data_max": sd == mx,
    }
    print(f"{fid:<5} {role:<11} {sd:<11} {mn:<9} {mx:<9} {avg:<8.2f} {top3}")

print()
print(f"Sum of schema dims:        {results['schema_json_total_dim']}")
print(f"Sum of data observed_max:  {results['data_observed_max_total']}")
print(f"ns_groups.json claim:      {NS_GROUPS_TOTAL}")
print(f"Gap (ns_groups − schema):  {results['gap_ns_groups_minus_schema']}")

OUT.write_text(json.dumps(results, indent=2))
print(f"\nSaved → {OUT}")
