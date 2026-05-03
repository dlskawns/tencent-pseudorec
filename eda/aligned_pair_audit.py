"""H011 P0 audit — verify aligned <id, weight> pair length consistency.

For each aligned fid k in {62, 63, 64, 65, 66, 89, 90, 91}:
- Read user_int_feats_k (list<int>) and user_dense_feats_k (list<float>) per row.
- Measure n_k (int array length) and m_k (dense array length) distribution.
- Compare: PASS if n_k == m_k for all rows (Option A position-wise binding).
- Compare schema dim vs observed max.

Outputs:
- eda/out/aligned_offsets.json  — schema dim per fid (from schema.json)
- eda/out/array_lengths.json    — per-row stats per fid (from demo_1000)
- eda/out/aligned_audit.json    — verdict (PASS/FAIL) + per-fid match counts
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
PARQUET = ROOT / "data" / "demo_1000.parquet"
SCHEMA = ROOT / "data" / "schema.json"
OUT_DIR = ROOT / "eda" / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALIGNED_FIDS = [62, 63, 64, 65, 66, 89, 90, 91]


def load_schema_dims():
    schema = json.loads(SCHEMA.read_text())
    user_int_dims = {fid: dim for fid, _vs, dim in schema["user_int"]}
    user_dense_dims = {fid: dim for fid, dim in schema["user_dense"]}
    return user_int_dims, user_dense_dims


def measure_per_fid(table, fid):
    """Return per-row (n, m) arrays for user_int_feats_fid and user_dense_feats_fid."""
    int_col = f"user_int_feats_{fid}"
    dense_col = f"user_dense_feats_{fid}"
    int_arr = table.column(int_col).to_pylist()
    dense_arr = table.column(dense_col).to_pylist()

    ns = []
    ms = []
    for r_int, r_dense in zip(int_arr, dense_arr):
        ns.append(0 if r_int is None else len(r_int))
        ms.append(0 if r_dense is None else len(r_dense))
    return ns, ms


def stats(arr):
    if not arr:
        return {}
    arr = list(arr)
    return {
        "min": min(arr),
        "max": max(arr),
        "mean": sum(arr) / len(arr),
        "n_zeros": sum(1 for x in arr if x == 0),
        "histogram": dict(Counter(arr).most_common(10)),
    }


def main():
    user_int_dims, user_dense_dims = load_schema_dims()

    cols = []
    for fid in ALIGNED_FIDS:
        cols.append(f"user_int_feats_{fid}")
        cols.append(f"user_dense_feats_{fid}")
    table = pq.read_table(PARQUET, columns=cols)
    n_rows = table.num_rows

    offsets = {}
    offset = 0
    for fid, dim in [(fid, user_dense_dims[fid]) for fid in [61, 62, 63, 64, 65, 66, 87, 89, 90, 91]]:
        offsets[fid] = {"offset": offset, "dim": dim}
        offset += dim
    total_dim = offset
    (OUT_DIR / "aligned_offsets.json").write_text(
        json.dumps({"per_fid": offsets, "total_dim": total_dim, "source": "data/schema.json"}, indent=2)
    )

    array_lengths = {}
    audit = {"n_rows": n_rows, "per_fid": {}, "all_match": True}

    for fid in ALIGNED_FIDS:
        ns, ms = measure_per_fid(table, fid)
        match_per_row = sum(1 for n, m in zip(ns, ms) if n == m)
        array_lengths[fid] = {
            "user_int_feats_lengths": stats(ns),
            "user_dense_feats_lengths": stats(ms),
            "schema_user_int_dim": user_int_dims[fid],
            "schema_user_dense_dim": user_dense_dims[fid],
        }
        per_fid_pass = (match_per_row == n_rows) and (
            user_int_dims[fid] == user_dense_dims[fid]
        )
        audit["per_fid"][fid] = {
            "rows_with_n_eq_m": match_per_row,
            "rows_total": n_rows,
            "match_rate": match_per_row / n_rows,
            "schema_dims_match": user_int_dims[fid] == user_dense_dims[fid],
            "schema_int_dim": user_int_dims[fid],
            "schema_dense_dim": user_dense_dims[fid],
            "data_int_max": max(ns) if ns else 0,
            "data_dense_max": max(ms) if ms else 0,
            "verdict": "PASS" if per_fid_pass else "FAIL",
        }
        if not per_fid_pass:
            audit["all_match"] = False

    audit["overall_verdict"] = "PASS — Option A (position-wise) viable" if audit["all_match"] else "FAIL — needs Option B/C"
    (OUT_DIR / "array_lengths.json").write_text(json.dumps(array_lengths, indent=2, default=str))
    (OUT_DIR / "aligned_audit.json").write_text(json.dumps(audit, indent=2))

    print(f"Rows: {n_rows}")
    print(f"Total user_dense_dim: {total_dim}")
    print()
    print(f"{'fid':<5} {'int_dim':<8} {'dense_dim':<10} {'data_n_max':<11} {'data_m_max':<11} {'rows_match':<11} {'verdict':<8}")
    for fid in ALIGNED_FIDS:
        a = audit["per_fid"][fid]
        print(
            f"{fid:<5} {a['schema_int_dim']:<8} {a['schema_dense_dim']:<10} "
            f"{a['data_int_max']:<11} {a['data_dense_max']:<11} "
            f"{a['rows_with_n_eq_m']}/{a['rows_total']:<8} {a['verdict']:<8}"
        )
    print()
    print(f"Overall verdict: {audit['overall_verdict']}")


if __name__ == "__main__":
    main()
