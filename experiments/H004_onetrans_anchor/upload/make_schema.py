"""Auto-generate ``schema.json`` next to a parquet file/directory.

The organizer's ``dataset.py`` requires a ``schema.json`` colocated with the
training parquet that lists feature ids, vocab sizes, and per-feature dim.
The HuggingFace ``data_sample_1000`` release does NOT ship one, so without
this script ``train.py`` aborts at the very first step.

This v2 implementation is **production-scale safe**:
- Recursive glob (handles nested partitioned dirs).
- Reads only the first ``--max-row-groups-per-file`` row groups (default 1)
  instead of full files.
- Uses pyarrow zero-copy + numpy-vectorized stats (no per-row Python loops).
- Applies a generous vocab safety multiplier so unseen IDs in later row
  groups still index in-bound.
- Prints per-file progress + total elapsed time so a hung job is obvious.

Schema format (consumed by ``dataset.py``):

    {
      "user_int":  [[fid, vocab_size, dim], ...],
      "item_int":  [[fid, vocab_size, dim], ...],
      "user_dense":[[fid, dim], ...],
      "seq": {
        "seq_a": {"prefix": "domain_a_seq", "ts_fid": <fid_or_null>,
                  "features": [[fid, vocab_size], ...]},
        ...
      }
    }

Usage:
    python make_schema.py <src_parquet_or_dir> <dst_schema.json> \\
        [--max-row-groups-per-file 1] [--vocab-safety-mult 10]
"""

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

UNIX_TS_LOW = 1_500_000_000
UNIX_TS_HIGH = 2_000_000_000
DEFAULT_VOCAB_MULT = 10
DEFAULT_VOCAB_PAD = 100
DEFAULT_DIM_MULT = 1.5
MAX_DIM_CAP = 2048


def _stat_int_scalar(col_chunked) -> int:
    mv = 0
    for chunk in col_chunked.chunks:
        if chunk.null_count == len(chunk):
            continue
        arr = chunk.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)
        valid = arr[arr >= 0]
        if len(valid):
            mv = max(mv, int(valid.max()))
    return mv


def _stat_int_list(col_chunked) -> tuple[int, int]:
    """Returns (max_value, max_length) using vectorized numpy ops."""
    mv, ml = 0, 0
    for chunk in col_chunked.chunks:
        if chunk.null_count == len(chunk):
            continue
        # offsets array for ListArray
        offsets = np.asarray(chunk.offsets, dtype=np.int64)
        if len(offsets) <= 1:
            continue
        lengths = np.diff(offsets)
        if len(lengths):
            ml = max(ml, int(lengths.max()))
        # values are flat across all list cells in this chunk
        values = chunk.values
        if len(values) == 0:
            continue
        arr = np.asarray(values, dtype=np.int64)
        valid = arr[arr >= 0]
        if len(valid):
            mv = max(mv, int(valid.max()))
    return mv, ml


def _stat_float_list_max_len(col_chunked) -> int:
    ml = 0
    for chunk in col_chunked.chunks:
        if chunk.null_count == len(chunk):
            continue
        offsets = np.asarray(chunk.offsets, dtype=np.int64)
        if len(offsets) <= 1:
            continue
        lengths = np.diff(offsets)
        if len(lengths):
            ml = max(ml, int(lengths.max()))
    return ml


def _read_sample(files: list[str], max_rgs_per_file: int) -> pa.Table:
    tables = []
    for idx, f in enumerate(files, 1):
        pf = pq.ParquetFile(f)
        n_rgs = pf.metadata.num_row_groups
        take = min(max_rgs_per_file, n_rgs)
        for i in range(take):
            tables.append(pf.read_row_group(i))
        print(f"[make_schema] [{idx}/{len(files)}] {os.path.basename(f)}: "
              f"{n_rgs} row groups, took {take}", flush=True)
    if not tables:
        raise RuntimeError("no row groups read")
    if len(tables) == 1:
        return tables[0]
    try:
        return pa.concat_tables(tables, promote_options="default")
    except (TypeError, AttributeError):
        try:
            return pa.concat_tables(tables, promote=True)
        except (pa.lib.ArrowInvalid, TypeError):
            return pa.concat_tables(tables)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", help="parquet file or directory (recursive)")
    parser.add_argument("dst", help="output schema.json path")
    parser.add_argument("--max-row-groups-per-file", type=int, default=1,
                        help="only read first N row groups per file (default 1)")
    parser.add_argument("--vocab-safety-mult", type=int, default=DEFAULT_VOCAB_MULT,
                        help="vocab_size = observed_max × MULT + PAD (default 10)")
    parser.add_argument("--vocab-safety-pad", type=int, default=DEFAULT_VOCAB_PAD,
                        help="absolute pad added to vocab (default 100)")
    args = parser.parse_args()

    t0 = time.time()

    # Recursive parquet discovery (nested partitioned layouts supported).
    if os.path.isdir(args.src):
        files = sorted(str(p) for p in Path(args.src).rglob("*.parquet"))
    elif os.path.isfile(args.src):
        files = [args.src]
    else:
        raise SystemExit(f"src not found: {args.src}")
    if not files:
        raise SystemExit(f"no .parquet under {args.src}")
    print(f"[make_schema] discovered {len(files)} parquet file(s) under {args.src}", flush=True)
    print(f"[make_schema] sampling first {args.max_row_groups_per_file} row group(s) per file", flush=True)

    sample = _read_sample(files, args.max_row_groups_per_file)
    print(f"[make_schema] sample: {sample.num_rows} rows × {len(sample.schema.names)} cols "
          f"(elapsed {time.time() - t0:.1f}s)", flush=True)

    pat_user_int = re.compile(r"^user_int_feats_(\d+)$")
    pat_item_int = re.compile(r"^item_int_feats_(\d+)$")
    pat_user_dense = re.compile(r"^user_dense_feats_(\d+)$")
    pat_seq = re.compile(r"^domain_([a-z])_seq_(\d+)$")

    user_int_cols, item_int_cols = [], []
    user_dense_cols = []
    seq_per_domain = defaultdict(list)
    raw_max_for_seq = defaultdict(dict)  # for ts_fid heuristic

    n_processed = 0
    for name in sample.schema.names:
        n_processed += 1
        col = sample.column(name)
        if (m := pat_user_int.match(name)):
            fid = int(m.group(1))
            if pa.types.is_list(col.type):
                mv, ml = _stat_int_list(col)
                user_int_cols.append([
                    fid,
                    mv * args.vocab_safety_mult + args.vocab_safety_pad,
                    min(int(ml * DEFAULT_DIM_MULT) or 1, MAX_DIM_CAP),
                ])
            else:
                mv = _stat_int_scalar(col)
                user_int_cols.append([
                    fid,
                    mv * args.vocab_safety_mult + args.vocab_safety_pad,
                    1,
                ])
        elif (m := pat_item_int.match(name)):
            fid = int(m.group(1))
            if pa.types.is_list(col.type):
                mv, ml = _stat_int_list(col)
                item_int_cols.append([
                    fid,
                    mv * args.vocab_safety_mult + args.vocab_safety_pad,
                    min(int(ml * DEFAULT_DIM_MULT) or 1, MAX_DIM_CAP),
                ])
            else:
                mv = _stat_int_scalar(col)
                item_int_cols.append([
                    fid,
                    mv * args.vocab_safety_mult + args.vocab_safety_pad,
                    1,
                ])
        elif (m := pat_user_dense.match(name)):
            fid = int(m.group(1))
            ml = _stat_float_list_max_len(col)
            user_dense_cols.append([
                fid,
                min(int(ml * DEFAULT_DIM_MULT) or 1, MAX_DIM_CAP),
            ])
        elif (m := pat_seq.match(name)):
            letter = m.group(1)
            fid = int(m.group(2))
            mv, ml = _stat_int_list(col)
            seq_per_domain[letter].append((fid, mv * args.vocab_safety_mult + args.vocab_safety_pad, ml))
            raw_max_for_seq[letter][fid] = mv

    # ts_fid heuristic — compare RAW max (pre-multiplier) against unix-ts range.
    seq_cfg = {}
    for letter, items in sorted(seq_per_domain.items()):
        items_sorted = sorted(items, key=lambda x: x[0])
        ts_fid = next(
            (fid for fid, _, _ in items_sorted
             if UNIX_TS_LOW <= raw_max_for_seq[letter].get(fid, 0) <= UNIX_TS_HIGH),
            None,
        )
        seq_cfg[f"seq_{letter}"] = {
            "prefix": f"domain_{letter}_seq",
            "ts_fid": ts_fid,
            "features": [[fid, vs] for fid, vs, _ in items_sorted],
        }

    schema = {
        "_source": str(args.src),
        "_note": (
            f"auto-generated from first {args.max_row_groups_per_file} row group(s) per file. "
            f"vocab_size = observed_max × {args.vocab_safety_mult} + {args.vocab_safety_pad}. "
            f"dim = max_length × {DEFAULT_DIM_MULT} capped at {MAX_DIM_CAP}. "
            f"ts_fid via unix-ts range heuristic on raw observed max."
        ),
        "_n_files_scanned": len(files),
        "_n_row_groups_per_file": args.max_row_groups_per_file,
        "_sample_n_rows": int(sample.num_rows),
        "_elapsed_seconds": round(time.time() - t0, 2),
        "user_int": sorted(user_int_cols),
        "item_int": sorted(item_int_cols),
        "user_dense": sorted(user_dense_cols),
        "seq": seq_cfg,
    }

    Path(args.dst).parent.mkdir(parents=True, exist_ok=True)
    with open(args.dst, "w") as fh:
        json.dump(schema, fh, indent=2)

    elapsed = time.time() - t0
    print(f"[make_schema] elapsed: {elapsed:.1f}s", flush=True)
    print(f"[make_schema] wrote schema -> {args.dst}", flush=True)
    print(f"  user_int : {len(user_int_cols)} cols", flush=True)
    print(f"  item_int : {len(item_int_cols)} cols", flush=True)
    print(f"  user_dense: {len(user_dense_cols)} cols", flush=True)
    for k, v in seq_cfg.items():
        print(f"  {k}: {len(v['features'])} fids, ts_fid={v['ts_fid']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
