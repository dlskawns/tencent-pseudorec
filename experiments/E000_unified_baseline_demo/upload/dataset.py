"""PCVR Parquet dataset module (performance-tuned).

Reads raw multi-column Parquet directly and obtains feature metadata from
``schema.json``.

Optimizations:
- Pre-allocated numpy buffers to eliminate ``np.zeros`` + ``np.stack`` overhead.
- Fused padding loop over sequence domains that writes directly into a 3D buffer.
- Pre-computed column-index lookup to avoid per-row string lookups.
- ``file_system`` tensor-sharing strategy to work around ``/dev/shm`` exhaustion
  when using many DataLoader workers.
"""

import os
import logging
import random
import json
import gc

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.multiprocessing
from torch.utils.data import IterableDataset, DataLoader
from typing import Any, Dict, Iterator, List, Optional, Tuple

# numpy.typing is available since numpy >= 1.20; on older numpy fall back to a
# no-op shim so that forward-referenced annotations like ``npt.NDArray[np.int64]``
# keep working as plain strings without raising at import time.
try:
    import numpy.typing as npt  # noqa: F401
except ImportError:  # pragma: no cover
    class _NptFallback:  # type: ignore[no-redef]
        NDArray = Any

    npt = _NptFallback()  # type: ignore[assignment]


# ─────────────────────────── Feature Schema ──────────────────────────────────


class FeatureSchema:
    """Records ``(feature_id, offset, length)`` for each feature so downstream
    code can locate the segment of the flattened tensor that belongs to a
    specific feature id.

    For int features:
      - int_value: length = 1
      - int_array: length = array length
      - int_array_and_float_array: int part length
    For dense features:
      - float_value: length = 1
      - float_array: length = array length
      - int_array_and_float_array: float part length
    """

    def __init__(self) -> None:
        # Ordered list of (feature_id, offset, length).
        self.entries: List[Tuple[int, int, int]] = []
        self.total_dim: int = 0
        # Quick lookup from fid to its (offset, length).
        self._fid_to_entry: Dict[int, Tuple[int, int]] = {}

    def add(self, feature_id: int, length: int) -> None:
        """Append a feature to the schema."""
        offset = self.total_dim
        self.entries.append((feature_id, offset, length))
        self._fid_to_entry[feature_id] = (offset, length)
        self.total_dim += length

    def get_offset_length(self, feature_id: int) -> Tuple[int, int]:
        """Get ``(offset, length)`` for a feature_id."""
        return self._fid_to_entry[feature_id]

    @property
    def feature_ids(self) -> List[int]:
        """Return all feature_ids in their insertion order."""
        return [fid for fid, _, _ in self.entries]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict (for JSON dumping)."""
        return {
            'entries': self.entries,
            'total_dim': self.total_dim,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'FeatureSchema':
        """Reconstruct a :class:`FeatureSchema` from its dict form."""
        schema = cls()
        for fid, offset, length in d['entries']:
            schema.entries.append((fid, offset, length))
            schema._fid_to_entry[fid] = (offset, length)
        schema.total_dim = d['total_dim']
        return schema

    def __repr__(self) -> str:
        lines = [f"FeatureSchema(total_dim={self.total_dim}, features=["]
        for fid, offset, length in self.entries:
            lines.append(f"  fid={fid}: offset={offset}, length={length}")
        lines.append("])")
        return "\n".join(lines)

# Use filesystem-based tensor sharing (instead of /dev/shm) to avoid running
# out of shared memory when many DataLoader workers are active.
torch.multiprocessing.set_sharing_strategy('file_system')


class _RowFilter:
    """Pickleable row-mask callable for ``PCVRParquetDataset.row_filter``.

    Given a pyarrow RecordBatch, returns a boolean numpy mask of length B
    selecting rows that belong to ``target`` ∈ {``train``, ``valid``,
    ``oof``}. Used to share a single set of source parquet files across
    three logical splits without materializing subset parquets to disk.
    """

    __slots__ = ('oof_users_sorted', 'cutoff', 'target')

    def __init__(self, oof_users_sorted: np.ndarray, cutoff: float, target: str) -> None:
        if target not in {'train', 'valid', 'oof'}:
            raise ValueError(f"target must be one of train/valid/oof, got {target!r}")
        self.oof_users_sorted = np.asarray(oof_users_sorted, dtype=np.int64)
        self.cutoff = float(cutoff)
        self.target = target

    def __call__(self, batch: "pa.RecordBatch") -> np.ndarray:
        ids = batch.column('user_id').to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        is_oof = np.isin(ids, self.oof_users_sorted)
        if self.target == 'oof':
            return is_oof
        lts = batch.column('label_time').to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        is_valid = (lts >= self.cutoff) & (~is_oof)
        if self.target == 'valid':
            return is_valid
        return (~is_valid) & (~is_oof)

# Time-delta bucket boundaries (64 edges -> 65 buckets: 0=padding, 1..64).
BUCKET_BOUNDARIES = np.array([
    5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60,
    120, 180, 240, 300, 360, 420, 480, 540, 600,
    900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600,
    5400, 7200, 9000, 10800, 12600, 14400, 16200, 18000, 19800, 21600,
    32400, 43200, 54000, 64800, 75600, 86400,
    172800, 259200, 345600, 432000, 518400, 604800,
    1123200, 1641600, 2160000, 2592000,
    4320000, 6048000, 7776000,
    11664000, 15552000,
    31536000,
], dtype=np.int64)

# Total number of time-bucket embedding slots (= number of boundaries + 1, with
# padding=0 included).
#
# This constant is uniquely determined by the length of BUCKET_BOUNDARIES; on
# the model side, ``nn.Embedding(num_embeddings=NUM_TIME_BUCKETS)`` must match
# this value exactly, otherwise an IndexError may be raised at runtime.
#
# That is why ``train.py`` / ``infer.py`` only expose the boolean flag
# ``--use_time_buckets`` and derive the concrete bucket count from here.
NUM_TIME_BUCKETS = len(BUCKET_BOUNDARIES) + 1


class PCVRParquetDataset(IterableDataset):
    """PCVR dataset that reads raw multi-column Parquet directly.

    - int features: scalar or list (multi-hot); values <= 0 are mapped to 0 (padding).
    - dense features: ``list<float>``, variable-length padded up to ``max_dim``.
    - sequence features: ``list<int64>``, grouped by domain; includes side-info
      columns and an optional timestamp column (used for time-bucketing).
    - label: mapped from ``label_type == 2``.
    """

    def __init__(
        self,
        parquet_path: str,
        schema_path: str,
        batch_size: int = 256,
        seq_max_lens: Optional[Dict[str, int]] = None,
        shuffle: bool = True,
        buffer_batches: int = 20,
        row_group_range: Optional[Tuple[int, int]] = None,
        clip_vocab: bool = True,
        is_training: bool = True,
        row_filter: Optional[Any] = None,
    ) -> None:
        """
        Args:
            parquet_path: either a directory containing ``*.parquet`` files or
                a single parquet file path.
            schema_path: path of the schema JSON describing feature layouts.
            batch_size: fixed batch size used for the pre-allocated buffers.
            seq_max_lens: optional per-domain override of sequence truncation,
                e.g. ``{'seq_d': 256}``. Domains not listed fall back to the
                schema default of 256.
            shuffle: whether to shuffle within a ``buffer_batches``-sized window.
            buffer_batches: shuffle buffer size in units of batches.
            row_group_range: ``(start, end)`` slice of Row Groups; ``None`` to
                use all Row Groups.
            clip_vocab: if True, clip out-of-bound ids to 0; if False, raise.
            is_training: if True, derive ``label`` from ``label_type == 2``;
                if False, return an all-zeros label column.
        """
        super().__init__()

        # Accept either a directory or a single file path.
        if os.path.isdir(parquet_path):
            import glob
            files = sorted(glob.glob(os.path.join(parquet_path, '*.parquet')))
            if not files:
                raise FileNotFoundError(f"No .parquet files in {parquet_path}")
            self._parquet_files = files
        else:
            self._parquet_files = [parquet_path]

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.buffer_batches = buffer_batches
        self.clip_vocab = clip_vocab
        self.is_training = is_training
        # Optional row-mask callable applied to every Arrow RecordBatch
        # right after pf.iter_batches(...) and before _convert_batch(). Lets
        # 3 logical splits share one set of source parquets without
        # subset materialization.
        self._row_filter = row_filter
        # Out-of-bound statistics:
        #   {(group, col_idx): {'count': N, 'max': M, 'min_oob': M, 'vocab': V}}
        self._oob_stats: Dict[Tuple[str, int], Dict[str, int]] = {}

        # Build the list of Row Groups.
        self._rg_list = []
        for f in self._parquet_files:
            pf = pq.ParquetFile(f)
            for i in range(pf.metadata.num_row_groups):
                self._rg_list.append((f, i, pf.metadata.row_group(i).num_rows))

        if row_group_range is not None:
            start, end = row_group_range
            self._rg_list = self._rg_list[start:end]

        self.num_rows = sum(r[2] for r in self._rg_list)

        # Load schema.json.
        self._load_schema(schema_path, seq_max_lens or {})

        # ---- Pre-compute column index lookup ----
        pf = pq.ParquetFile(self._parquet_files[0])
        schema_names = pf.schema_arrow.names
        self._col_idx = {name: i for i, name in enumerate(schema_names)}

        # ---- Pre-allocate numpy buffers ----
        B = batch_size
        self._buf_user_int = np.zeros((B, self.user_int_schema.total_dim), dtype=np.int64)
        self._buf_item_int = np.zeros((B, self.item_int_schema.total_dim), dtype=np.int64)
        self._buf_user_dense = np.zeros((B, self.user_dense_schema.total_dim), dtype=np.float32)
        self._buf_seq = {}
        self._buf_seq_tb = {}
        self._buf_seq_lens = {}
        for domain in self.seq_domains:
            max_len = self._seq_maxlen[domain]
            n_feats = len(self.sideinfo_fids[domain])
            self._buf_seq[domain] = np.zeros((B, n_feats, max_len), dtype=np.int64)
            self._buf_seq_tb[domain] = np.zeros((B, max_len), dtype=np.int64)
            self._buf_seq_lens[domain] = np.zeros(B, dtype=np.int64)

        # ---- Pre-compute (col_idx, offset, vocab_size) plans for int columns ----
        self._user_int_plan = []  # [(col_idx, dim, offset, vocab_size), ...]
        offset = 0
        for fid, vs, dim in self._user_int_cols:
            ci = self._col_idx.get(f'user_int_feats_{fid}')
            self._user_int_plan.append((ci, dim, offset, vs))
            offset += dim

        self._item_int_plan = []
        offset = 0
        for fid, vs, dim in self._item_int_cols:
            ci = self._col_idx.get(f'item_int_feats_{fid}')
            self._item_int_plan.append((ci, dim, offset, vs))
            offset += dim

        self._user_dense_plan = []
        offset = 0
        for fid, dim in self._user_dense_cols:
            ci = self._col_idx.get(f'user_dense_feats_{fid}')
            self._user_dense_plan.append((ci, dim, offset))
            offset += dim

        # Sequence column plan: {domain: ([(col_idx, feat_slot, vocab_size), ...], ts_col_idx)}
        self._seq_plan = {}
        for domain in self.seq_domains:
            prefix = self._seq_prefix[domain]
            sideinfo_fids = self.sideinfo_fids[domain]
            ts_fid = self.ts_fids[domain]
            side_plan = []
            for slot, fid in enumerate(sideinfo_fids):
                ci = self._col_idx.get(f'{prefix}_{fid}')
                vs = self.seq_vocab_sizes[domain][fid]
                side_plan.append((ci, slot, vs))
            ts_ci = self._col_idx.get(f'{prefix}_{ts_fid}') if ts_fid is not None else None
            self._seq_plan[domain] = (side_plan, ts_ci)

        logging.info(
            f"PCVRParquetDataset: {self.num_rows} rows from "
            f"{len(self._parquet_files)} file(s), batch_size={batch_size}, "
            f"buffer_batches={buffer_batches}, shuffle={shuffle}")

    def _load_schema(self, schema_path: str, seq_max_lens: Dict[str, int]) -> None:
        """Populate per-group schema information from ``schema_path``."""
        with open(schema_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        # ---- user_int: [[fid, vocab_size, dim], ...] ----
        self._user_int_cols: List[List[int]] = raw['user_int']
        self.user_int_schema: FeatureSchema = FeatureSchema()
        self.user_int_vocab_sizes: List[int] = []
        for fid, vs, dim in self._user_int_cols:
            self.user_int_schema.add(fid, dim)
            self.user_int_vocab_sizes.extend([vs] * dim)

        # ---- item_int ----
        self._item_int_cols: List[List[int]] = raw['item_int']
        self.item_int_schema: FeatureSchema = FeatureSchema()
        self.item_int_vocab_sizes: List[int] = []
        for fid, vs, dim in self._item_int_cols:
            self.item_int_schema.add(fid, dim)
            self.item_int_vocab_sizes.extend([vs] * dim)

        # ---- user_dense: [[fid, dim], ...] ----
        self._user_dense_cols: List[List[int]] = raw['user_dense']
        self.user_dense_schema: FeatureSchema = FeatureSchema()
        for fid, dim in self._user_dense_cols:
            self.user_dense_schema.add(fid, dim)

        # ---- item_dense (empty) ----
        self.item_dense_schema: FeatureSchema = FeatureSchema()

        # ---- sequence domains ----
        self._seq_cfg: Dict[str, Dict[str, Any]] = raw['seq']
        self.seq_domains: List[str] = sorted(self._seq_cfg.keys())
        self.seq_feature_ids: Dict[str, List[int]] = {}
        self.seq_vocab_sizes: Dict[str, Dict[int, int]] = {}
        self.seq_domain_vocab_sizes: Dict[str, List[int]] = {}
        self.ts_fids: Dict[str, Optional[int]] = {}
        self.sideinfo_fids: Dict[str, List[int]] = {}
        self._seq_prefix: Dict[str, str] = {}
        self._seq_maxlen: Dict[str, int] = {}

        for domain in self.seq_domains:
            cfg = self._seq_cfg[domain]
            self._seq_prefix[domain] = cfg['prefix']
            ts_fid = cfg['ts_fid']
            self.ts_fids[domain] = ts_fid

            all_fids = [fid for fid, vs in cfg['features']]
            self.seq_feature_ids[domain] = all_fids
            self.seq_vocab_sizes[domain] = {fid: vs for fid, vs in cfg['features']}

            sideinfo = [fid for fid in all_fids if fid != ts_fid]
            self.sideinfo_fids[domain] = sideinfo
            self.seq_domain_vocab_sizes[domain] = [
                self.seq_vocab_sizes[domain][fid] for fid in sideinfo
            ]

            # max_len: from seq_max_lens arg; unspecified domains fall back to 256.
            self._seq_maxlen[domain] = seq_max_lens.get(domain, 256)

    def __len__(self) -> int:
        # Ceiling per Row Group; this is an upper bound on the true batch count.
        return sum((n + self.batch_size - 1) // self.batch_size
                   for _, _, n in self._rg_list)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        worker_info = torch.utils.data.get_worker_info()
        rg_list = self._rg_list
        if worker_info is not None and worker_info.num_workers > 1:
            rg_list = [rg for i, rg in enumerate(rg_list)
                       if i % worker_info.num_workers == worker_info.id]

        buffer: List[Dict[str, Any]] = []
        for file_path, rg_idx, _ in rg_list:
            pf = pq.ParquetFile(file_path)
            for batch in pf.iter_batches(batch_size=self.batch_size, row_groups=[rg_idx]):
                if self._row_filter is not None:
                    mask = self._row_filter(batch)
                    if not mask.any():
                        continue
                    batch = batch.filter(pa.array(mask))
                    if batch.num_rows == 0:
                        continue
                batch_dict = self._convert_batch(batch)
                if self.shuffle and self.buffer_batches > 1:
                    buffer.append(batch_dict)
                    if len(buffer) >= self.buffer_batches:
                        yield from self._flush_buffer(buffer)
                        buffer = []
                else:
                    yield batch_dict

        if buffer:
            yield from self._flush_buffer(buffer)

        del buffer
        gc.collect()

    def _flush_buffer(
        self, buffer: List[Dict[str, Any]]
    ) -> Iterator[Dict[str, Any]]:
        """Concatenate the buffered batches, shuffle at the row level, then
        re-slice and yield batch-sized chunks.
        """
        merged: Dict[str, torch.Tensor] = {}
        non_tensor_keys: Dict[str, Any] = {}
        for k in buffer[0].keys():
            if isinstance(buffer[0][k], torch.Tensor):
                merged[k] = torch.cat([b[k] for b in buffer], dim=0)
            else:
                non_tensor_keys[k] = buffer[0][k]
        total_rows = merged['label'].shape[0]
        rand_idx = torch.randperm(total_rows) if self.shuffle else torch.arange(total_rows)
        for i in range(0, total_rows, self.batch_size):
            end = min(i + self.batch_size, total_rows)
            batch: Dict[str, Any] = {k: v[rand_idx[i:end]] for k, v in merged.items()}
            batch.update(non_tensor_keys)
            yield batch
        del merged
        buffer.clear()

    # ---- Helpers ----

    def _record_oob(
        self,
        group: str,
        col_idx: int,
        arr: "npt.NDArray[np.int64]",
        vocab_size: int,
    ) -> None:
        """Record out-of-bound indices and (optionally) clip them to 0,
        without printing to the console.
        """
        oob_mask = arr >= vocab_size
        if not oob_mask.any():
            return
        key = (group, col_idx)
        oob_vals = arr[oob_mask]
        n = int(oob_mask.sum())
        mx = int(oob_vals.max())
        mn = int(oob_vals.min())
        if key in self._oob_stats:
            s = self._oob_stats[key]
            s['count'] += n
            s['max'] = max(s['max'], mx)
            s['min_oob'] = min(s['min_oob'], mn)
        else:
            self._oob_stats[key] = {
                'count': n, 'max': mx, 'min_oob': mn, 'vocab': vocab_size,
            }
        if self.clip_vocab:
            arr[oob_mask] = 0
        else:
            raise ValueError(
                f"{group} col_idx={col_idx}: {n} values out of range "
                f"[0, {vocab_size}), actual=[{mn}, {mx}]. "
                f"Use clip_vocab=True to clip or fix schema.json")

    def dump_oob_stats(self, path: Optional[str] = None) -> None:
        """Dump out-of-bound statistics to a file if ``path`` is provided,
        otherwise to ``logging.info``.
        """
        if not self._oob_stats:
            logging.info("No out-of-bound values detected.")
            return
        lines = ["=== Out-of-Bound Stats ==="]
        for (group, ci), s in sorted(self._oob_stats.items()):
            direction = "TOO_HIGH" if s['min_oob'] >= s['vocab'] else "TOO_LOW"
            lines.append(
                f"  {group} col_idx={ci}: vocab={s['vocab']}, "
                f"oob_count={s['count']}, range=[{s['min_oob']}, {s['max']}], "
                f"{direction}")
        msg = "\n".join(lines)
        if path:
            with open(path, 'w') as f:
                f.write(msg + "\n")
            logging.info(f"OOB stats written to {path}")
        else:
            logging.info(msg)

    def _pad_varlen_int_column(
        self,
        arrow_col: "pa.ListArray",
        max_len: int,
        B: int,
    ) -> Tuple["npt.NDArray[np.int64]", "npt.NDArray[np.int64]"]:
        """Pad an Arrow ``ListArray`` of ints to shape ``[B, max_len]``.

        Values <= 0 are mapped to 0 (padding). Note: the raw data contains -1
        (missing); currently treated the same way as 0 (padding).

        Returns:
            A tuple ``(padded, lengths)`` where ``padded`` has shape
            ``[B, max_len]`` and ``lengths`` has shape ``[B]``.
        """
        offsets = arrow_col.offsets.to_numpy()
        values = arrow_col.values.to_numpy()

        padded = np.zeros((B, max_len), dtype=np.int64)
        lengths = np.zeros(B, dtype=np.int64)

        for i in range(B):
            start, end = int(offsets[i]), int(offsets[i + 1])
            raw_len = end - start
            if raw_len <= 0:
                continue
            use_len = min(raw_len, max_len)
            padded[i, :use_len] = values[start:start + use_len]
            lengths[i] = use_len

        padded[padded <= 0] = 0
        return padded, lengths

    # Backwards-compatible alias kept for bench_raw_dataset.py and other
    # external callers that pre-date the rename. New code should call
    # `_pad_varlen_int_column` directly.
    _pad_varlen_column = _pad_varlen_int_column

    def _pad_varlen_float_column(
        self,
        arrow_col: "pa.ListArray",
        max_dim: int,
        B: int,
    ) -> "npt.NDArray[np.float32]":
        """Pad an Arrow ``ListArray<float>`` to shape ``[B, max_dim]``."""
        offsets = arrow_col.offsets.to_numpy()
        values = arrow_col.values.to_numpy()

        padded = np.zeros((B, max_dim), dtype=np.float32)

        for i in range(B):
            start, end = int(offsets[i]), int(offsets[i + 1])
            raw_len = end - start
            if raw_len <= 0:
                continue
            use_len = min(raw_len, max_dim)
            padded[i, :use_len] = values[start:start + use_len]

        return padded

    def _convert_batch(self, batch: "pa.RecordBatch") -> Dict[str, Any]:
        """Convert an Arrow RecordBatch into a training-ready dict of tensors."""
        B = batch.num_rows

        # ---- meta ----
        timestamps = batch.column(self._col_idx['timestamp']).to_numpy().astype(np.int64)
        if self.is_training:
            labels = (batch.column(self._col_idx['label_type']).fill_null(0)
                      .to_numpy(zero_copy_only=False).astype(np.int64) == 2).astype(np.int64)
        else:
            labels = np.zeros(B, dtype=np.int64)
        user_ids = batch.column(self._col_idx['user_id']).to_pylist()

        # ---- user_int: write into pre-allocated buffer ----
        # Note: null -> 0 (via fill_null), -1 -> 0 (via arr<=0); missing values
        # are treated the same as padding. Features with vs==0 have no vocab
        # information and are forced to 0 on the dataset side so that the
        # model's 1-slot Embedding (created for vs=0) is never indexed out of
        # range.
        user_int = self._buf_user_int[:B]
        user_int[:] = 0
        for ci, dim, offset, vs in self._user_int_plan:
            col = batch.column(ci)
            if dim == 1:
                arr = col.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)
                arr[arr <= 0] = 0
                if vs > 0:
                    self._record_oob('user_int', ci, arr, vs)
                else:
                    arr[:] = 0
                user_int[:, offset] = arr
            else:
                padded, _ = self._pad_varlen_int_column(col, dim, B)
                if vs > 0:
                    self._record_oob('user_int', ci, padded, vs)
                else:
                    padded[:] = 0
                user_int[:, offset:offset + dim] = padded

        # ---- item_int ----
        item_int = self._buf_item_int[:B]
        item_int[:] = 0
        for ci, dim, offset, vs in self._item_int_plan:
            col = batch.column(ci)
            if dim == 1:
                arr = col.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)
                arr[arr <= 0] = 0
                if vs > 0:
                    self._record_oob('item_int', ci, arr, vs)
                else:
                    arr[:] = 0
                item_int[:, offset] = arr
            else:
                padded, _ = self._pad_varlen_int_column(col, dim, B)
                if vs > 0:
                    self._record_oob('item_int', ci, padded, vs)
                else:
                    padded[:] = 0
                item_int[:, offset:offset + dim] = padded

        # ---- user_dense ----
        user_dense = self._buf_user_dense[:B]
        user_dense[:] = 0
        for ci, dim, offset in self._user_dense_plan:
            col = batch.column(ci)
            padded = self._pad_varlen_float_column(col, dim, B)
            user_dense[:, offset:offset + dim] = padded

        result = {
            'user_int_feats': torch.from_numpy(user_int.copy()),
            'user_dense_feats': torch.from_numpy(user_dense.copy()),
            'item_int_feats': torch.from_numpy(item_int.copy()),
            'item_dense_feats': torch.zeros(B, 0, dtype=torch.float32),
            'label': torch.from_numpy(labels),
            'timestamp': torch.from_numpy(timestamps),
            'user_id': user_ids,
            '_seq_domains': self.seq_domains,
        }

        # ---- Sequence features: fused padding directly into the 3D buffer ----
        for domain in self.seq_domains:
            max_len = self._seq_maxlen[domain]
            side_plan, ts_ci = self._seq_plan[domain]

            # Write directly into the pre-allocated 3D buffer.
            out = self._buf_seq[domain][:B]
            out[:] = 0
            lengths = self._buf_seq_lens[domain][:B]
            lengths[:] = 0

            # Fused path: first collect (offsets, values, vocab_size, col_idx)
            # for every side-info column, then fill the buffer in a single pass.
            col_data = []
            for ci, slot, vs in side_plan:
                col = batch.column(ci)
                col_data.append((col.offsets.to_numpy(), col.values.to_numpy(), vs, ci))

            for c, (offs, vals, vs, ci) in enumerate(col_data):
                for i in range(B):
                    s = int(offs[i])
                    e = int(offs[i + 1])
                    rl = e - s
                    if rl <= 0:
                        continue
                    ul = min(rl, max_len)
                    out[i, c, :ul] = vals[s:s + ul]
                    if ul > lengths[i]:
                        lengths[i] = ul

            # Values <= 0 -> 0.
            out[out <= 0] = 0

            # Check out-of-bound values per feature's vocab_size.
            # vs==0 means no vocab info; force the whole slice to 0 so that
            # the model's 1-slot Embedding is never indexed out of range.
            for c, (_, _, vs, ci) in enumerate(col_data):
                slice_c = out[:, c, :]
                if vs > 0:
                    self._record_oob(f'seq_{domain}', ci, slice_c, vs)
                else:
                    slice_c[:] = 0

            result[domain] = torch.from_numpy(out.copy())
            result[f'{domain}_len'] = torch.from_numpy(lengths.copy())

            # Time bucketing.
            time_bucket = self._buf_seq_tb[domain][:B]
            time_bucket[:] = 0
            if ts_ci is not None:
                ts_col = batch.column(ts_ci)
                ts_offs = ts_col.offsets.to_numpy()
                ts_vals = ts_col.values.to_numpy()
                # Pad timestamps into shape (B, max_len).
                ts_padded = np.zeros((B, max_len), dtype=np.int64)
                for i in range(B):
                    s = int(ts_offs[i])
                    e = int(ts_offs[i + 1])
                    rl = e - s
                    if rl <= 0:
                        continue
                    ul = min(rl, max_len)
                    ts_padded[i, :ul] = ts_vals[s:s + ul]

                ts_expanded = timestamps.reshape(-1, 1)
                time_diff = np.maximum(ts_expanded - ts_padded, 0)
                # np.searchsorted returns values in [0, len(BUCKET_BOUNDARIES)].
                # After +1 the nominal range is [1, len(BUCKET_BOUNDARIES)+1];
                # the upper bound only appears when time_diff exceeds the
                # largest boundary (~1 year) and would index past
                # nn.Embedding(NUM_TIME_BUCKETS=len(BUCKET_BOUNDARIES)+1).
                # Clip raw result to [0, len(BUCKET_BOUNDARIES)-1] so the final
                # bucket id (after +1) stays within [1, len(BUCKET_BOUNDARIES)]
                # and is always a valid Embedding index. Time-diffs beyond the
                # largest boundary collapse into the last bucket.
                raw_buckets = np.clip(
                    np.searchsorted(BUCKET_BOUNDARIES, time_diff.ravel()),
                    0, len(BUCKET_BOUNDARIES) - 1,
                )
                buckets = raw_buckets.reshape(B, max_len) + 1
                buckets[ts_padded == 0] = 0
                time_bucket[:] = buckets

            result[f'{domain}_time_bucket'] = torch.from_numpy(time_bucket.copy())

        return result


def get_pcvr_data(
    data_dir: str,
    schema_path: str,
    batch_size: int = 256,
    valid_ratio: float = 0.1,
    train_ratio: float = 1.0,
    num_workers: int = 16,
    buffer_batches: int = 20,
    shuffle_train: bool = True,
    seed: int = 42,
    clip_vocab: bool = True,
    seq_max_lens: Optional[Dict[str, int]] = None,
    **kwargs: Any,
) -> Tuple[DataLoader, DataLoader, PCVRParquetDataset]:
    """Create train / valid DataLoaders from raw multi-column Parquet files.

    The validation split is taken as the last ``valid_ratio`` fraction of Row
    Groups (in the file order returned by ``glob``).

    Returns:
        A tuple ``(train_loader, valid_loader, train_dataset)``. The third
        element is returned so the caller can access the feature schema
        (``user_int_schema``, ``item_int_schema``, ...) needed to construct
        the model.
    """
    random.seed(seed)

    import glob as _glob
    pq_files = sorted(_glob.glob(os.path.join(data_dir, '*.parquet')))

    rg_info = []
    for f in pq_files:
        pf = pq.ParquetFile(f)
        for i in range(pf.metadata.num_row_groups):
            rg_info.append((f, i, pf.metadata.row_group(i).num_rows))
    total_rgs = len(rg_info)

    n_valid_rgs = max(1, int(total_rgs * valid_ratio))
    n_train_rgs = total_rgs - n_valid_rgs

    # train_ratio: use only the first N% of the training Row Groups.
    if train_ratio < 1.0:
        n_train_rgs = max(1, int(n_train_rgs * train_ratio))
        logging.info(f"train_ratio={train_ratio}: using {n_train_rgs} train Row Groups")

    train_rows = sum(r[2] for r in rg_info[:n_train_rgs])
    valid_rows = sum(r[2] for r in rg_info[n_train_rgs:])

    logging.info(f"Row Group split: {n_train_rgs} train ({train_rows} rows), "
                 f"{n_valid_rgs} valid ({valid_rows} rows)")

    train_dataset = PCVRParquetDataset(
        parquet_path=data_dir,
        schema_path=schema_path,
        batch_size=batch_size,
        seq_max_lens=seq_max_lens,
        shuffle=shuffle_train,
        buffer_batches=buffer_batches,
        row_group_range=(0, n_train_rgs),
        clip_vocab=clip_vocab,
    )

    use_cuda = torch.cuda.is_available()
    _train_kw = {}
    if num_workers > 0:
        _train_kw['persistent_workers'] = True
        _train_kw['prefetch_factor'] = 2

    train_loader = DataLoader(
        train_dataset, batch_size=None,
        num_workers=num_workers, pin_memory=use_cuda, **_train_kw,
    )

    valid_dataset = PCVRParquetDataset(
        parquet_path=data_dir,
        schema_path=schema_path,
        batch_size=batch_size,
        seq_max_lens=seq_max_lens,
        shuffle=False,
        buffer_batches=0,
        row_group_range=(n_train_rgs, total_rgs),
        clip_vocab=clip_vocab,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=None,
        num_workers=0, pin_memory=use_cuda,
    )

    logging.info(f"Parquet train: {train_rows} rows, valid: {valid_rows} rows, "
                 f"batch_size={batch_size}, buffer_batches={buffer_batches}")

    return train_loader, valid_loader, train_dataset


# ─────────────────────────── v2 split helpers ────────────────────────────────
#
# Added by tencent-cc2 patch (2026-04-26). The organizer-shipped split keys
# the validation set on Row Group order, which (i) leaks future->past when
# parquet files are not pre-sorted by ``label_time`` and (ii) collapses to an
# empty training set on the demo (single Row Group). The functions below
# implement a ``label_time``-aware split with a 10% user_id OOF holdout
# (CLAUDE.md §4.3, §4.4) and write the three subsets as separate parquet
# files so the existing PCVRParquetDataset infrastructure can be reused.


def split_parquet_by_label_time(
    src_parquet: str,
    work_dir: str,
    valid_ratio: float = 0.1,
    oof_user_ratio: float = 0.1,
    seed: int = 42,
    schema_path: Optional[str] = None,
    pass1_batch: int = 200_000,
    pass2_batch: int = 50_000,
) -> Dict[str, Any]:
    """Streaming 2-pass split — production-scale safe (no full-table load).

    Pass 1: scan ``user_id`` + ``label_time`` columns only across all files
    (16 bytes/row peak). Compute OOF user set + label_time cutoff.

    Pass 2: stream row-groups via ``pq.iter_batches(batch_size=pass2_batch)``
    and write each batch to one of three subset parquet files via
    ``pq.ParquetWriter``. Memory peak = single batch (~tens of MB).

    Guarantees:
        - oof_user_ids sampled with ``np.random.default_rng(seed)``,
          appear in NEITHER train NOR valid.
        - ``max(train.label_time) <= cutoff <= min(valid.label_time)``
          (label_time-ordered cut on non-OOF rows).
        - cutoff = (1 - valid_ratio) quantile of non-OOF label_times.

    Side-effect file: ``${work_dir}/_split_meta.json``. ``schema.json`` is
    copied into each subset dir when ``schema_path`` is given.
    """
    from pathlib import Path as _P
    if os.path.isdir(src_parquet):
        files = sorted(str(p) for p in _P(src_parquet).rglob('*.parquet'))
    else:
        files = [src_parquet]
    if not files:
        raise FileNotFoundError(f"no parquet under {src_parquet}")

    logging.info(f"split: scanning {len(files)} parquet file(s) for user_id+label_time")

    # ---- Pass 1: ids + label_times only ----
    user_id_chunks = []
    label_time_chunks = []
    for fi, f in enumerate(files, 1):
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=pass1_batch,
                                     columns=['user_id', 'label_time']):
            user_id_chunks.append(
                batch.column('user_id').to_numpy(zero_copy_only=False).astype(np.int64, copy=False))
            label_time_chunks.append(
                batch.column('label_time').to_numpy(zero_copy_only=False).astype(np.int64, copy=False))
        n_so_far = sum(len(c) for c in user_id_chunks)
        logging.info(f"split pass 1 [{fi}/{len(files)}] {os.path.basename(f)}: cum {n_so_far} rows")

    user_ids = np.concatenate(user_id_chunks)
    label_times = np.concatenate(label_time_chunks)
    del user_id_chunks, label_time_chunks
    n_total = len(user_ids)
    if n_total == 0:
        raise RuntimeError("source parquet has 0 rows")

    rng = np.random.default_rng(seed)
    unique_users = np.unique(user_ids)
    n_oof_users = max(1, int(round(len(unique_users) * oof_user_ratio)))
    oof_user_arr = rng.choice(unique_users, size=n_oof_users, replace=False)
    oof_user_arr_sorted = np.sort(oof_user_arr.astype(np.int64, copy=False))
    oof_user_set_for_meta = set(int(u) for u in oof_user_arr_sorted[:20].tolist())

    is_oof_all = np.isin(user_ids, oof_user_arr_sorted)
    non_oof_lt = label_times[~is_oof_all]
    if len(non_oof_lt) == 0:
        raise RuntimeError("OOF holdout consumed all rows; lower oof_user_ratio")
    cutoff = float(np.quantile(non_oof_lt, max(0.0, 1.0 - valid_ratio)))
    n_oof_pre = int(is_oof_all.sum())
    n_total_users = int(len(unique_users))

    del user_ids, label_times, is_oof_all, non_oof_lt, unique_users
    logging.info(f"split: total_rows={n_total} unique_users={n_total_users} "
                 f"oof_users={n_oof_users} cutoff={cutoff:.0f}")

    # ---- Pass 2: stream + write 3 subset parquets ----
    work = _P(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    train_dir = work / 'train'
    valid_dir = work / 'valid'
    oof_dir = work / 'oof'
    for d in (train_dir, valid_dir, oof_dir):
        d.mkdir(parents=True, exist_ok=True)

    schema_arrow = pq.ParquetFile(files[0]).schema_arrow
    train_w = pq.ParquetWriter(str(train_dir / 'data.parquet'), schema_arrow)
    valid_w = pq.ParquetWriter(str(valid_dir / 'data.parquet'), schema_arrow)
    oof_w = pq.ParquetWriter(str(oof_dir / 'data.parquet'), schema_arrow)

    n_train = n_valid = n_oof = 0
    cum = 0
    try:
        for fi, f in enumerate(files, 1):
            pf = pq.ParquetFile(f)
            for batch in pf.iter_batches(batch_size=pass2_batch):
                table_chunk = pa.Table.from_batches([batch], schema=schema_arrow)
                ids = (batch.column('user_id')
                       .to_numpy(zero_copy_only=False).astype(np.int64, copy=False))
                lts = (batch.column('label_time')
                       .to_numpy(zero_copy_only=False).astype(np.int64, copy=False))
                is_oof = np.isin(ids, oof_user_arr_sorted)
                is_valid = (lts >= cutoff) & (~is_oof)
                is_train = (~is_valid) & (~is_oof)

                if is_train.any():
                    train_w.write_table(table_chunk.filter(pa.array(is_train)))
                    n_train += int(is_train.sum())
                if is_valid.any():
                    valid_w.write_table(table_chunk.filter(pa.array(is_valid)))
                    n_valid += int(is_valid.sum())
                if is_oof.any():
                    oof_w.write_table(table_chunk.filter(pa.array(is_oof)))
                    n_oof += int(is_oof.sum())
                cum += batch.num_rows
            logging.info(f"split pass 2 [{fi}/{len(files)}] {os.path.basename(f)}: "
                         f"cum={cum} train={n_train} valid={n_valid} oof={n_oof}")
    finally:
        train_w.close()
        valid_w.close()
        oof_w.close()

    if n_train == 0:
        raise RuntimeError(
            "train split is empty after streaming filter "
            f"(total={cum}, valid={n_valid}, oof={n_oof}); "
            "lower valid_ratio or check label_time distribution")

    meta: Dict[str, Any] = {
        'src': str(src_parquet),
        'seed': int(seed),
        'valid_ratio': float(valid_ratio),
        'oof_user_ratio': float(oof_user_ratio),
        'label_time_cutoff': cutoff,
        'n_train': n_train,
        'n_valid': n_valid,
        'n_oof': n_oof,
        'n_oof_users': int(n_oof_users),
        'n_total_users': n_total_users,
        'oof_user_ids_sample': sorted(oof_user_set_for_meta),
        'pass1_batch': pass1_batch,
        'pass2_batch': pass2_batch,
    }
    with open(work / '_split_meta.json', 'w') as fh:
        json.dump(meta, fh, indent=2)

    if schema_path and os.path.exists(schema_path):
        import shutil as _sh
        for d in (train_dir, valid_dir, oof_dir):
            _sh.copy2(schema_path, d / 'schema.json')

    logging.info(
        "split_parquet_by_label_time: train=%d valid=%d oof=%d "
        "(cutoff=%.0f, %d/%d OOF users)",
        n_train, n_valid, n_oof, cutoff, n_oof_users, n_total_users,
    )

    return {
        'train_dir': str(train_dir),
        'valid_dir': str(valid_dir),
        'oof_dir': str(oof_dir),
        'meta': meta,
        'meta_path': str(work / '_split_meta.json'),
    }


def get_pcvr_data_v2(
    data_dir: str,
    schema_path: str,
    work_dir: str,
    batch_size: int = 256,
    valid_ratio: float = 0.1,
    oof_user_ratio: float = 0.1,
    num_workers: int = 0,
    buffer_batches: int = 20,
    shuffle_train: bool = True,
    seed: int = 42,
    split_seed: Optional[int] = None,
    clip_vocab: bool = True,
    seq_max_lens: Optional[Dict[str, int]] = None,
    pass1_batch: int = 200_000,
    **kwargs: Any,
) -> Tuple[DataLoader, DataLoader, DataLoader, PCVRParquetDataset, Dict[str, Any]]:
    """Inline-filter v2 — NO subset materialization.

    Pass 1: scan ``user_id`` + ``label_time`` columns once across all files
    (16 bytes/row peak) to compute the OOF user set + label_time cutoff.
    Caches the result to ``${work_dir}/_split_meta.json``; subsequent runs
    with the same split config skip pass 1.

    Pass 2 is **eliminated** — no train/valid/oof.parquet files are ever
    written. Instead, three ``PCVRParquetDataset`` instances point at the
    same source files and differ only in their ``row_filter`` callable
    (``_RowFilter``), which masks rows in-flight inside ``__iter__``.

    For multi-seed campaigns, pass ``split_seed`` separate from training
    ``seed`` so OOF holdout stays constant across seeds (paired Δ rigor).
    Defaults: ``split_seed=42`` (independent of training seed) per CLAUDE.md
    §4.4. If only ``seed`` is provided, ``split_seed`` falls back to it.

    Returns: ``(train_loader, valid_loader, oof_loader, train_dataset,
    split_meta)``. ``oof_loader`` is for end-of-training generalization
    measurement only; never used for early stopping (CLAUDE.md §4.4).
    """
    if split_seed is None:
        split_seed = 42  # default: constant across training seeds
    work = __import__('pathlib').Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    meta_path = work / '_split_meta.json'

    cached = None
    if meta_path.exists():
        try:
            with open(meta_path, 'r') as fh:
                cached = json.load(fh)
            same = (
                int(cached.get('split_seed', cached.get('seed', -1))) == int(split_seed)
                and abs(float(cached.get('valid_ratio', -1)) - valid_ratio) < 1e-9
                and abs(float(cached.get('oof_user_ratio', -1)) - oof_user_ratio) < 1e-9
                and 'oof_user_ids' in cached
                and 'label_time_cutoff' in cached
            )
            if not same:
                cached = None
        except Exception:
            cached = None

    if cached is not None:
        logging.info(f"split: reusing cached metadata from {meta_path}")
        oof_users_sorted = np.sort(np.asarray(cached['oof_user_ids'], dtype=np.int64))
        cutoff = float(cached['label_time_cutoff'])
        n_total = int(cached.get('n_total', 0))
        n_total_users = int(cached.get('n_total_users', 0))
        n_oof_users = int(cached.get('n_oof_users', len(oof_users_sorted)))
    else:
        if os.path.isdir(data_dir):
            files = sorted(str(p) for p in
                           __import__('pathlib').Path(data_dir).rglob('*.parquet'))
        else:
            files = [data_dir]
        if not files:
            raise FileNotFoundError(f"no parquet under {data_dir}")
        logging.info(f"split pass 1: scanning {len(files)} parquet file(s) "
                     f"for user_id+label_time only (split_seed={split_seed})")

        user_id_chunks = []
        label_time_chunks = []
        for fi, f in enumerate(files, 1):
            pf = pq.ParquetFile(f)
            for batch in pf.iter_batches(batch_size=pass1_batch,
                                         columns=['user_id', 'label_time']):
                user_id_chunks.append(
                    batch.column('user_id').to_numpy(zero_copy_only=False)
                    .astype(np.int64, copy=False))
                label_time_chunks.append(
                    batch.column('label_time').to_numpy(zero_copy_only=False)
                    .astype(np.int64, copy=False))
            n_so_far = sum(len(c) for c in user_id_chunks)
            logging.info(f"split pass 1 [{fi}/{len(files)}] "
                         f"{os.path.basename(f)}: cum {n_so_far} rows")

        user_ids = np.concatenate(user_id_chunks)
        label_times = np.concatenate(label_time_chunks)
        del user_id_chunks, label_time_chunks
        n_total = int(len(user_ids))
        if n_total == 0:
            raise RuntimeError("source parquet has 0 rows")

        rng = np.random.default_rng(int(split_seed))
        unique_users = np.unique(user_ids)
        n_total_users = int(len(unique_users))
        n_oof_users = max(1, int(round(n_total_users * oof_user_ratio)))
        oof_user_arr = rng.choice(unique_users, size=n_oof_users, replace=False)
        oof_users_sorted = np.sort(oof_user_arr.astype(np.int64, copy=False))

        is_oof_all = np.isin(user_ids, oof_users_sorted)
        non_oof_lt = label_times[~is_oof_all]
        if len(non_oof_lt) == 0:
            raise RuntimeError("OOF holdout consumed all rows; lower oof_user_ratio")
        cutoff = float(np.quantile(non_oof_lt, max(0.0, 1.0 - valid_ratio)))
        del user_ids, label_times, is_oof_all, non_oof_lt, unique_users

        meta_to_cache: Dict[str, Any] = {
            'src': str(data_dir),
            'split_seed': int(split_seed),
            'valid_ratio': float(valid_ratio),
            'oof_user_ratio': float(oof_user_ratio),
            'label_time_cutoff': cutoff,
            'n_total': n_total,
            'n_total_users': n_total_users,
            'n_oof_users': int(n_oof_users),
            'oof_user_ids': sorted(int(u) for u in oof_users_sorted.tolist()),
            'mode': 'inline_filter',
        }
        with open(meta_path, 'w') as fh:
            json.dump(meta_to_cache, fh, indent=2)
        logging.info(f"split: total_rows={n_total} unique_users={n_total_users} "
                     f"oof_users={n_oof_users} cutoff={cutoff:.0f} "
                     f"(cached to {meta_path})")

    random.seed(seed)

    train_filter = _RowFilter(oof_users_sorted, cutoff, 'train')
    valid_filter = _RowFilter(oof_users_sorted, cutoff, 'valid')
    oof_filter = _RowFilter(oof_users_sorted, cutoff, 'oof')

    train_dataset = PCVRParquetDataset(
        parquet_path=data_dir,
        schema_path=schema_path,
        batch_size=batch_size,
        seq_max_lens=seq_max_lens,
        shuffle=shuffle_train,
        buffer_batches=buffer_batches,
        clip_vocab=clip_vocab,
        is_training=True,
        row_filter=train_filter,
    )
    valid_dataset = PCVRParquetDataset(
        parquet_path=data_dir,
        schema_path=schema_path,
        batch_size=batch_size,
        seq_max_lens=seq_max_lens,
        shuffle=False,
        buffer_batches=0,
        clip_vocab=clip_vocab,
        is_training=True,
        row_filter=valid_filter,
    )
    oof_dataset = PCVRParquetDataset(
        parquet_path=data_dir,
        schema_path=schema_path,
        batch_size=batch_size,
        seq_max_lens=seq_max_lens,
        shuffle=False,
        buffer_batches=0,
        clip_vocab=clip_vocab,
        is_training=True,
        row_filter=oof_filter,
    )

    use_cuda = torch.cuda.is_available()
    train_kw: Dict[str, Any] = {}
    if num_workers > 0:
        train_kw['persistent_workers'] = True
        train_kw['prefetch_factor'] = 2

    train_loader = DataLoader(
        train_dataset, batch_size=None,
        num_workers=num_workers, pin_memory=use_cuda, **train_kw,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=None,
        num_workers=0, pin_memory=use_cuda,
    )
    oof_loader = DataLoader(
        oof_dataset, batch_size=None,
        num_workers=0, pin_memory=use_cuda,
    )

    meta_for_return: Dict[str, Any] = {
        'split_seed': int(split_seed),
        'valid_ratio': float(valid_ratio),
        'oof_user_ratio': float(oof_user_ratio),
        'label_time_cutoff': cutoff,
        'n_total': n_total,
        'n_total_users': n_total_users,
        'n_oof_users': n_oof_users,
        'mode': 'inline_filter',
        'cache_path': str(meta_path),
    }
    return train_loader, valid_loader, oof_loader, train_dataset, meta_for_return
