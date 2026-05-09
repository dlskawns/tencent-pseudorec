"""TAAC 2026 UNI-REC submission — inference entry point.

Contract (CLAUDE.md §13):
- ``def main()`` takes zero args, directly executable.
- All paths come from environment variables (no hardcoded literals).
- Output: ``${EVAL_RESULT_PATH}/predictions.json`` with strict format
    {"predictions": {"<user_id>": <float in [0,1]>, ...}}
- Every test ``user_id`` appears exactly once.

Resolution order at runtime:
  1. If ``MODEL_OUTPUT_PATH`` resolves to a PCVRHyFormer checkpoint
     (model.pt + schema.json + train_config.json), load it via the bundled
     ``competition/`` modules and run forward inference.
  2. Else if ``MODEL_OUTPUT_PATH/prior.json`` exists, use that prior.
  3. Else fall back to a hardcoded class-prior.

The script imports torch lazily so that a torch-less eval container still
runs the prior path successfully.
"""

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow.parquet as pq


DUP_USER_POLICY = "mean"
RNG_SEED = 42
DEFAULT_PRIOR = 0.124
PROB_FLOOR = 1e-6
PROB_CEIL = 1.0 - 1e-6


def _list_parquet(eval_data: str) -> list[str]:
    p = Path(eval_data)
    if p.is_file() and p.suffix == ".parquet":
        return [str(p)]
    if not p.is_dir():
        raise RuntimeError(f"EVAL_DATA_PATH not found or not parquet/dir: {eval_data}")
    files = sorted(str(x) for x in p.rglob("*.parquet"))
    if not files:
        raise RuntimeError(f"No .parquet files under EVAL_DATA_PATH={eval_data}")
    return files


def _load_prior(model_output_path: str | None) -> float:
    if not model_output_path:
        return DEFAULT_PRIOR
    candidate = Path(model_output_path) / "prior.json"
    if not candidate.exists():
        return DEFAULT_PRIOR
    try:
        with candidate.open("r") as fh:
            obj = json.load(fh)
        prior = float(obj.get("prior", DEFAULT_PRIOR))
        return float(np.clip(prior, PROB_FLOOR, PROB_CEIL))
    except Exception:
        return DEFAULT_PRIOR


def _find_competition_dir() -> str | None:
    """Locate the directory shipping ``model.py``/``dataset.py``.

    Resolution order:
    1. ``competition/`` subdir or parent's ``competition/`` (project layout).
    2. The directory of ``infer.py`` itself (TAAC platform flat-upload layout —
       all code files live at the same namespace level).
    """
    here = Path(__file__).resolve().parent
    for candidate in (here / "competition", here.parent / "competition",
                      Path.cwd() / "competition"):
        if (candidate / "model.py").exists() and (candidate / "dataset.py").exists():
            return str(candidate)
    if (here / "model.py").exists() and (here / "dataset.py").exists():
        return str(here)
    return None


def _find_best_ckpt_dir(ckpt_root: str) -> str | None:
    if not ckpt_root or not os.path.isdir(ckpt_root):
        return None
    def _step(p: str) -> int:
        m = re.search(r"global_step(\d+)", os.path.basename(p))
        return int(m.group(1)) if m else -1

    import glob
    best = sorted(glob.glob(os.path.join(ckpt_root, "global_step*.best_model")), key=_step)
    if best:
        return best[-1]
    other = sorted(glob.glob(os.path.join(ckpt_root, "global_step*")), key=_step)
    if other:
        return other[-1]
    if all(os.path.exists(os.path.join(ckpt_root, n)) for n in ("model.pt", "schema.json", "train_config.json")):
        return ckpt_root
    return None


def _try_torch_inference(ckpt_dir: str, eval_files: list[str]):
    """Returns dict {user_id -> probability} or ``None`` on any failure."""
    try:
        import torch
        from torch.utils.data import DataLoader
    except Exception:
        return None

    comp_dir = _find_competition_dir()
    if comp_dir is None:
        return None
    if comp_dir not in sys.path:
        sys.path.insert(0, comp_dir)
    try:
        from dataset import PCVRParquetDataset, NUM_TIME_BUCKETS  # type: ignore
        from model import PCVRHyFormer, ModelInput  # type: ignore
    except Exception:
        return None

    schema_path = os.path.join(ckpt_dir, "schema.json")
    cfg_path = os.path.join(ckpt_dir, "train_config.json")
    model_pt = os.path.join(ckpt_dir, "model.pt")
    if not all(os.path.exists(p) for p in (schema_path, cfg_path, model_pt)):
        return None

    with open(cfg_path, "r") as fh:
        cfg = json.load(fh)

    seq_max_lens = {}
    if cfg.get("seq_max_lens"):
        for pair in cfg["seq_max_lens"].split(","):
            k, v = pair.split(":")
            seq_max_lens[k.strip()] = int(v.strip())

    eval_data_dir = str(Path(eval_files[0]).parent) if len(eval_files) == 1 else os.environ["EVAL_DATA_PATH"]
    # Inference batch size: MUST be passed at construction time so the
    # dataset's internal buffers (`_buf_user_int`, `_buf_item_int`, etc.) are
    # sized correctly — overriding `eval_ds.batch_size` after init produces
    # buffer/batch shape mismatches at file boundaries.
    # Default 256: Taiji eval is on a shared vGPU partition with highly
    # variable free memory (other tenants hog it). Larger batches (1024)
    # reliably OOM at sum-pool / attention layers. 256 is the safe baseline.
    # Override with INFER_BATCH_SIZE env var if the partition has free room.
    infer_batch_size = int(os.environ.get("INFER_BATCH_SIZE", "256"))
    # Reduce CUDA memory fragmentation on tight vGPU partitions.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    eval_ds = PCVRParquetDataset(
        parquet_path=eval_data_dir,
        schema_path=schema_path,
        batch_size=infer_batch_size,
        seq_max_lens=seq_max_lens,
        shuffle=False,
        buffer_batches=0,
        clip_vocab=True,
        is_training=False,
    )

    user_int_specs = [(max(eval_ds.user_int_vocab_sizes[off:off + ln]), off, ln)
                      for _, off, ln in eval_ds.user_int_schema.entries]
    item_int_specs = [(max(eval_ds.item_int_vocab_sizes[off:off + ln]), off, ln)
                      for _, off, ln in eval_ds.item_int_schema.entries]
    user_ns_groups = [[i] for i in range(len(eval_ds.user_int_schema.entries))]
    item_ns_groups = [[i] for i in range(len(eval_ds.item_int_schema.entries))]

    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"

    model = PCVRHyFormer(
        user_int_feature_specs=user_int_specs,
        item_int_feature_specs=item_int_specs,
        user_dense_dim=eval_ds.user_dense_schema.total_dim,
        item_dense_dim=eval_ds.item_dense_schema.total_dim,
        seq_vocab_sizes=eval_ds.seq_domain_vocab_sizes,
        user_ns_groups=user_ns_groups,
        item_ns_groups=item_ns_groups,
        d_model=cfg.get("d_model", 64),
        emb_dim=cfg.get("emb_dim", 64),
        num_queries=cfg.get("num_queries", 1),
        num_hyformer_blocks=cfg.get("num_hyformer_blocks", 2),
        num_heads=cfg.get("num_heads", 4),
        seq_encoder_type=cfg.get("seq_encoder_type", "transformer"),
        hidden_mult=cfg.get("hidden_mult", 4),
        dropout_rate=cfg.get("dropout_rate", 0.01),
        seq_top_k=cfg.get("seq_top_k", 50),
        seq_causal=cfg.get("seq_causal", False),
        action_num=cfg.get("action_num", 1),
        num_time_buckets=NUM_TIME_BUCKETS if cfg.get("use_time_buckets", True) else 0,
        rank_mixer_mode=cfg.get("rank_mixer_mode", "full"),
        use_rope=cfg.get("use_rope", False),
        rope_base=cfg.get("rope_base", 10000.0),
        emb_skip_threshold=cfg.get("emb_skip_threshold", 0),
        seq_id_threshold=cfg.get("seq_id_threshold", 10000),
        ns_tokenizer_type=cfg.get("ns_tokenizer_type", "rankmixer"),
        user_ns_tokens=cfg.get("user_ns_tokens", 0),
        item_ns_tokens=cfg.get("item_ns_tokens", 0),
        # H004 anchor — backbone routing (defaults preserve hyformer compatibility)
        backbone=cfg.get("backbone", "hyformer"),
        num_onetrans_layers=cfg.get("num_onetrans_layers", 2),
        mixed_causal_anchor=cfg.get("mixed_causal_anchor", "timestamp"),
        domain_id_embedding=cfg.get("domain_id_embedding", True),
        log_attn_entropy=False,  # never log entropy during inference
        # H008 anchor — fusion mechanism dispatch
        fusion_type=cfg.get("fusion_type", "rankmixer"),
        dcn_v2_num_layers=cfg.get("dcn_v2_num_layers", 2),
        dcn_v2_rank=cfg.get("dcn_v2_rank", 8),
        # H010 — NS→S full bidirectional cross-attention
        use_ns_to_s_xattn=cfg.get("use_ns_to_s_xattn", False),
        ns_xattn_num_heads=cfg.get("ns_xattn_num_heads", 4),
        # H019 — TWIN long-seq retrieval
        use_twin_retrieval=cfg.get("use_twin_retrieval", False),
        twin_top_k=cfg.get("twin_top_k", 64),
        twin_num_heads=cfg.get("twin_num_heads", 4),
        twin_gate_init=cfg.get("twin_gate_init", -2.0),
        # H047 — per-domain auxiliary heads (heads needed for state_dict load)
        use_per_domain_aux=cfg.get("use_per_domain_aux", False),
    ).to(device)

    state = torch.load(model_pt, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    # num_workers: 2 by default — parallel data load while GPU is busy.
    # Taiji + CFS deadlocked at 8 during training; 2 has been safe across H001/H002/H004/H005.
    # Set INFER_NUM_WORKERS=0 if eval container differs and deadlocks.
    infer_workers = int(os.environ.get("INFER_NUM_WORKERS", "2"))
    loader = DataLoader(
        eval_ds, batch_size=None,
        num_workers=infer_workers, pin_memory=use_cuda,
        persistent_workers=(infer_workers > 0),
        prefetch_factor=2 if infer_workers > 0 else None,
    )

    sums: dict[str, float] = defaultdict(float)
    cnts: dict[str, int] = defaultdict(int)
    heartbeat_every = int(os.environ.get("INFER_HEARTBEAT_EVERY_N_BATCHES", "50"))
    n_batches = 0
    print(f"[infer] starting inference loop (num_workers={infer_workers}, "
          f"batch_size={infer_batch_size}, heartbeat_every={heartbeat_every})", flush=True)

    with torch.no_grad():
        for batch in loader:
            n_batches += 1
            uids = [str(u) for u in batch["user_id"]]
            seq_domains = batch["_seq_domains"]
            seq_data = {d: batch[d].to(device) for d in seq_domains}
            seq_lens = {d: batch[f"{d}_len"].to(device) for d in seq_domains}
            seq_tb = {d: batch.get(f"{d}_time_bucket", torch.zeros_like(batch[f"{d}_len"])).to(device)
                      for d in seq_domains}
            mi = ModelInput(
                user_int_feats=batch["user_int_feats"].to(device),
                item_int_feats=batch["item_int_feats"].to(device),
                user_dense_feats=batch["user_dense_feats"].to(device),
                item_dense_feats=batch["item_dense_feats"].to(device),
                seq_data=seq_data,
                seq_lens=seq_lens,
                seq_time_buckets=seq_tb,
            )
            logits, _ = model.predict(mi)
            probs = torch.sigmoid(logits.squeeze(-1)).detach().cpu().numpy()
            probs = np.where(np.isnan(probs), DEFAULT_PRIOR, probs)
            probs = np.clip(probs, PROB_FLOOR, PROB_CEIL).astype(float)
            for uid, p in zip(uids, probs):
                sums[uid] += float(p)
                cnts[uid] += 1

            if n_batches == 1 or n_batches % heartbeat_every == 0:
                print(f"[infer] batch {n_batches} processed_users={len(sums)}",
                      flush=True)

    print(f"[infer] inference loop done: {n_batches} batches, "
          f"{len(sums)} unique users", flush=True)
    return {uid: sums[uid] / cnts[uid] for uid in sums}


def _iter_batches(files: Iterable[str], batch_rows: int = 1024):
    for path in files:
        pf = pq.ParquetFile(path)
        for rb in pf.iter_batches(batch_size=batch_rows, columns=["user_id"]):
            yield rb


def _heuristic_predictions(eval_files: list[str], prior: float) -> dict[str, float]:
    sums: dict[str, float] = defaultdict(float)
    cnts: dict[str, int] = defaultdict(int)
    for rb in _iter_batches(eval_files):
        if "user_id" not in rb.schema.names:
            raise RuntimeError("test parquet missing required column: user_id")
        for uid in rb.column("user_id").to_pylist():
            key = str(uid)
            sums[key] += prior
            cnts[key] += 1
    return {uid: sums[uid] / cnts[uid] for uid in sums}


def main() -> None:
    eval_data = os.environ["EVAL_DATA_PATH"]
    eval_out = os.environ["EVAL_RESULT_PATH"]
    model_output = os.environ.get("MODEL_OUTPUT_PATH")

    np.random.seed(RNG_SEED)
    files = _list_parquet(eval_data)
    prior = _load_prior(model_output)

    preds: dict[str, float] | None = None
    ckpt_dir = _find_best_ckpt_dir(model_output) if model_output else None
    print(f"[infer] MODEL_OUTPUT_PATH={model_output} ckpt_dir={ckpt_dir}", flush=True)
    if ckpt_dir is not None:
        try:
            preds = _try_torch_inference(ckpt_dir, files)
            if preds is None:
                print("[infer] WARNING: torch path returned None (sidecar missing or schema fail)", flush=True)
        except Exception as e:
            import traceback
            print(f"[infer] WARNING: torch path raised {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            preds = None
    else:
        print("[infer] WARNING: no ckpt_dir found, skipping torch path", flush=True)
    if preds is None:
        print("[infer] FALLBACK: using heuristic prior", flush=True)
        preds = _heuristic_predictions(files, prior)
    else:
        print(f"[infer] OK: torch path produced {len(preds)} predictions", flush=True)

    preds = {uid: float(np.clip(v, PROB_FLOOR, PROB_CEIL)) for uid, v in preds.items()}

    out_dir = Path(eval_out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "predictions.json"
    tmp_path = out_dir / "predictions.json.tmp"
    with tmp_path.open("w") as fh:
        json.dump({"predictions": preds}, fh, sort_keys=True, separators=(",", ":"))
    tmp_path.replace(out_path)
    print(f"[infer] wrote {len(preds)} predictions to {out_path}", flush=True)


if __name__ == "__main__":
    main()
