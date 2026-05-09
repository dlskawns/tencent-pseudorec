"""PCVRHyFormer training entry point (self-contained baseline).

Usage:
    python train.py [--num_epochs 10] [--batch_size 256] ...

Environment variables (take precedence over CLI flags):
    TRAIN_DATA_PATH  Training data directory (*.parquet + schema.json)
    TRAIN_CKPT_PATH  Checkpoint output directory
    TRAIN_LOG_PATH   Log directory
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import torch

from utils import set_seed, EarlyStopping, create_logger
from dataset import (
    FeatureSchema,
    get_pcvr_data,
    get_pcvr_data_v2,
    NUM_TIME_BUCKETS,
)
from model import PCVRHyFormer
from trainer import PCVRHyFormerRankingTrainer


def build_feature_specs(
    schema: FeatureSchema,
    per_position_vocab_sizes: List[int],
) -> List[Tuple[int, int, int]]:
    """Build feature_specs of the form ``[(vocab_size, offset, length), ...]``
    ordered by the positions recorded in ``schema.entries``.
    """
    specs: List[Tuple[int, int, int]] = []
    for fid, offset, length in schema.entries:
        vs = max(per_position_vocab_sizes[offset:offset + length])
        specs.append((vs, offset, length))
    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PCVRHyFormer Training")

    # Paths (environment variables take precedence).
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Training data directory (env: TRAIN_DATA_PATH)')
    parser.add_argument('--schema_path', type=str, default=None,
                        help='Schema JSON path (defaults to <data_dir>/schema.json)')
    parser.add_argument('--ckpt_dir', type=str, default=None,
                        help='Checkpoint output directory (env: TRAIN_CKPT_PATH)')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Log directory (env: TRAIN_LOG_PATH)')

    # Training hyperparameters.
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for both training and validation')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for dense parameters (AdamW)')
    parser.add_argument('--num_epochs', type=int, default=999,
                        help='Maximum number of training epochs '
                             '(typically terminated earlier by early stopping)')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early-stopping patience '
                             '(number of validations without improvement)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Training device, e.g. cuda or cpu')

    # Data pipeline.
    parser.add_argument('--num_workers', type=int, default=16,
                        help='Number of DataLoader workers')
    parser.add_argument('--buffer_batches', type=int, default=20,
                        help='Shuffle buffer size, in units of batches. '
                             'Lower values reduce memory usage.')
    parser.add_argument('--train_ratio', type=float, default=1.0,
                        help='Fraction of training Row Groups to use (takes the first N%)')
    parser.add_argument('--valid_ratio', type=float, default=0.1,
                        help='Fraction of all Row Groups used for validation (takes the tail)')
    parser.add_argument('--eval_every_n_steps', type=int, default=0,
                        help='Run validation every N steps '
                             '(0 = only at the end of each epoch)')
    parser.add_argument('--seq_max_lens', type=str,
                        default='seq_a:256,seq_b:256,seq_c:512,seq_d:512',
                        help='Per-domain sequence truncation, format: seq_d:256,seq_c:128')

    # Model hyperparameters.
    parser.add_argument('--d_model', type=int, default=64,
                        help='Backbone hidden dimension (output size of each block)')
    parser.add_argument('--emb_dim', type=int, default=64,
                        help='Per-Embedding-table dimension (before projection)')
    parser.add_argument('--num_queries', type=int, default=1,
                        help='Number of Query tokens generated independently per sequence domain')
    parser.add_argument('--num_hyformer_blocks', type=int, default=2,
                        help='Number of stacked MultiSeqHyFormerBlock layers')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Number of attention heads (must satisfy d_model %% num_heads == 0)')
    parser.add_argument('--seq_encoder_type', type=str, default='transformer',
                        choices=['swiglu', 'transformer', 'longer'],
                        help='Sequence encoder variant: '
                             'swiglu = SwiGLU without attention, '
                             'transformer = standard self-attention, '
                             'longer = Top-K compressed encoder '
                             '(only this variant consumes --seq_top_k / --seq_causal)')
    parser.add_argument('--hidden_mult', type=int, default=4,
                        help='FFN inner-dim multiplier relative to d_model')
    parser.add_argument('--dropout_rate', type=float, default=0.01,
                        help='Dropout rate for the backbone '
                             '(seq id-embedding dropout is twice this value)')
    parser.add_argument('--seq_top_k', type=int, default=50,
                        help='Number of most-recent tokens kept by LongerEncoder '
                             '(only effective when --seq_encoder_type=longer)')
    parser.add_argument('--seq_causal', action='store_true', default=False,
                        help='Whether the LongerEncoder self-attention uses a causal mask '
                             '(only effective when --seq_encoder_type=longer)')
    parser.add_argument('--action_num', type=int, default=1,
                        help='Classifier output dimension '
                             '(1 = single binary-classification logit; >1 = multi-label)')
    parser.add_argument('--use_time_buckets', action='store_true', default=True,
                        help='Enable the time-bucket embedding (default on). '
                             'The actual bucket count is uniquely determined by '
                             'dataset.BUCKET_BOUNDARIES; this flag is a pure on/off switch.')
    parser.add_argument('--no_time_buckets', dest='use_time_buckets', action='store_false',
                        help='Disable the time-bucket embedding')
    parser.add_argument('--rank_mixer_mode', type=str, default='full',
                        choices=['full', 'ffn_only', 'none'],
                        help='RankMixerBlock mode: '
                             'full = token mixing + per-token FFN (requires d_model divisible by T), '
                             'ffn_only = per-token FFN only, '
                             'none = identity passthrough')
    parser.add_argument('--use_rope', action='store_true', default=False,
                        help='Enable RoPE positional encoding in sequence attention')
    parser.add_argument('--rope_base', type=float, default=10000.0,
                        help='RoPE base frequency (default 10000)')

    # Loss function.
    parser.add_argument('--loss_type', type=str, default='bce', choices=['bce', 'focal'],
                        help='Loss type: bce = BCEWithLogits, focal = Focal Loss')
    parser.add_argument('--focal_alpha', type=float, default=0.1,
                        help='Focal Loss positive-class weight alpha '
                             '(effective only when --loss_type=focal)')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal Loss focusing parameter gamma '
                             '(effective only when --loss_type=focal)')

    # Sparse optimizer.
    parser.add_argument('--sparse_lr', type=float, default=0.05,
                        help='Learning rate for sparse parameters (Adagrad over Embeddings)')
    parser.add_argument('--sparse_weight_decay', type=float, default=0.0,
                        help='Weight decay for sparse parameters (Adagrad over Embeddings)')
    parser.add_argument('--reinit_sparse_after_epoch', type=int, default=1,
                        help='Starting from the N-th epoch, at the end of every epoch '
                             're-initialize Embeddings with vocab_size > '
                             '--reinit_cardinality_threshold and rebuild the Adagrad '
                             'optimizer state (cold-restart trick for high-cardinality '
                             'features to reduce overfitting)')
    parser.add_argument('--reinit_cardinality_threshold', type=int, default=0,
                        help='Cardinality threshold used by the re-init strategy: '
                             'Embeddings whose vocab_size exceeds this value are reset '
                             'at each epoch end (0 = never reset any Embedding)')

    # Embedding construction control.
    parser.add_argument('--emb_skip_threshold', type=int, default=0,
                        help='At model construction time, features whose vocab_size '
                             'exceeds this value get no Embedding and are represented '
                             'by a zero vector at forward time (0 = no skipping; '
                             'all features get an Embedding). Useful for saving GPU '
                             'memory on ultra-high-cardinality features.')
    parser.add_argument('--seq_id_threshold', type=int, default=10000,
                        help='Within the sequence tokenizer, features with vocab_size '
                             'exceeding this value are treated as id features and receive '
                             'extra dropout(rate*2) during training to reduce overfitting. '
                             'Features at or below this threshold are treated as side-info '
                             'and receive no extra dropout.')

    _default_ns_groups = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'ns_groups.json')
    parser.add_argument('--ns_groups_json', type=str, default=_default_ns_groups,
                        help='Path to the NS-groups JSON file. If it does not exist, '
                             'each feature is placed in its own singleton group.')

    # NS tokenizer variant.
    parser.add_argument('--ns_tokenizer_type', type=str, default='rankmixer',
                        choices=['group', 'rankmixer'],
                        help='NS tokenizer variant: '
                             'group = project each group to one token, '
                             'rankmixer = concatenate all embeddings then split into '
                             'equal-size chunks (token count is tunable)')
    parser.add_argument('--user_ns_tokens', type=int, default=0,
                        help='Number of user NS tokens in rankmixer mode '
                             '(0 = automatically use the number of user groups)')
    parser.add_argument('--item_ns_tokens', type=int, default=0,
                        help='Number of item NS tokens in rankmixer mode '
                             '(0 = automatically use the number of item groups)')

    # ---- H004 anchor: backbone selection ----
    parser.add_argument('--backbone', type=str, default='hyformer',
                        choices=['hyformer', 'onetrans'],
                        help='Backbone architecture: hyformer (PCVRHyFormer baseline) '
                             'or onetrans (single-stream + mixed-causal block).')
    parser.add_argument('--num_onetrans_layers', type=int, default=2,
                        help='Number of OneTrans single-stream blocks. '
                             'Effective only with --backbone onetrans.')
    parser.add_argument('--mixed_causal_anchor', type=str, default='timestamp',
                        choices=['timestamp', 'seq_index'],
                        help='Mask anchor for NS->S attention: timestamp (paper, '
                             "candidate-time-filtered) or seq_index (padding-only fallback).")
    parser.add_argument('--domain_id_embedding', action='store_true', default=True,
                        help='Add learnable domain ID embedding to S-tokens (OneTrans only).')
    parser.add_argument('--no_domain_id_embedding', dest='domain_id_embedding',
                        action='store_false',
                        help='Disable domain ID embedding for OneTrans S-tokens.')
    parser.add_argument('--log_attn_entropy', action='store_true', default=False,
                        help='Log per-layer attention entropy (CLAUDE.md §10.9 abort threshold). '
                             'Effective only with --backbone onetrans.')

    # ---- H008 anchor: fusion mechanism dispatch (block-level) ----
    parser.add_argument('--fusion_type', type=str, default='rankmixer',
                        choices=['rankmixer', 'dcn_v2'],
                        help='H008: token fusion at MultiSeqHyFormerBlock step 3. '
                             '"rankmixer" = RankMixerBlock token-mixing (default, anchor). '
                             '"dcn_v2" = DCNV2CrossBlock explicit polynomial cross with x_0 residual '
                             '(Wang et al. WWW 2021).')
    parser.add_argument('--dcn_v2_num_layers', type=int, default=2,
                        help='Number of stacked DCN-V2 cross layers (degree = N+1). '
                             'Effective only with --fusion_type dcn_v2.')
    parser.add_argument('--use_ns_to_s_xattn', action='store_true', default=False,
                        help='H010: enable NS→S full bidirectional cross-attention '
                             '(paper-grade OneTrans NS→S half). NS tokens (Q) attend to '
                             'all per-domain S tokens concatenated (K=V). NS dimension '
                             'preserved → DCN-V2 fusion input unchanged → H009 위치 충돌 회피.')
    parser.add_argument('--ns_xattn_num_heads', type=int, default=4,
                        help='H010: NS xattn num_heads (default = num_heads). '
                             'Effective only with --use_ns_to_s_xattn.')

    # H019 — TWIN long-seq retrieval
    parser.add_argument('--use_twin_retrieval', action='store_true', default=False,
                        help='H019: enable per-domain TWIN GSU+ESU retrieval. '
                             'GSU = parameter-free inner product, top_k filter, '
                             'ESU = MultiHeadAttention with candidate Q. 4 (B, D) → '
                             'mean → Linear → residual ADD post-backbone. Gate init '
                             'sigmoid(-2)≈0.12 (§10.10). Pair with longer --seq_max_lens '
                             'for tail access. EDA: §3.5 seq p90=1393~2215.')
    parser.add_argument('--twin_top_k', type=int, default=64,
                        help='H019: GSU top-K filter K. Effective with --use_twin_retrieval.')
    parser.add_argument('--twin_num_heads', type=int, default=4,
                        help='H019: ESU MHA num_heads. Effective with --use_twin_retrieval.')
    parser.add_argument('--twin_gate_init', type=float, default=-2.0,
                        help='H019: twin_gate init (sigmoid(-2.0)≈0.12 per §10.10).')

    # H052 — User-Item Contrastive auxiliary (InfoNCE on representations)
    parser.add_argument('--use_user_item_contrast', action='store_true', default=False,
                        help='H052: enable user-item InfoNCE contrastive aux on backbone output × item_ns mean. '
                             'Positive = same-row (user_i, item_i). Negatives = in-batch other rows.')
    parser.add_argument('--contrast_lambda', type=float, default=0.1,
                        help='H052: weight for InfoNCE aux loss (0.0 → H019 byte-identical).')
    parser.add_argument('--contrast_temperature', type=float, default=0.1,
                        help='H052: InfoNCE temperature (lower = sharper, default 0.1).')

    parser.add_argument('--dcn_v2_rank', type=int, default=8,
                        help='Low-rank approximation rank r for DCN-V2 cross weights '
                             '(W = U V^T, U: D×r, V: r×D). '
                             'Effective only with --fusion_type dcn_v2.')

    # ---- tencent-cc2 patch: label_time split + OOF holdout ----
    parser.add_argument('--use_label_time_split', action='store_true',
                        default=False,
                        help='Use label_time-ordered cutoff + 10% user OOF holdout '
                             'instead of the unpatched Row-Group split.')
    parser.add_argument('--oof_user_ratio', type=float, default=0.1,
                        help='Fraction of unique user_ids reserved for OOF holdout. '
                             'Effective only with --use_label_time_split.')
    parser.add_argument('--split_seed', type=int, default=42,
                        help='RNG seed for OOF user sampling. Decoupled from training '
                             '--seed so multi-seed campaigns share the same OOF set '
                             '(CLAUDE.md §4.4 paired Delta rigor). Default 42.')
    parser.add_argument('--work_dir', type=str, default=None,
                        help='Working directory for the v2 split (subset parquets + '
                             '_split_meta.json). Defaults to <ckpt_dir>/../work.')

    args = parser.parse_args()

    # Environment variables take precedence over CLI defaults.
    args.data_dir = os.environ.get('TRAIN_DATA_PATH', args.data_dir)
    args.ckpt_dir = os.environ.get('TRAIN_CKPT_PATH', args.ckpt_dir)
    args.log_dir = os.environ.get('TRAIN_LOG_PATH', args.log_dir)
    args.tf_events_dir = os.environ.get('TRAIN_TF_EVENTS_PATH')

    # Sane defaults: derive missing dirs from --ckpt_dir so the unpatched
    # ``Path(None).mkdir`` crash never reaches main().
    if args.ckpt_dir is None:
        raise SystemExit("TRAIN_CKPT_PATH or --ckpt_dir is required")
    base = os.path.dirname(args.ckpt_dir.rstrip('/'))
    if args.log_dir is None:
        args.log_dir = os.path.join(base or args.ckpt_dir, 'logs')
    if args.tf_events_dir is None:
        args.tf_events_dir = os.path.join(base or args.ckpt_dir, 'tf_events')
    if args.work_dir is None:
        args.work_dir = os.path.join(base or args.ckpt_dir, 'work')

    return args


def main() -> None:
    args = parse_args()

    # Create output directories.
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.tf_events_dir).mkdir(parents=True, exist_ok=True)

    # Initialize logger and RNG.
    set_seed(args.seed)
    create_logger(os.path.join(args.log_dir, 'train.log'))
    logging.info(f"Args: {vars(args)}")

    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(args.tf_events_dir)
    except Exception as _tb_err:  # tensorboard not installed in this env
        logging.warning(f"tensorboard unavailable ({_tb_err}); using no-op writer")
        class _NullWriter:
            def add_scalar(self, *a, **kw): pass
            def close(self): pass
        writer = _NullWriter()

    # ---- Data loading ----
    if args.schema_path:
        schema_path = args.schema_path
    else:
        schema_path = os.path.join(args.data_dir, 'schema.json')

    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"schema file not found at {schema_path}")

    # Parse per-domain sequence-length overrides.
    seq_max_lens = {}
    if args.seq_max_lens:
        for pair in args.seq_max_lens.split(','):
            k, v = pair.split(':')
            seq_max_lens[k.strip()] = int(v.strip())
        logging.info(f"Seq max_lens override: {seq_max_lens}")

    logging.info("Using Parquet data format (IterableDataset)")
    oof_loader = None
    split_meta = None
    if args.use_label_time_split:
        logging.info(f"v2 split (inline filter): label_time cutoff + "
                     f"{args.oof_user_ratio:.0%} user OOF holdout, "
                     f"split_seed={args.split_seed} (training seed={args.seed})")
        train_loader, valid_loader, oof_loader, pcvr_dataset, split_meta = get_pcvr_data_v2(
            data_dir=args.data_dir,
            schema_path=schema_path,
            work_dir=args.work_dir,
            batch_size=args.batch_size,
            valid_ratio=args.valid_ratio,
            oof_user_ratio=args.oof_user_ratio,
            num_workers=args.num_workers,
            buffer_batches=args.buffer_batches,
            seed=args.seed,
            split_seed=args.split_seed,
            seq_max_lens=seq_max_lens,
        )
    else:
        logging.info(
            "Using organizer row-group split (competition/dataset.py:get_pcvr_data). "
            "No OOF holdout, valid = last %.0f%% of row groups in glob order. "
            "For label_time-aware split + 10%% user OOF, pass --use_label_time_split.",
            args.valid_ratio * 100,
        )
        train_loader, valid_loader, pcvr_dataset = get_pcvr_data(
            data_dir=args.data_dir,
            schema_path=schema_path,
            batch_size=args.batch_size,
            valid_ratio=args.valid_ratio,
            train_ratio=args.train_ratio,
            num_workers=args.num_workers,
            buffer_batches=args.buffer_batches,
            seed=args.seed,
            seq_max_lens=seq_max_lens,
        )

    # ---- NS groups ----
    if args.ns_groups_json and os.path.exists(args.ns_groups_json):
        logging.info(f"Loading NS groups from {args.ns_groups_json}")
        with open(args.ns_groups_json, 'r') as f:
            ns_groups_cfg = json.load(f)
        user_fid_to_idx = {fid: i for i, (fid, _, _) in enumerate(pcvr_dataset.user_int_schema.entries)}
        item_fid_to_idx = {fid: i for i, (fid, _, _) in enumerate(pcvr_dataset.item_int_schema.entries)}
        user_ns_groups = [[user_fid_to_idx[f] for f in fids] for fids in ns_groups_cfg['user_ns_groups'].values()]
        item_ns_groups = [[item_fid_to_idx[f] for f in fids] for fids in ns_groups_cfg['item_ns_groups'].values()]
        logging.info(f"User NS groups ({len(user_ns_groups)}): {list(ns_groups_cfg['user_ns_groups'].keys())}")
        logging.info(f"Item NS groups ({len(item_ns_groups)}): {list(ns_groups_cfg['item_ns_groups'].keys())}")
    else:
        logging.info("No NS groups JSON found, using default: each feature as one group")
        user_ns_groups = [[i] for i in range(len(pcvr_dataset.user_int_schema.entries))]
        item_ns_groups = [[i] for i in range(len(pcvr_dataset.item_int_schema.entries))]

    # ---- Build model ----
    user_int_feature_specs = build_feature_specs(
        pcvr_dataset.user_int_schema, pcvr_dataset.user_int_vocab_sizes)
    item_int_feature_specs = build_feature_specs(
        pcvr_dataset.item_int_schema, pcvr_dataset.item_int_vocab_sizes)

    model_args = {
        "user_int_feature_specs": user_int_feature_specs,
        "item_int_feature_specs": item_int_feature_specs,
        "user_dense_dim": pcvr_dataset.user_dense_schema.total_dim,
        "item_dense_dim": pcvr_dataset.item_dense_schema.total_dim,
        "seq_vocab_sizes": pcvr_dataset.seq_domain_vocab_sizes,
        "user_ns_groups": user_ns_groups,
        "item_ns_groups": item_ns_groups,
        "d_model": args.d_model,
        "emb_dim": args.emb_dim,
        "num_queries": args.num_queries,
        "num_hyformer_blocks": args.num_hyformer_blocks,
        "num_heads": args.num_heads,
        "seq_encoder_type": args.seq_encoder_type,
        "hidden_mult": args.hidden_mult,
        "dropout_rate": args.dropout_rate,
        "seq_top_k": args.seq_top_k,
        "seq_causal": args.seq_causal,
        "action_num": args.action_num,
        "num_time_buckets": NUM_TIME_BUCKETS if args.use_time_buckets else 0,
        "rank_mixer_mode": args.rank_mixer_mode,
        "use_rope": args.use_rope,
        "rope_base": args.rope_base,
        "emb_skip_threshold": args.emb_skip_threshold,
        "seq_id_threshold": args.seq_id_threshold,
        "ns_tokenizer_type": args.ns_tokenizer_type,
        "user_ns_tokens": args.user_ns_tokens,
        "item_ns_tokens": args.item_ns_tokens,
        # H004 anchor — backbone routing
        "backbone": args.backbone,
        "num_onetrans_layers": args.num_onetrans_layers,
        "mixed_causal_anchor": args.mixed_causal_anchor,
        "domain_id_embedding": args.domain_id_embedding,
        "log_attn_entropy": args.log_attn_entropy,
        # H008 anchor — fusion mechanism dispatch
        "fusion_type": args.fusion_type,
        "dcn_v2_num_layers": args.dcn_v2_num_layers,
        "dcn_v2_rank": args.dcn_v2_rank,
        # H010 — NS→S full bidirectional cross-attention
        "use_ns_to_s_xattn": args.use_ns_to_s_xattn,
        "ns_xattn_num_heads": args.ns_xattn_num_heads,
        # H019 — TWIN long-seq retrieval
        "use_twin_retrieval": args.use_twin_retrieval,
        "twin_top_k": args.twin_top_k,
        "twin_num_heads": args.twin_num_heads,
        "twin_gate_init": args.twin_gate_init,
        # H052 — User-Item Contrastive aux
        "use_user_item_contrast": args.use_user_item_contrast,
    }

    model = PCVRHyFormer(**model_args).to(args.device)
    logging.info(f"Backbone: {args.backbone}"
                 + (f", num_onetrans_layers={args.num_onetrans_layers}, "
                    f"mixed_causal_anchor={args.mixed_causal_anchor}, "
                    f"domain_id_embedding={args.domain_id_embedding}, "
                    f"log_attn_entropy={args.log_attn_entropy}"
                    if args.backbone == 'onetrans' else ""))

    # Log model sizing info.
    num_sequences = len(pcvr_dataset.seq_domains)
    num_ns = model.num_ns
    T = args.num_queries * num_sequences + num_ns
    logging.info(f"PCVRHyFormer model created: num_ns={num_ns}, T={T}, d_model={args.d_model}, rank_mixer_mode={args.rank_mixer_mode}")
    logging.info(f"User NS groups: {user_ns_groups}")
    logging.info(f"Item NS groups: {item_ns_groups}")
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Total parameters: {total_params:,}")

    # ---- Training ----
    early_stopping = EarlyStopping(
        checkpoint_path=os.path.join(args.ckpt_dir, "placeholder", "model.pt"),
        patience=args.patience,
        label='model',
    )

    ckpt_params = {
        "layer": args.num_hyformer_blocks,
        "head": args.num_heads,
        "hidden": args.d_model,
    }

    trainer = PCVRHyFormerRankingTrainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        lr=args.lr,
        num_epochs=args.num_epochs,
        device=args.device,
        save_dir=args.ckpt_dir,
        early_stopping=early_stopping,
        loss_type=args.loss_type,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        contrast_lambda=args.contrast_lambda,    # H052
        contrast_temperature=args.contrast_temperature,    # H052
        sparse_lr=args.sparse_lr,
        sparse_weight_decay=args.sparse_weight_decay,
        reinit_sparse_after_epoch=args.reinit_sparse_after_epoch,
        reinit_cardinality_threshold=args.reinit_cardinality_threshold,
        ckpt_params=ckpt_params,
        writer=writer,
        schema_path=schema_path,
        ns_groups_path=args.ns_groups_json if args.ns_groups_json and os.path.exists(args.ns_groups_json) else None,
        eval_every_n_steps=args.eval_every_n_steps,
        train_config=vars(args),
    )

    trainer.train()
    writer.close()

    # ---- tencent-cc2 patch: optional OOF evaluation + metrics.json ----
    best_val_auc = float(early_stopping.best_score) if early_stopping.best_score is not None else float('nan')
    best_extra = early_stopping.best_extra_metrics or {}
    metrics_out = {
        'seed': args.seed,
        'host': os.uname().nodename if hasattr(os, 'uname') else '',
        'best_val_AUC': best_val_auc,
        'best_val_logloss': best_extra.get('best_val_logloss', None),
        'config': vars(args),
    }
    if oof_loader is not None and trainer.early_stopping.best_model is not None:
        logging.info("Loading best weights for OOF evaluation")
        model.load_state_dict(trainer.early_stopping.best_model)
        original_valid = trainer.valid_loader
        trainer.valid_loader = oof_loader
        try:
            oof_auc, oof_ll = trainer.evaluate(epoch=-1)
        finally:
            trainer.valid_loader = original_valid
        metrics_out['best_oof_AUC'] = float(oof_auc)
        metrics_out['best_oof_logloss'] = float(oof_ll)
        logging.info(f"OOF AUC: {oof_auc:.4f}, OOF LogLoss: {oof_ll:.4f}")
    if split_meta is not None:
        metrics_out['split_meta'] = split_meta

    # ---- H004 anchor: §10.9 attention entropy diagnostic ----
    # OneTrans backbone caches per-layer mean attention entropy on each forward.
    # Run a single diagnostic batch through the best model to capture the values.
    if (args.backbone == 'onetrans' or args.use_ns_to_s_xattn) and args.log_attn_entropy:
        try:
            if trainer.early_stopping.best_model is not None:
                model.load_state_dict(trainer.early_stopping.best_model)
            model.eval()
            # Pull one valid batch (train_loader is also fine — eval mode).
            diag_loader = oof_loader if oof_loader is not None else trainer.valid_loader
            diag_batch = next(iter(diag_loader))
            with torch.no_grad():
                trainer._evaluate_step(diag_batch)
            entropies = model.collect_attn_entropies() or []
            metrics_out['attn_entropy_per_layer'] = entropies
            # Threshold: 0.95 * log(N_tokens). N_tokens at runtime = sum(seq_max_lens) + num_ns + 1 (CLS).
            import math
            seq_total = sum(int(v) for v in (
                {pair.split(':')[0].strip(): int(pair.split(':')[1])
                 for pair in args.seq_max_lens.split(',')} if args.seq_max_lens else {}
            ).values())
            if args.use_ns_to_s_xattn and args.backbone == 'hyformer':
                # H010 NS→S xattn: K=V is concatenated S tokens only (per-domain seq encoder outputs).
                n_tokens = max(seq_total, 2)
            else:
                # OneTrans backbone: token sequence = S + NS + CLS.
                n_tokens = max(seq_total + int(model.num_ns) + 1, 2)
            threshold = 0.95 * math.log(n_tokens)
            metrics_out['attn_entropy_threshold'] = threshold
            metrics_out['attn_entropy_violation'] = any(
                (e is not None and e >= threshold) for e in entropies
            )
            logging.info(f"attn_entropy_per_layer={entropies}, threshold={threshold:.4f}, "
                         f"violation={metrics_out['attn_entropy_violation']}")
        except Exception as e:  # diagnostic failure must not break metrics dump
            logging.warning(f"attn entropy diagnostic failed: {e}")
            metrics_out['attn_entropy_per_layer'] = None
            metrics_out['attn_entropy_violation'] = None

    # Optional repro metadata.
    try:
        import subprocess
        sha = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        metrics_out['git_sha'] = sha
    except Exception:
        metrics_out['git_sha'] = 'unknown'
    try:
        import hashlib
        cfg_blob = json.dumps(vars(args), sort_keys=True, default=str).encode()
        metrics_out['config_sha256'] = hashlib.sha256(cfg_blob).hexdigest()[:16]
    except Exception:
        pass

    metrics_path = os.path.join(args.ckpt_dir, 'metrics.json')
    with open(metrics_path, 'w') as fh:
        json.dump(metrics_out, fh, indent=2, default=str)
    logging.info(f"Metrics dumped to {metrics_path}")

    logging.info("Training complete!")


if __name__ == "__main__":
    main()
