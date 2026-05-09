#!/usr/bin/env python3
"""H024 — Ensemble of H010 + H015 + H018 predictions (LOCAL script, no cloud retrain).

Inputs (3 predictions.json files, each from a separate H run):
    --h010-preds  /path/to/H010_predictions.json
    --h015-preds  /path/to/H015_predictions.json
    --h018-preds  /path/to/H018_predictions.json

Output (1 ensembled predictions.json):
    --output      /path/to/H024_ensemble_predictions.json

Format: each predictions.json is `{user_id_str: prob_float}` mapping (Taiji
standard format). Ensemble = arithmetic mean of probs across 3 H. Optional
weighted ensemble via --weights (default uniform 1/3 each).

Usage (local, no cloud submit needed for the ensemble step itself —
just upload the OUTPUT predictions.json to Taiji):

    python ensemble.py \
        --h010-preds /path/H010_predictions.json \
        --h015-preds /path/H015_predictions.json \
        --h018-preds /path/H018_predictions.json \
        --output     /path/H024_ensemble_predictions.json \
        --weights    1.0 1.0 1.0

Then upload H024_ensemble_predictions.json to platform for scoring.

§17.2 EXEMPT — measurement H (no model retrain, prediction-level
combination only).
§17.4 — `measurement` re-entry justified (H022/H023 sibling, ensemble
form variant).
§17.6 — cost $0 (local CPU < 1 minute) + 1 platform submission.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List


def load_predictions(path: Path) -> Dict[str, float]:
    """Load Taiji predictions.json — {user_id_str: prob_float}."""
    with open(path) as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected dict, got {type(data).__name__}")
    n = len(data)
    if n == 0:
        raise ValueError(f"{path}: empty predictions")
    # Sanity: probs in [0, 1].
    sample_key = next(iter(data))
    sample_val = float(data[sample_key])
    if not (0.0 <= sample_val <= 1.0):
        print(
            f"WARN {path}: sample prob {sample_val} outside [0,1] (logit?). "
            "Continuing — ensemble caller responsible for format.",
            file=sys.stderr,
        )
    return {str(k): float(v) for k, v in data.items()}


def ensemble(
    pred_dicts: List[Dict[str, float]],
    weights: List[float],
) -> Dict[str, float]:
    """Weighted-mean ensemble across N prediction dicts.

    All dicts must share the same key set (user_id strings). Mismatched keys
    raise — silently skipping users would corrupt platform scoring.
    """
    if len(pred_dicts) != len(weights):
        raise ValueError(
            f"weights count ({len(weights)}) must match "
            f"prediction dicts count ({len(pred_dicts)})"
        )
    keys_0 = set(pred_dicts[0].keys())
    for i, d in enumerate(pred_dicts[1:], start=1):
        keys_i = set(d.keys())
        if keys_i != keys_0:
            missing = keys_0 - keys_i
            extra = keys_i - keys_0
            raise ValueError(
                f"prediction dict #{i} key mismatch with #0: "
                f"missing {len(missing)} keys (e.g., {sorted(missing)[:3]}), "
                f"extra {len(extra)} keys (e.g., {sorted(extra)[:3]})"
            )
    weight_sum = float(sum(weights))
    if weight_sum <= 0.0:
        raise ValueError(f"weight_sum {weight_sum} must be > 0")
    norm_weights = [w / weight_sum for w in weights]

    out: Dict[str, float] = {}
    for k in keys_0:
        prob = 0.0
        for d, w in zip(pred_dicts, norm_weights):
            prob += w * d[k]
        # Clamp to [0, 1] to defend against numerical drift.
        prob = max(0.0, min(1.0, prob))
        out[k] = prob
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="H024 — ensemble of H010+H015+H018 predictions"
    )
    parser.add_argument('--h010-preds', type=Path, required=True,
                        help='H010 predictions.json (corrected anchor)')
    parser.add_argument('--h015-preds', type=Path, required=True,
                        help='H015 predictions.json (per-batch recency)')
    parser.add_argument('--h018-preds', type=Path, required=True,
                        help='H018 predictions.json (per-user recency)')
    parser.add_argument('--output', type=Path, required=True,
                        help='Output ensembled predictions.json')
    parser.add_argument(
        '--weights',
        type=float,
        nargs=3,
        default=[1.0, 1.0, 1.0],
        metavar=('w_H010', 'w_H015', 'w_H018'),
        help='Per-H weights (will be normalized to sum=1). Default uniform.',
    )
    args = parser.parse_args()

    print(f"[H024] loading H010 from {args.h010_preds}")
    p_h010 = load_predictions(args.h010_preds)
    print(f"[H024] loading H015 from {args.h015_preds}")
    p_h015 = load_predictions(args.h015_preds)
    print(f"[H024] loading H018 from {args.h018_preds}")
    p_h018 = load_predictions(args.h018_preds)

    n_h010, n_h015, n_h018 = len(p_h010), len(p_h015), len(p_h018)
    print(f"[H024] N_h010={n_h010}, N_h015={n_h015}, N_h018={n_h018}")
    if not (n_h010 == n_h015 == n_h018):
        print(
            f"ERROR row count mismatch: H010={n_h010} H015={n_h015} H018={n_h018}",
            file=sys.stderr,
        )
        return 1

    print(f"[H024] ensemble weights (raw): {args.weights}")
    weight_sum = sum(args.weights)
    print(f"[H024] ensemble weights (normalized): "
          f"{[w / weight_sum for w in args.weights]}")

    ensembled = ensemble([p_h010, p_h015, p_h018], args.weights)
    print(f"[H024] ensembled {len(ensembled)} predictions")

    # Sanity: prob distribution.
    probs = list(ensembled.values())
    p_mean = sum(probs) / len(probs)
    p_min = min(probs)
    p_max = max(probs)
    p_var = sum((p - p_mean) ** 2 for p in probs) / len(probs)
    p_std = p_var ** 0.5
    print(
        f"[H024] prob stats: mean={p_mean:.4f} std={p_std:.4f} "
        f"min={p_min:.4f} max={p_max:.4f}"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as fh:
        json.dump(ensembled, fh)
    print(f"[H024] wrote ensembled predictions to {args.output}")
    print("[H024] next step: upload this file to platform for scoring.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
