"""
spearman.py - Spearman rank correlation between MoRF and LeRF curves (image).

Computes per-step and overall Spearman ρ between MoRF and LeRF accuracy
curves across attribution methods.  A high ρ indicates that MoRF and LeRF
rank attribution methods consistently, suggesting discriminating masking.

Supported datasets : brain_mri | imagenet | oxford_pet
Supported models   : resnet_50 | efficientnet_b0 | repvgg_b0
Supported masking  : zero | pgd | road
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DATASETS  = ["brain_mri", "imagenet", "oxford_pet"]
MODELS    = ["resnet_50", "efficientnet_b0", "repvgg_b0"]
MASKS     = ["zero", "pgd", "road"]

# Attribution methods in a canonical display order
METHOD_ORDER = [
    "gradients",
    "smoothgrad",
    "gradcam",
    "gradcampp",
    "gradients_abs",
    "smoothgrad_abs",
    "random",
]

METHOD_DISPLAY = {
    "gradients":       "GD",
    "smoothgrad":      "SG",
    "gradcam":         "GC",
    "gradcampp":       "GCPP",
    "gradients_abs":   "GDA",
    "smoothgrad_abs":  "SGA",
    "random":          "RD",
}


# ============================================================
# Spearman helpers
# ============================================================

def spearman_rank(
    r1: np.ndarray,
    r2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Compute Spearman ρ between two vectors using two complementary formulas.

    MoRF accuracy is ranked ascending (lower is better).
    LeRF accuracy is ranked descending (higher is better), so r2 is negated.

    Args:
        r1: MoRF accuracy values for each attribution method at a fixed step.
        r2: LeRF accuracy values for each attribution method at the same step.

    Returns:
        (rankq1, rankq2, rho_diff, rho_corr).
    """
    rank1 = rankdata(r1)            * 1.0
    rank2 = rankdata(-np.asarray(r2)) * 1.0

    rankq1, rankq2 = rank1.copy(), rank2.copy()

    # Half-integer tie adjustment
    uq1, cnt1 = np.unique(rank1, return_counts=True)
    uq2, cnt2 = np.unique(rank2, return_counts=True)
    for u, c in zip(uq1, cnt1):
        if c > 1 and u % 1 > 0:
            rankq1[rank1 == u] += 0.5
    for u, c in zip(uq2, cnt2):
        if c > 1 and u % 1 > 0:
            rankq2[rank2 == u] += 0.5

    rho_corr = float(np.corrcoef(rankq1, rankq2)[0, 1])
    n        = len(rankq1)
    rho_diff = float(
        1 - 6 * np.sum((rankq1 - rankq2) ** 2) / (n * (n * n - 1))
    )
    return rankq1, rankq2, rho_diff, rho_corr


# ============================================================
# I/O helpers
# ============================================================

def load_curve(pkl_path: str, drop_first: bool = True) -> np.ndarray:
    """
    Load a MoRF or LeRF accuracy curve from a pickle file.

    Args:
        pkl_path:   Path to the .pkl file.
        drop_first: If True, drop the first element (step 0, 0% masked).

    Returns:
        1-D float array of accuracies.
    """
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)

    # Support dict-wrapped curves (legacy format)
    if isinstance(obj, dict):
        for key in ("acc", "accuracy", "acc_curve", "curve", "accs"):
            if key in obj:
                obj = obj[key]
                break
        else:
            for v in obj.values():
                if isinstance(v, (list, tuple, np.ndarray)):
                    obj = v
                    break

    arr = np.asarray(obj, dtype=float).squeeze()
    if arr.ndim == 0:
        raise ValueError(f"Scalar found in {pkl_path}; expected a curve.")
    if arr.ndim >= 2:
        arr = arr.mean(axis=0)
    if drop_first and len(arr) > 1:
        arr = arr[1:]
    return arr


def compute_folder_spearman(
    folder: str,
    drop_first: bool = True,
) -> tuple[list[str], pd.DataFrame]:
    """
    Compute per-step Spearman ρ over all attribution methods in a result folder.

    Args:
        folder:     Directory containing *_morf.pkl and *_lerf.pkl files.
        drop_first: If True, drop the 0% masked step.

    Returns:
        (valid_methods, df_ratio) where df_ratio has columns
        [ratio_index, rho_diff, rho_corr].
    """
    valid_methods, morf_curves, lerf_curves = [], [], []

    for method in METHOD_ORDER:
        morf_path = os.path.join(folder, f"{method}_morf.pkl")
        lerf_path = os.path.join(folder, f"{method}_lerf.pkl")
        if not (os.path.isfile(morf_path) and os.path.isfile(lerf_path)):
            continue
        m = load_curve(morf_path, drop_first=drop_first)
        l = load_curve(lerf_path, drop_first=drop_first)
        L = min(len(m), len(l))
        valid_methods.append(method)
        morf_curves.append(m[:L])
        lerf_curves.append(l[:L])

    if len(valid_methods) < 2:
        raise ValueError(
            f"Need at least 2 methods; found {valid_methods} in {folder}"
        )

    min_len = min(len(x) for x in morf_curves)
    alignM  = np.stack([x[:min_len] for x in morf_curves], axis=0)  # (M, T)
    alignL  = np.stack([x[:min_len] for x in lerf_curves], axis=0)

    rows = []
    for r in range(min_len):
        _, _, rho_diff, rho_corr = spearman_rank(alignM[:, r], alignL[:, r])
        rows.append({
            "ratio_index": r + 1,
            "rho_diff":    rho_diff,
            "rho_corr":    rho_corr,
        })
    return valid_methods, pd.DataFrame(rows)


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Spearman ρ between MoRF and LeRF curves (image)."
    )
    parser.add_argument("--root",       default="morf_lerf_image",
                        help="Root directory of morf_lerf result folders.")
    parser.add_argument("--dataset",    choices=DATASETS + ["all"], default="all")
    parser.add_argument("--model",      default="all")
    parser.add_argument("--mask",       choices=MASKS + ["all"], default="all")
    parser.add_argument("--drop_first", action="store_true")
    parser.add_argument("--out_csv",    default="image_spearman_summary.csv")
    args = parser.parse_args()

    # Resolve root relative to project
    root = args.root if os.path.isabs(args.root) else os.path.join(PROJECT_ROOT, args.root)

    datasets = DATASETS if args.dataset == "all" else [args.dataset]
    models   = MODELS   if args.model   == "all" else [args.model]
    masks    = MASKS    if args.mask    == "all" else [args.mask]

    summary_rows: list[dict] = []

    for dataset in datasets:
        for model in models:
            for mask in masks:
                folder = os.path.join(root, model, dataset, mask)
                if not os.path.isdir(folder):
                    print(f"[SKIP] {folder}")
                    continue

                try:
                    methods, df = compute_folder_spearman(
                        folder, drop_first=args.drop_first
                    )
                except Exception as exc:
                    print(f"[SKIP] {folder}: {exc}")
                    continue

                flat_diff = df["rho_diff"].values
                flat_corr = df["rho_corr"].values
                summary_rows.append({
                    "dataset":       dataset,
                    "model":         model,
                    "mask":          mask,
                    "n_methods":     len(methods),
                    "methods":       ",".join(methods),
                    "n_steps":       len(df),
                    "rho_diff_mean": float(flat_diff.mean()),
                    "rho_diff_std":  float(flat_diff.std(ddof=0)),
                    "rho_corr_mean": float(flat_corr.mean()),
                    "rho_corr_std":  float(flat_corr.std(ddof=0)),
                    "rho_diff_pretty": (
                        f"{flat_diff.mean():.3f}±{flat_diff.std(ddof=0):.3f}"
                    ),
                })
                print(
                    f"[OK] {dataset}/{model}/{mask}  "
                    f"ρ_diff={flat_diff.mean():.3f}±{flat_diff.std(ddof=0):.3f}  "
                    f"ρ_corr={flat_corr.mean():.3f}±{flat_corr.std(ddof=0):.3f}"
                )

    if summary_rows:
        df_out = pd.DataFrame(summary_rows)
        df_out.to_csv(args.out_csv, index=False)
        print(f"\nSaved summary to: {args.out_csv}")

        print("\n=== compact table ===")
        for _, row in df_out.iterrows():
            print(
                f"{row['dataset']:12s} {row['model']:17s} {row['mask']:5s}  "
                f"ρ={row['rho_diff_pretty']}"
            )
    else:
        print("[WARN] No results found. Check --root path and folder structure.")


if __name__ == "__main__":
    main()
