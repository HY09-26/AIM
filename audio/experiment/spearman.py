"""
spearman.py - Spearman rank correlation between MoRF and LeRF curves.

Computes per-step and overall Spearman ρ between MoRF and LeRF accuracy
curves across attribution methods.  A high ρ indicates that MoRF and LeRF
rank methods in the same order, suggesting that the masking protocol is
discriminating attribution quality.

Supported datasets : audiomnist, esc50, msos
Supported models   : alexnet, cnn14
Supported masking  : zero | pgd | road
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle

import numpy as np
import pandas as pd
from scipy.stats import rankdata

DATASETS = ["audiomnist", "esc50", "msos"]
MODELS   = ["alexnet", "cnn14"]
MASKS    = ["pgd", "zero", "road"]

METHOD_ORDER = [
    "gradient",
    "gradinput",
    "smoothgrad",
    "smoothgrad_sq",
    "vargrad",
    "integrad",
    "gradient_abs",
    "gradinput_abs",
    "smoothgrad_abs",
    "integrad_abs",
    "random",
]


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
        (rankq1, rankq2, rho_diff, rho_corr):
          rankq1/rankq2  — tie-adjusted rank vectors.
          rho_diff       — Spearman ρ via the d² formula.
          rho_corr       — Spearman ρ via the Pearson correlation of ranks.
    """
    rank1 = rankdata(r1)           * 1.0
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

    # Pearson correlation on ranks
    rho_corr = float(np.corrcoef(rankq1, rankq2)[0, 1])

    # d² formula
    n = len(rankq1)
    rho_diff = float(1 - 6 * np.sum((rankq1 - rankq2) ** 2) / (n * (n * n - 1)))

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
        arr = pickle.load(f)
    arr = np.asarray(arr, dtype=float).squeeze()
    if arr.ndim != 1:
        raise ValueError(f"{pkl_path} is not 1-D, got shape {arr.shape}")
    if drop_first and len(arr) > 1:
        arr = arr[1:]
    return arr


def compute_one_folder(
    folder: str,
    drop_first: bool = True,
) -> tuple[list[str], pd.DataFrame]:
    """
    Compute per-step Spearman ρ over all attribution methods in a result folder.

    Args:
        folder:     Directory containing *_morf.pkl and *_lerf.pkl files.
        drop_first: If True, drop the 0% masked step.

    Returns:
        (valid_methods, df_ratio):
          valid_methods — list of method names that had both morf and lerf files.
          df_ratio      — DataFrame with columns
                          [ratio_index, rho_diff, rho_corr] per masking step.
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
        raise ValueError(f"Not enough methods in {folder}. Found: {valid_methods}")

    min_len = min(len(x) for x in morf_curves)
    alignM = np.stack([x[:min_len] for x in morf_curves], axis=0)  # (M, T)
    alignL = np.stack([x[:min_len] for x in lerf_curves], axis=0)

    ratio_rows = []
    for r in range(min_len):
        _, _, rho_diff, rho_corr = spearman_rank(alignM[:, r], alignL[:, r])
        ratio_rows.append({
            "ratio_index": r + 1,
            "rho_diff": rho_diff,
            "rho_corr": rho_corr,
        })

    return valid_methods, pd.DataFrame(ratio_rows)


def get_target_folders(root: str, dataset: str, model: str, mask: str) -> list[str]:
    """Return result folders for the given configuration, expanding ESC-50 folds."""
    if dataset == "esc50":
        pattern = os.path.join(root, dataset, model, mask, "fold_*")
        return sorted(p for p in glob.glob(pattern) if os.path.isdir(p))
    folder = os.path.join(root, dataset, model, mask)
    return [folder] if os.path.isdir(folder) else []


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Spearman rank correlation between MoRF and LeRF curves."
    )
    parser.add_argument("--root",        default="morf_lerf",
                        help="Root directory of morf/lerf result folders.")
    parser.add_argument("--drop_first",  action="store_true",
                        help="Drop the 0%% masked step from curves.")
    parser.add_argument("--out_summary", default="audio_spearman_summary.csv")
    parser.add_argument("--out_ratio",   default="audio_spearman_by_ratio.csv")
    parser.add_argument("--out_fold",    default="audio_spearman_by_fold.csv")
    args = parser.parse_args()

    summary_rows: list[dict] = []
    ratio_rows_all: list[dict] = []
    fold_rows_all: list[dict] = []

    for dataset in DATASETS:
        for model in MODELS:
            for mask in MASKS:
                folders = get_target_folders(args.root, dataset, model, mask)
                if not folders:
                    print(f"[SKIP] no folder: {dataset}/{model}/{mask}")
                    continue

                fold_ratio_tables: list[pd.DataFrame] = []
                fold_names: list[str] = []

                for folder in folders:
                    fold_name = os.path.basename(folder) if dataset == "esc50" else "no_fold"
                    try:
                        methods, df_ratio = compute_one_folder(
                            folder, drop_first=args.drop_first
                        )
                    except Exception as exc:
                        print(f"[SKIP] {folder}: {exc}")
                        continue

                    fold_names.append(fold_name)
                    fold_ratio_tables.append(df_ratio)
                    fold_rows_all.append({
                        "dataset": dataset, "model": model, "mask": mask,
                        "fold": fold_name, "folder": folder,
                        "n_methods": len(methods), "methods": ",".join(methods),
                        "n_ratios": len(df_ratio),
                        "rho_diff_mean": float(df_ratio["rho_diff"].mean()),
                        "rho_diff_std":  float(df_ratio["rho_diff"].std(ddof=0)),
                        "rho_corr_mean": float(df_ratio["rho_corr"].mean()),
                        "rho_corr_std":  float(df_ratio["rho_corr"].std(ddof=0)),
                        "rho_diff_pretty": (
                            f"{df_ratio['rho_diff'].mean():.3f}"
                            f"±{df_ratio['rho_diff'].std(ddof=0):.3f}"
                        ),
                    })

                if not fold_ratio_tables:
                    continue

                min_len = min(len(df) for df in fold_ratio_tables)
                rho_diff_mat = np.stack(
                    [df["rho_diff"].values[:min_len] for df in fold_ratio_tables], axis=0
                )
                rho_corr_mat = np.stack(
                    [df["rho_corr"].values[:min_len] for df in fold_ratio_tables], axis=0
                )

                for r in range(min_len):
                    ratio_rows_all.append({
                        "dataset": dataset, "model": model, "mask": mask,
                        "ratio_index": r + 1,
                        "n_folds":       rho_diff_mat.shape[0],
                        "rho_diff_mean": float(rho_diff_mat[:, r].mean()),
                        "rho_diff_std":  float(rho_diff_mat[:, r].std(ddof=0)),
                        "rho_corr_mean": float(rho_corr_mat[:, r].mean()),
                        "rho_corr_std":  float(rho_corr_mat[:, r].std(ddof=0)),
                    })

                flat_diff = rho_diff_mat.ravel()
                flat_corr = rho_corr_mat.ravel()
                summary_rows.append({
                    "dataset": dataset, "model": model, "mask": mask,
                    "n_folds": rho_diff_mat.shape[0], "n_ratios": min_len,
                    "rho_diff_mean": float(flat_diff.mean()),
                    "rho_diff_std":  float(flat_diff.std(ddof=0)),
                    "rho_corr_mean": float(flat_corr.mean()),
                    "rho_corr_std":  float(flat_corr.std(ddof=0)),
                    "rho_diff_pretty": (
                        f"{flat_diff.mean():.3f}±{flat_diff.std(ddof=0):.3f}"
                    ),
                    "rho_corr_pretty": (
                        f"{flat_corr.mean():.3f}±{flat_corr.std(ddof=0):.3f}"
                    ),
                })

                print(
                    f"[OK] {dataset}/{model}/{mask}  "
                    f"folds={fold_names}  "
                    f"ρ_diff={flat_diff.mean():.3f}±{flat_diff.std(ddof=0):.3f}"
                )

    pd.DataFrame(summary_rows).to_csv(args.out_summary, index=False)
    pd.DataFrame(ratio_rows_all).to_csv(args.out_ratio, index=False)
    pd.DataFrame(fold_rows_all).to_csv(args.out_fold, index=False)

    print(f"\nSaved: {args.out_summary}, {args.out_ratio}, {args.out_fold}")


if __name__ == "__main__":
    main()
