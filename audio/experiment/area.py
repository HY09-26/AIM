"""
area.py - AOC / ABC / AUC metrics from MoRF and LeRF result files.

Metrics:
  AOC (Area Over Curve)      = mean(1 - MoRF accuracy)
  ABC (Area Between Curves)  = mean(max(LeRF - MoRF, 0))
  AUC (Area Under Curve)     = mean(LeRF accuracy)

Expected directory structure (produced by test_waveform.py / test_waveform_interval.py):
  <morf_lerf_root>/<dataset>/<model>/<mask_type>/[fold_N/]{method}_morf.pkl
  <morf_lerf_root>/<dataset>/<model>/<mask_type>/[fold_N/]{method}_lerf.pkl

For ESC-50, results are stored inside fold_1/ ... fold_5/ subdirectories.
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# Constants
# ============================================================

DATASET_CHANCE: dict[str, float] = {
    "audiomnist": 1 / 10,   # 10 classes
    "esc50":      1 / 50,   # 50 classes
    "msos":       1 / 5,    # 5 classes
}

DATASETS = ["audiomnist", "esc50", "msos"]
MODELS   = ["alexnet", "cnn14", "audionet", "res1dnet31"]
MASKS    = ["zero", "pgd", "road"]

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

METHOD_DISPLAY = {
    "gradient":       "GD",
    "gradinput":      "GI",
    "smoothgrad":     "SG",
    "smoothgrad_sq":  "SS",
    "vargrad":        "VG",
    "integrad":       "IG",
    "gradient_abs":   "GDA",
    "gradinput_abs":  "GIA",
    "smoothgrad_abs": "SGA",
    "integrad_abs":   "IGA",
    "random":         "RD",
}


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
        1-D float array of per-step accuracies.
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
        raise ValueError(f"Scalar found in {pkl_path}; expected a curve array.")
    if arr.ndim >= 2:
        arr = arr.mean(axis=0)   # average over runs if stacked

    if drop_first and len(arr) > 1:
        arr = arr[1:]
    return arr


# ============================================================
# Metrics
# ============================================================

def compute_metrics(
    curve_morf: np.ndarray,
    curve_lerf: np.ndarray,
    chance: float,
) -> dict[str, float]:
    """
    Compute AOC, ABC, and AUC from paired MoRF and LeRF accuracy curves.

    Values are clipped to [chance, 1.0] before metric computation, consistent
    with the original AIM paper.

    Args:
        curve_morf: MoRF accuracy at each masking step, shape (K,).
        curve_lerf: LeRF accuracy at each masking step, shape (K,).
        chance:     Random-chance accuracy (1 / num_classes).

    Returns:
        Dictionary with keys "aoc", "abc", "auc".
    """
    L = min(len(curve_morf), len(curve_lerf))
    m = np.clip(curve_morf[:L], chance, 1.0)
    l = np.clip(curve_lerf[:L], chance, 1.0)

    return {
        "aoc": float(np.mean(1.0 - m)),
        "abc": float(np.mean(np.clip(l - m, 0.0, None))),
        "auc": float(np.mean(l)),
    }


# ============================================================
# Folder scanning
# ============================================================

def get_result_folders(root: str, dataset: str, model: str, mask: str) -> list[str]:
    """Return result directories, expanding ESC-50 fold subdirectories."""
    if dataset == "esc50":
        pattern = os.path.join(root, dataset, model, mask, "fold_*")
        return sorted(p for p in glob.glob(pattern) if os.path.isdir(p))
    folder = os.path.join(root, dataset, model, mask)
    return [folder] if os.path.isdir(folder) else []


def compute_folder_metrics(
    folder: str,
    chance: float,
    drop_first: bool = True,
) -> pd.DataFrame:
    """
    Compute AOC/ABC/AUC for every (method, morf/lerf) pair found in a folder.

    Args:
        folder:     Directory containing *_morf.pkl and *_lerf.pkl files.
        chance:     Random-chance accuracy for clipping.
        drop_first: If True, drop step-0 (0% masked) from curves.

    Returns:
        DataFrame with columns [method, display, aoc, abc, auc].
    """
    rows = []
    for method in METHOD_ORDER:
        morf_path = os.path.join(folder, f"{method}_morf.pkl")
        lerf_path = os.path.join(folder, f"{method}_lerf.pkl")
        if not (os.path.isfile(morf_path) and os.path.isfile(lerf_path)):
            continue
        try:
            curve_m = load_curve(morf_path, drop_first=drop_first)
            curve_l = load_curve(lerf_path, drop_first=drop_first)
        except Exception as exc:
            print(f"[WARN] Could not load {method} in {folder}: {exc}")
            continue

        metrics = compute_metrics(curve_m, curve_l, chance)
        rows.append({
            "method":  method,
            "display": METHOD_DISPLAY.get(method, method.upper()),
            **metrics,
        })

    return pd.DataFrame(rows)


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute AOC/ABC/AUC metrics from MoRF/LeRF result files."
    )
    parser.add_argument("--root",       default="morf_lerf",
                        help="Root directory of morf/lerf result folders.")
    parser.add_argument("--dataset",    choices=DATASETS + ["all"], default="all")
    parser.add_argument("--model",      default="all",
                        help="Model name, or 'all'.")
    parser.add_argument("--mask",       choices=MASKS + ["all"], default="all")
    parser.add_argument("--drop_first", action="store_true",
                        help="Drop the 0%% masked step from curves.")
    parser.add_argument("--out_csv",    default="audio_area_metrics.csv",
                        help="Output CSV path.")
    args = parser.parse_args()

    datasets = DATASETS if args.dataset == "all" else [args.dataset]
    models   = MODELS   if args.model   == "all" else [args.model]
    masks    = MASKS    if args.mask    == "all" else [args.mask]

    all_rows: list[dict] = []

    for dataset in datasets:
        chance = DATASET_CHANCE[dataset]
        for model in models:
            for mask in masks:
                folders = get_result_folders(args.root, dataset, model, mask)
                if not folders:
                    continue

                fold_dfs: list[pd.DataFrame] = []
                for folder in folders:
                    fold = os.path.basename(folder) if dataset == "esc50" else "no_fold"
                    df = compute_folder_metrics(folder, chance, drop_first=args.drop_first)
                    if df.empty:
                        continue
                    df["fold"] = fold
                    fold_dfs.append(df)

                if not fold_dfs:
                    continue

                # Average metrics across folds
                combined = pd.concat(fold_dfs, ignore_index=True)
                agg = (
                    combined.groupby("method")[["aoc", "abc", "auc"]]
                    .agg(["mean", "std"])
                )
                agg.columns = ["_".join(c) for c in agg.columns]
                agg = agg.reset_index()

                print(f"\n{'='*60}")
                print(f"Dataset={dataset.upper()}  Model={model}  Mask={mask}")
                print(f"{'='*60}")
                print(f"{'Method':>6}  {'AOC':>12}  {'ABC':>12}  {'AUC':>12}")
                for _, row in agg.iterrows():
                    disp = METHOD_DISPLAY.get(row["method"], row["method"].upper())
                    print(
                        f"{disp:>6}  "
                        f"{row['aoc_mean']:.3f}±{row['aoc_std']:.3f}  "
                        f"{row['abc_mean']:.3f}±{row['abc_std']:.3f}  "
                        f"{row['auc_mean']:.3f}±{row['auc_std']:.3f}"
                    )

                for _, row in agg.iterrows():
                    all_rows.append({
                        "dataset": dataset, "model": model, "mask": mask,
                        "method":  row["method"],
                        "display": METHOD_DISPLAY.get(row["method"], row["method"].upper()),
                        "aoc_mean": row["aoc_mean"], "aoc_std": row["aoc_std"],
                        "abc_mean": row["abc_mean"], "abc_std": row["abc_std"],
                        "auc_mean": row["auc_mean"], "auc_std": row["auc_std"],
                    })

    if all_rows:
        pd.DataFrame(all_rows).to_csv(args.out_csv, index=False)
        print(f"\nSaved metrics to: {args.out_csv}")
    else:
        print("\n[WARN] No results found. Check --root path and directory structure.")


if __name__ == "__main__":
    main()
