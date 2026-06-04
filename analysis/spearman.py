"""Unified MoRF–LeRF Spearman ρ — image, audio, and EEG domains.

Usage examples
--------------
# from project root:
python analysis/spearman.py --domain image
python analysis/spearman.py --domain audio --dataset esc50
python analysis/spearman.py --domain eeg   --eeg_domain ts

# or via per-domain thin wrappers (original paths still work):
python image/experiment/spearman.py
python audio/experiment/spearman.py
python eeg/result_process/result_spears.py

Interpretation
--------------
For each masking step k, rank attribution methods by their MoRF accuracy
(ascending — lower is better) and LeRF accuracy (descending — higher is
better), then compute Spearman ρ.  A high ρ means MoRF and LeRF agree on
which methods produce more informative attributions, indicating that the
masking protocol discriminates attribution quality consistently.
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ANALYSIS_DIR)


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers (shared with area.py)
# ─────────────────────────────────────────────────────────────────────────────

class _CompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core")
        return super().find_class(module, name)


def _load_pickle(path: str):
    with open(path, "rb") as f:
        return _CompatUnpickler(f).load()


def _load_curve_pkl(path: str, drop_first: bool = False) -> np.ndarray:
    obj = _load_pickle(path)
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
        raise ValueError(f"Scalar in {path}")
    if arr.ndim >= 2:
        arr = arr.mean(axis=0)
    if drop_first and len(arr) > 1:
        arr = arr[1:]
    return arr.astype(float)


# ─────────────────────────────────────────────────────────────────────────────
# Spearman rank (identical across image / audio / EEG)
# ─────────────────────────────────────────────────────────────────────────────

def spearman_rank(
    r1: np.ndarray,
    r2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Spearman ρ between MoRF and LeRF accuracy vectors.

    MoRF is ranked ascending (lower accuracy = higher rank = better attribution).
    LeRF is ranked descending (negated before ranking).

    Args:
        r1: MoRF accuracies for each attribution method at a fixed step.
        r2: LeRF accuracies for each attribution method at the same step.

    Returns:
        (rankq1, rankq2, rho_diff, rho_corr)
          rankq1/rankq2 — tie-adjusted rank vectors
          rho_diff      — Spearman ρ via the d² formula
          rho_corr      — Spearman ρ via Pearson correlation of ranks
    """
    rank1 = rankdata(r1)                    * 1.0
    rank2 = rankdata(-np.asarray(r2, float)) * 1.0

    rankq1, rankq2 = rank1.copy(), rank2.copy()
    # half-integer tie adjustment
    for rk, rq in ((rank1, rankq1), (rank2, rankq2)):
        uq, cnt = np.unique(rk, return_counts=True)
        for u, c in zip(uq, cnt):
            if c > 1 and u % 1 > 0:
                rq[rk == u] += 0.5

    rho_corr = float(np.corrcoef(rankq1, rankq2)[0, 1])
    n        = len(rankq1)
    rho_diff = float(1 - 6 * np.sum((rankq1 - rankq2) ** 2) / (n * (n * n - 1)))
    return rankq1, rankq2, rho_diff, rho_corr


# ─────────────────────────────────────────────────────────────────────────────
# Image domain
# ─────────────────────────────────────────────────────────────────────────────

_IMG_DATASETS = ["brain_mri", "imagenet", "oxford_pet"]
_IMG_MODELS   = ["resnet_50", "efficientnet_b0", "repvgg_b0"]
_IMG_MASKS    = ["zero", "pgd", "road"]
_IMG_METHODS  = [
    "gradients", "smoothgrad", "integrad", "gradinput",
    "gradcam", "gradcampp", "scorecam", "smoothgradcampp", "random",
]


def _compute_folder_spearman(
    folder: str,
    method_order: list[str],
    drop_first: bool,
) -> tuple[list[str], pd.DataFrame]:
    """Per-step Spearman ρ over all methods in a result folder."""
    valid_methods, morf_curves, lerf_curves = [], [], []
    for method in method_order:
        mp = os.path.join(folder, f"{method}_morf.pkl")
        lp = os.path.join(folder, f"{method}_lerf.pkl")
        if not (os.path.isfile(mp) and os.path.isfile(lp)):
            continue
        try:
            m = _load_curve_pkl(mp, drop_first=drop_first)
            l = _load_curve_pkl(lp, drop_first=drop_first)
        except Exception as exc:
            print(f"[WARN] {method} in {folder}: {exc}")
            continue
        L = min(len(m), len(l))
        valid_methods.append(method)
        morf_curves.append(m[:L])
        lerf_curves.append(l[:L])

    if len(valid_methods) < 2:
        raise ValueError(f"Need ≥2 methods; found {valid_methods} in {folder}")

    min_len = min(len(x) for x in morf_curves)
    M = np.stack([x[:min_len] for x in morf_curves], axis=0)  # (n_methods, T)
    L = np.stack([x[:min_len] for x in lerf_curves], axis=0)

    rows = []
    for k in range(min_len):
        _, _, rho_diff, rho_corr = spearman_rank(M[:, k], L[:, k])
        rows.append({"step": k + 1, "rho_diff": rho_diff, "rho_corr": rho_corr})
    return valid_methods, pd.DataFrame(rows)


def _run_image(args: argparse.Namespace) -> None:
    root = args.root or os.path.join(PROJECT_ROOT, "morf_lerf_image")
    datasets = _IMG_DATASETS if args.dataset == "all" else [args.dataset]
    models   = _IMG_MODELS   if args.model   == "all" else [args.model]
    masks    = _IMG_MASKS    if args.mask    == "all" else [args.mask]

    summary_rows: list[dict] = []

    for dataset in datasets:
        for model in models:
            for mask in masks:
                folder = os.path.join(root, model, dataset, mask)
                if not os.path.isdir(folder):
                    print(f"[SKIP] {folder}")
                    continue
                try:
                    methods, df_step = _compute_folder_spearman(
                        folder, _IMG_METHODS, drop_first=False)
                except Exception as exc:
                    print(f"[SKIP] {folder}: {exc}")
                    continue
                diff = df_step["rho_diff"].values
                corr = df_step["rho_corr"].values
                summary_rows.append({
                    "domain":  "image", "dataset": dataset,
                    "model":   model,   "mask":     mask,
                    "n_methods":     len(methods),
                    "n_steps":       len(df_step),
                    "rho_diff_mean": float(diff.mean()),
                    "rho_diff_std":  float(diff.std(ddof=0)),
                    "rho_corr_mean": float(corr.mean()),
                    "rho_corr_std":  float(corr.std(ddof=0)),
                })
                print(
                    f"[OK] {dataset}/{model}/{mask}  "
                    f"ρ_diff={diff.mean():.3f}±{diff.std(ddof=0):.3f}  "
                    f"ρ_corr={corr.mean():.3f}±{corr.std(ddof=0):.3f}"
                )

    if not summary_rows:
        print("[WARN] No image results found.")
        return

    df_out = pd.DataFrame(summary_rows)
    out_csv = args.out_csv or "image_spearman_summary.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"\nSaved to: {out_csv}")


# ─────────────────────────────────────────────────────────────────────────────
# Audio domain
# ─────────────────────────────────────────────────────────────────────────────

_AUD_DATASETS = ["audiomnist", "esc50", "msos"]
_AUD_MODELS   = ["alexnet", "cnn14", "audionet", "res1dnet31"]
_AUD_MASKS    = ["zero", "pgd", "road"]
_AUD_METHODS  = [
    "gradient", "gradinput", "smoothgrad", "smoothgrad_sq", "vargrad",
    "integrad", "gradient_abs", "gradinput_abs", "smoothgrad_abs",
    "integrad_abs", "random",
]


def _run_audio(args: argparse.Namespace) -> None:
    root = args.root or os.path.join(PROJECT_ROOT, "audio", "morf_lerf")
    datasets = _AUD_DATASETS if args.dataset == "all" else [args.dataset]
    models   = _AUD_MODELS   if args.model   == "all" else [args.model]
    masks    = _AUD_MASKS    if args.mask    == "all" else [args.mask]

    summary_rows:   list[dict] = []
    fold_rows_all:  list[dict] = []
    ratio_rows_all: list[dict] = []

    for dataset in datasets:
        for model in models:
            for mask in masks:
                if dataset == "esc50":
                    folders = sorted(
                        p for p in glob.glob(
                            os.path.join(root, dataset, model, mask, "fold_*"))
                        if os.path.isdir(p)
                    )
                else:
                    fd = os.path.join(root, dataset, model, mask)
                    folders = [fd] if os.path.isdir(fd) else []

                fold_tables: list[pd.DataFrame] = []
                fold_names:  list[str]          = []
                for folder in folders:
                    fold_name = os.path.basename(folder) if dataset == "esc50" else "no_fold"
                    try:
                        methods, df_step = _compute_folder_spearman(
                            folder, _AUD_METHODS, drop_first=True)
                    except Exception as exc:
                        print(f"[SKIP] {folder}: {exc}")
                        continue
                    fold_names.append(fold_name)
                    fold_tables.append(df_step)
                    fold_rows_all.append({
                        "dataset": dataset, "model": model, "mask": mask,
                        "fold":     fold_name,
                        "n_methods": len(methods),
                        "rho_diff_mean": float(df_step["rho_diff"].mean()),
                        "rho_diff_std":  float(df_step["rho_diff"].std(ddof=0)),
                        "rho_corr_mean": float(df_step["rho_corr"].mean()),
                        "rho_corr_std":  float(df_step["rho_corr"].std(ddof=0)),
                    })

                if not fold_tables:
                    continue

                min_len = min(len(df) for df in fold_tables)
                diff_mat = np.stack([df["rho_diff"].values[:min_len] for df in fold_tables])
                corr_mat = np.stack([df["rho_corr"].values[:min_len] for df in fold_tables])

                for k in range(min_len):
                    ratio_rows_all.append({
                        "dataset": dataset, "model": model, "mask": mask,
                        "step":          k + 1,
                        "rho_diff_mean": float(diff_mat[:, k].mean()),
                        "rho_diff_std":  float(diff_mat[:, k].std(ddof=0)),
                        "rho_corr_mean": float(corr_mat[:, k].mean()),
                        "rho_corr_std":  float(corr_mat[:, k].std(ddof=0)),
                    })

                flat_diff = diff_mat.ravel()
                flat_corr = corr_mat.ravel()
                summary_rows.append({
                    "domain":  "audio", "dataset": dataset,
                    "model":   model,   "mask":     mask,
                    "n_folds":       diff_mat.shape[0],
                    "rho_diff_mean": float(flat_diff.mean()),
                    "rho_diff_std":  float(flat_diff.std(ddof=0)),
                    "rho_corr_mean": float(flat_corr.mean()),
                    "rho_corr_std":  float(flat_corr.std(ddof=0)),
                })
                print(
                    f"[OK] {dataset}/{model}/{mask}  folds={fold_names}  "
                    f"ρ_diff={flat_diff.mean():.3f}±{flat_diff.std(ddof=0):.3f}"
                )

    if not summary_rows:
        print("[WARN] No audio results found.")
        return

    out_csv     = args.out_csv or "audio_spearman_summary.csv"
    out_ratio   = getattr(args, "out_ratio",   None) or "audio_spearman_by_step.csv"
    out_fold    = getattr(args, "out_fold",    None) or "audio_spearman_by_fold.csv"

    pd.DataFrame(summary_rows).to_csv(out_csv, index=False)
    pd.DataFrame(ratio_rows_all).to_csv(out_ratio, index=False)
    pd.DataFrame(fold_rows_all).to_csv(out_fold, index=False)
    print(f"\nSaved: {out_csv}, {out_ratio}, {out_fold}")


# ─────────────────────────────────────────────────────────────────────────────
# EEG domain
# ─────────────────────────────────────────────────────────────────────────────

_EEG_BASE_DIR    = "/mnt/left/home/2025/tony/hsinyuan/AIM_eeg/experiment"
_EEG_DATASETS    = ["MI", "ERN", "SSVEP"]
_EEG_MODELS      = ["eegnet", "icnn", "sccnet"]
_EEG_DOMAINS     = ["ch", "ts", "fq"]
_EEG_DOMAIN_NAME = {"ch": "Spatial", "ts": "Temporal", "fq": "Spectral"}
_EEG_SUFFIX      = {"ch": "", "ts": "_sto", "fq": "_si"}
_LEG_NABS = ["GD", "GI", "SG", "SS", "VG", "IG"]
_LEG_ABS  = ["GDA", "GIA", "SGA", "IGA"]
_LEG_ALL  = _LEG_NABS + _LEG_ABS + ["RD"]


def _run_eeg(args: argparse.Namespace) -> None:
    base     = args.root or _EEG_BASE_DIR
    datasets = _EEG_DATASETS if args.dataset    == "all" else [args.dataset]
    models   = _EEG_MODELS   if args.model      == "all" else [args.model]
    domains  = _EEG_DOMAINS  if args.eeg_domain == "all" else [args.eeg_domain]

    rows: list[dict] = []

    for dataset in datasets:
        for model in models:
            for domain in domains:
                suffix   = _EEG_SUFFIX[domain]
                all_rhos: list[float] = []

                for rep in range(5):
                    data_dir = os.path.join(
                        base, f"repeat{rep}",
                        f"{domain}_test_zero", f"{dataset}{suffix}",
                    )
                    try:
                        morf_nabs = _load_pickle(os.path.join(
                            data_dir, f"{model}_{domain}_nabs_morf.pickle"))
                        lerf_nabs = _load_pickle(os.path.join(
                            data_dir, f"{model}_{domain}_nabs_lerf.pickle"))
                        morf_abs  = _load_pickle(os.path.join(
                            data_dir, f"{model}_{domain}_abs_morf.pickle"))
                        lerf_abs  = _load_pickle(os.path.join(
                            data_dir, f"{model}_{domain}_abs_lerf.pickle"))
                    except FileNotFoundError:
                        continue

                    # Build per-method mean curves: shape (n_methods, K)
                    morf_stack, lerf_stack = [], []
                    for i in range(len(_LEG_NABS)):
                        morf_stack.append(morf_nabs[i]["acc"].mean(axis=0))
                        lerf_stack.append(lerf_nabs[i]["acc"].mean(axis=0))
                    for i in range(len(_LEG_ABS)):
                        morf_stack.append(morf_abs[i]["acc"].mean(axis=0))
                        lerf_stack.append(lerf_abs[i]["acc"].mean(axis=0))
                    # RD
                    morf_stack.append(
                        (morf_nabs[-1]["acc"] + morf_abs[-1]["acc"]).mean(axis=0) / 2
                    )
                    lerf_stack.append(
                        (lerf_nabs[-1]["acc"] + lerf_abs[-1]["acc"]).mean(axis=0) / 2
                    )

                    morf_arr = np.array(morf_stack)  # (n_methods, K)
                    lerf_arr = np.array(lerf_stack)
                    K        = morf_arr.shape[1]
                    use_K    = K // 2 if (domain == "ch" and dataset != "ERN") else K

                    for k in range(use_K):
                        rho = spearman_rank(morf_arr[:, k], lerf_arr[:, k])[2]
                        all_rhos.append(rho)

                if not all_rhos:
                    continue
                rho_arr = np.array(all_rhos)
                rows.append({
                    "domain":   "eeg",
                    "dataset":  dataset,
                    "model":    model,
                    "eeg_domain":      domain,
                    "eeg_domain_name": _EEG_DOMAIN_NAME[domain],
                    "rho_mean":  float(np.nanmean(rho_arr)),
                    "rho_std":   float(np.nanstd(rho_arr)),
                    "n_reps":    5,
                })

    if not rows:
        print("[WARN] No EEG results found.")
        return

    df = pd.DataFrame(rows)

    # Paper-style table
    W = 14
    cell = lambda x: f"{x:<{W}}"
    print("\n" + "=" * 120)
    print(
        cell("Domain") +
        cell("SMR-EEGNet") + cell("SMR-ICNN") + cell("SMR-SCCNet") +
        cell("ERN-EEGNet") + cell("ERN-ICNN") + cell("ERN-SCCNet") +
        cell("SSVEP-EEGNet") + cell("SSVEP-ICNN") + cell("SSVEP-SCCNet")
    )
    print("-" * 120)
    for domain_key in ["ch", "ts", "fq"]:
        row_str = [cell(_EEG_DOMAIN_NAME[domain_key])]
        for ds in ["MI", "ERN", "SSVEP"]:
            for m in ["eegnet", "icnn", "sccnet"]:
                sub = df[(df["dataset"] == ds) & (df["model"] == m) &
                         (df["eeg_domain"] == domain_key)]
                if len(sub) == 0:
                    row_str.append(cell("-"))
                else:
                    r = sub.iloc[0]
                    row_str.append(cell(f"{r['rho_mean']:.3f}±{r['rho_std']:.3f}"))
        print("".join(row_str))
    print("=" * 120)

    out_csv = args.out_csv or "eeg_spearman_summary.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved to: {out_csv}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute MoRF–LeRF Spearman ρ (image / audio / EEG).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--domain", required=True, choices=["image", "audio", "eeg"])
    parser.add_argument("--root",       default=None)
    parser.add_argument("--dataset",    default="all")
    parser.add_argument("--model",      default="all")
    parser.add_argument("--mask",       default="all")
    parser.add_argument("--eeg_domain", default="all",
                        choices=["all", "ch", "ts", "fq"])
    parser.add_argument("--out_csv",    default=None)
    # audio-specific multi-file outputs
    parser.add_argument("--out_ratio",   default=None,
                        help="(audio) Per-step ρ CSV.")
    parser.add_argument("--out_fold",    default=None,
                        help="(audio) Per-fold ρ CSV.")
    # backward compat alias
    parser.add_argument("--out_summary", dest="out_csv", default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--drop_first", action="store_true",
                        help=argparse.SUPPRESS)
    args = parser.parse_args()

    dispatch = {"image": _run_image, "audio": _run_audio, "eeg": _run_eeg}
    dispatch[args.domain](args)


if __name__ == "__main__":
    main()
