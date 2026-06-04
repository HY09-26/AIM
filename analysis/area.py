"""Unified AOC / ABC / AUC area metrics — image, audio, and EEG domains.

Usage examples
--------------
# from project root:
python analysis/area.py --domain image
python analysis/area.py --domain audio --dataset audiomnist --model audionet
python analysis/area.py --domain eeg   --eeg_domain ch

# or via the per-domain thin wrappers (original paths still work):
python image/experiment/area.py
python audio/experiment/area.py --dataset esc50 --model cnn14
python eeg/result_process/result_areas.py --eeg_domain fq

Formulas
--------
Image  — normalised (paper formula):
  AOC = (1/K) Σ_k clip((acc0 - morf_k) / (acc0 - morf_K), 0, 1)
  ABC = (1/K) Σ_k clip((lerf_k - morf_k) / (acc0 - morf_K), 0, 1)
  AUC = (1/K) Σ_k clip((lerf_k - lerf_K) / (acc0 - lerf_K), 0, 1)

Audio / EEG — unnormalised with chance clipping:
  AOC = mean(1 - morf),  ABC = mean(clip(lerf - morf, 0)),  AUC = mean(lerf)
  Curves are clipped to [chance, 1] before computing.
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle

import numpy as np
import pandas as pd

ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ANALYSIS_DIR)


# ─────────────────────────────────────────────────────────────────────────────
# Generic I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

class _CompatUnpickler(pickle.Unpickler):
    """Remap numpy._core → numpy.core for cross-version compatibility."""
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core")
        return super().find_class(module, name)


def _load_pickle(path: str):
    with open(path, "rb") as f:
        return _CompatUnpickler(f).load()


def _load_curve_pkl(path: str, drop_first: bool = False) -> np.ndarray:
    """Load a 1-D accuracy curve from a .pkl file (image / audio format)."""
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


def _scan_pkl_pairs(folder: str) -> dict[str, tuple[str, str]]:
    """Return {method: (morf_path, lerf_path)} for all *_morf.pkl / *_lerf.pkl pairs."""
    pairs: dict[str, dict[str, str]] = {}
    for fname in os.listdir(folder):
        if fname.endswith("_morf.pkl"):
            pairs.setdefault(fname[:-9], {})["morf"] = os.path.join(folder, fname)
        elif fname.endswith("_lerf.pkl"):
            pairs.setdefault(fname[:-9], {})["lerf"] = os.path.join(folder, fname)
    return {
        m: (v["morf"], v["lerf"])
        for m, v in pairs.items()
        if "morf" in v and "lerf" in v
    }


# ─────────────────────────────────────────────────────────────────────────────
# Metric formulas
# ─────────────────────────────────────────────────────────────────────────────

def _compute_area_normalised(
    morf: np.ndarray,
    lerf: np.ndarray,
) -> tuple[float, float, float]:
    """Normalised AOC / ABC / AUC (image domain).

    Step 0 (index 0) is the unmasked accuracy; step K (last index) is the
    all-masked accuracy.  Returns (nan, nan, nan) if denominators are ~0.
    """
    L = min(len(morf), len(lerf))
    morf, lerf = morf[:L], lerf[:L]
    acc0   = morf[0]
    accM_K = morf[-1]  # all-masked under MoRF masking
    accL_K = lerf[-1]  # all-masked under LeRF masking
    denom_M = acc0 - accM_K
    denom_L = acc0 - accL_K
    if np.isclose(denom_M, 0.0) or np.isclose(denom_L, 0.0):
        return np.nan, np.nan, np.nan
    K_plus1 = L
    aoc = sum(min((acc0 - morf[k]) / denom_M, 1.0) for k in range(K_plus1)) / K_plus1
    abc = sum(min((lerf[k] - morf[k]) / denom_M, 1.0) for k in range(K_plus1)) / K_plus1
    auc = sum(min((lerf[k] - accL_K) / denom_L, 1.0) for k in range(K_plus1)) / K_plus1
    return float(aoc), float(abc), float(auc)


def _compute_area_unnorm(
    morf: np.ndarray,
    lerf: np.ndarray,
    chance: float = 0.0,
) -> tuple[float, float, float]:
    """Unnormalised AOC / ABC / AUC with chance-level clipping (audio / EEG)."""
    L = min(len(morf), len(lerf))
    m = np.clip(morf[:L], chance, 1.0)
    l = np.clip(lerf[:L], chance, 1.0)
    return (
        float(np.mean(1.0 - m)),
        float(np.mean(np.clip(l - m, 0.0, None))),
        float(np.mean(l)),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Image domain
# ─────────────────────────────────────────────────────────────────────────────

_IMG_DATASETS = ["brain_mri", "imagenet", "oxford_pet"]
_IMG_MODELS   = ["efficientnet_b0", "repvgg_b0", "resnet_50"]
_IMG_MASKS    = ["zero", "pgd", "road"]
_IMG_METHOD_ORDER = [
    "gradients", "smoothgrad", "integrad", "gradinput",
    "gradcam", "gradcampp", "scorecam", "smoothgradcampp", "random",
]
_IMG_METHOD_DISPLAY = {
    "gradients": "GD", "smoothgrad": "SG", "integrad": "IG", "gradinput": "GI",
    "gradcam": "GC", "gradcampp": "GC++", "scorecam": "SC",
    "smoothgradcampp": "SGC++", "random": "RD",
}
_IMG_MASK_DISPLAY = {"pgd": "mdAR", "road": "mdROAD", "zero": "Zeroing"}
_IMG_MASK_ORDER   = ["mdROAD", "mdAR", "Zeroing"]
_IMG_METRIC_ORDER = ["AOC", "ABC", "AUC"]


def _run_image(args: argparse.Namespace) -> None:
    root = args.root or os.path.join(PROJECT_ROOT, "morf_lerf_image")
    datasets = _IMG_DATASETS if args.dataset == "all" else [args.dataset]
    models   = _IMG_MODELS   if args.model   == "all" else [args.model]
    masks    = _IMG_MASKS    if args.mask    == "all" else [args.mask]

    rows: list[dict] = []
    for dataset in datasets:
        for model in models:
            for mask in masks:
                folder = os.path.join(root, model, dataset, mask)
                if not os.path.isdir(folder):
                    print(f"[SKIP] {folder}")
                    continue
                for method, (pm, pl) in _scan_pkl_pairs(folder).items():
                    try:
                        morf = _load_curve_pkl(pm, drop_first=False)
                        lerf = _load_curve_pkl(pl, drop_first=False)
                    except Exception as exc:
                        print(f"[WARN] {method} in {folder}: {exc}")
                        continue
                    aoc, abc, auc = _compute_area_normalised(morf, lerf)
                    rows.append({
                        "domain": "image", "dataset": dataset,
                        "model": model, "mask": mask,
                        "mask_show":   _IMG_MASK_DISPLAY.get(mask, mask),
                        "method":      method,
                        "method_show": _IMG_METHOD_DISPLAY.get(method, method),
                        "AOC": aoc, "ABC": abc, "AUC": auc,
                    })

    if not rows:
        print("[WARN] No image results found. Check --root path.")
        return

    df = pd.DataFrame(rows)
    print("\n=== IMAGE — raw results ===")
    print(df.to_string(index=False))

    # Summary: mean ± std aggregated over dataset × model
    fmt = lambda mu, sd: f"{mu:.3f}±{sd:.3f}" if not np.isnan(mu) else ""
    summary = (
        df.groupby(["mask_show", "method", "method_show"], as_index=False)
        .agg(
            AOC_mean=("AOC", "mean"), AOC_std=("AOC", "std"),
            ABC_mean=("ABC", "mean"), ABC_std=("ABC", "std"),
            AUC_mean=("AUC", "mean"), AUC_std=("AUC", "std"),
        )
    )
    summary["method"] = pd.Categorical(
        summary["method"], categories=_IMG_METHOD_ORDER, ordered=True)
    summary["mask_show"] = pd.Categorical(
        summary["mask_show"], categories=_IMG_MASK_ORDER, ordered=True)
    summary = summary.sort_values(["method", "mask_show"]).reset_index(drop=True)

    print("\n=== IMAGE — summary (mean ± std over dataset × model) ===")
    rows_compact = []
    for _, r in summary.iterrows():
        rows_compact.append({
            "method": r["method_show"],
            "mask":   r["mask_show"],
            "AOC":    fmt(r["AOC_mean"], r["AOC_std"]),
            "ABC":    fmt(r["ABC_mean"], r["ABC_std"]),
            "AUC":    fmt(r["AUC_mean"], r["AUC_std"]),
        })
    compact = pd.DataFrame(rows_compact)
    pivot = compact.pivot_table(
        index="method", columns="mask",
        values=["AOC", "ABC", "AUC"], aggfunc="first",
    )
    wanted = [(m, ms) for ms in _IMG_MASK_ORDER for m in _IMG_METRIC_ORDER
              if (m, ms) in pivot.columns]
    pivot = pivot[wanted].swaplevel(0, 1, axis=1).sort_index(axis=1, level=0)
    print(pivot.to_string())

    out_csv = args.out_csv or "image_area_metrics.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved to: {out_csv}")


# ─────────────────────────────────────────────────────────────────────────────
# Audio domain
# ─────────────────────────────────────────────────────────────────────────────

_AUD_CHANCE   = {"audiomnist": 1 / 10, "esc50": 1 / 50, "msos": 1 / 5}
_AUD_DATASETS = ["audiomnist", "esc50", "msos"]
_AUD_MODELS   = ["alexnet", "cnn14", "audionet", "res1dnet31"]
_AUD_MASKS    = ["zero", "pgd", "road"]
_AUD_METHOD_DISPLAY = {
    "gradient": "GD", "gradinput": "GI", "smoothgrad": "SG",
    "smoothgrad_sq": "SS", "vargrad": "VG", "integrad": "IG",
    "gradient_abs": "GDA", "gradinput_abs": "GIA",
    "smoothgrad_abs": "SGA", "integrad_abs": "IGA", "random": "RD",
}


def _run_audio(args: argparse.Namespace) -> None:
    root = args.root or os.path.join(PROJECT_ROOT, "audio", "morf_lerf")
    datasets = _AUD_DATASETS if args.dataset == "all" else [args.dataset]
    models   = _AUD_MODELS   if args.model   == "all" else [args.model]
    masks    = _AUD_MASKS    if args.mask    == "all" else [args.mask]

    rows: list[dict] = []
    for dataset in datasets:
        chance = _AUD_CHANCE[dataset]
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

                for folder in folders:
                    fold = os.path.basename(folder) if dataset == "esc50" else None
                    for method, (pm, pl) in _scan_pkl_pairs(folder).items():
                        try:
                            morf = _load_curve_pkl(pm, drop_first=True)
                            lerf = _load_curve_pkl(pl, drop_first=True)
                        except Exception as exc:
                            print(f"[WARN] {method} in {folder}: {exc}")
                            continue
                        aoc, abc, auc = _compute_area_unnorm(morf, lerf, chance)
                        rows.append({
                            "domain": "audio", "dataset": dataset,
                            "model": model, "mask": mask, "fold": fold,
                            "method":      method,
                            "method_show": _AUD_METHOD_DISPLAY.get(method, method),
                            "AOC": aoc, "ABC": abc, "AUC": auc,
                        })

    if not rows:
        print("[WARN] No audio results found. Check --root path.")
        return

    df = pd.DataFrame(rows)

    # Aggregate ESC-50 across folds
    agg_df = (
        df.groupby(["dataset", "model", "mask", "method", "method_show"], as_index=False)
        .agg(AOC_mean=("AOC", "mean"), AOC_std=("AOC", "std"),
             ABC_mean=("ABC", "mean"), ABC_std=("ABC", "std"),
             AUC_mean=("AUC", "mean"), AUC_std=("AUC", "std"))
    )

    print("\n=== AUDIO — metrics (mean±std over folds) ===")
    for _, r in agg_df.iterrows():
        print(
            f"{r['dataset']:10s} {r['model']:12s} {r['mask']:5s} "
            f"{r['method_show']:>5s}  "
            f"AOC={r['AOC_mean']:.3f}±{r['AOC_std']:.3f}  "
            f"ABC={r['ABC_mean']:.3f}±{r['ABC_std']:.3f}  "
            f"AUC={r['AUC_mean']:.3f}±{r['AUC_std']:.3f}"
        )

    out_csv = args.out_csv or "audio_area_metrics.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved to: {out_csv}")


# ─────────────────────────────────────────────────────────────────────────────
# EEG domain
# ─────────────────────────────────────────────────────────────────────────────

_EEG_BASE_DIR    = "/mnt/left/home/2025/tony/hsinyuan/AIM_eeg/experiment"
_EEG_DATASETS    = ["MI", "ERN", "SSVEP"]
_EEG_MODELS      = ["eegnet", "icnn", "sccnet"]
_EEG_DOMAINS     = ["ch", "ts", "fq"]
_EEG_DOMAIN_NAME = {"ch": "Spatial", "ts": "Temporal", "fq": "Spectral"}
_EEG_CHANCE      = {"MI": 0.25, "ERN": 0.5, "SSVEP": 0.2}
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
        chance = _EEG_CHANCE[dataset]
        for model in models:
            for domain in domains:
                suffix = _EEG_SUFFIX[domain]
                # accumulate per-rep metrics then average
                per_method: dict[str, dict[str, list[float]]] = {
                    leg: {"aoc": [], "abc": [], "auc": []} for leg in _LEG_ALL
                }

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

                    K     = morf_nabs[0]["acc"].shape[1]
                    use_K = K // 2 if (domain == "ch" and dataset != "ERN") else K

                    for i, leg in enumerate(_LEG_NABS):
                        mc = morf_nabs[i]["acc"].mean(axis=0)[:use_K]
                        lc = lerf_nabs[i]["acc"].mean(axis=0)[:use_K]
                        aoc, abc, auc = _compute_area_unnorm(mc, lc, chance)
                        per_method[leg]["aoc"].append(aoc)
                        per_method[leg]["abc"].append(abc)
                        per_method[leg]["auc"].append(auc)

                    for i, leg in enumerate(_LEG_ABS):
                        mc = morf_abs[i]["acc"].mean(axis=0)[:use_K]
                        lc = lerf_abs[i]["acc"].mean(axis=0)[:use_K]
                        aoc, abc, auc = _compute_area_unnorm(mc, lc, chance)
                        per_method[leg]["aoc"].append(aoc)
                        per_method[leg]["abc"].append(abc)
                        per_method[leg]["auc"].append(auc)

                    # RD = average of the last nabs entry and last abs entry
                    mc_rd = (morf_nabs[-1]["acc"] + morf_abs[-1]["acc"]) / 2
                    lc_rd = (lerf_nabs[-1]["acc"] + lerf_abs[-1]["acc"]) / 2
                    aoc, abc, auc = _compute_area_unnorm(
                        mc_rd.mean(axis=0)[:use_K],
                        lc_rd.mean(axis=0)[:use_K],
                        chance,
                    )
                    per_method["RD"]["aoc"].append(aoc)
                    per_method["RD"]["abc"].append(abc)
                    per_method["RD"]["auc"].append(auc)

                for leg in _LEG_ALL:
                    vals = per_method[leg]
                    if not vals["aoc"]:
                        continue
                    rows.append({
                        "domain": "eeg", "dataset": dataset,
                        "model": model, "eeg_domain": domain,
                        "eeg_domain_name": _EEG_DOMAIN_NAME[domain],
                        "method": leg, "method_show": leg,
                        "AOC_mean": float(np.mean(vals["aoc"])),
                        "AOC_std":  float(np.std(vals["aoc"])),
                        "ABC_mean": float(np.mean(vals["abc"])),
                        "ABC_std":  float(np.std(vals["abc"])),
                        "AUC_mean": float(np.mean(vals["auc"])),
                        "AUC_std":  float(np.std(vals["auc"])),
                    })

    if not rows:
        print("[WARN] No EEG results found. Check --root path.")
        return

    df = pd.DataFrame(rows)
    print("\n=== EEG — metrics (mean±std over reps) ===")
    for _, r in df.iterrows():
        print(
            f"{r['dataset']:6s} {r['model']:7s} {r['eeg_domain_name']:9s} "
            f"{r['method']:>5s}  "
            f"AOC={r['AOC_mean']:.3f}±{r['AOC_std']:.3f}  "
            f"ABC={r['ABC_mean']:.3f}±{r['ABC_std']:.3f}  "
            f"AUC={r['AUC_mean']:.3f}±{r['AUC_std']:.3f}"
        )

    out_csv = args.out_csv or "eeg_area_metrics.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved to: {out_csv}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute AOC / ABC / AUC area metrics (image / audio / EEG).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--domain", required=True, choices=["image", "audio", "eeg"],
                        help="Experiment domain.")
    parser.add_argument("--root",       default=None,
                        help="Override result root directory.")
    parser.add_argument("--dataset",    default="all",
                        help="Dataset name, or 'all'.")
    parser.add_argument("--model",      default="all",
                        help="Model name, or 'all'.")
    parser.add_argument("--mask",       default="all",
                        help="Mask type (zero/pgd/road), or 'all'.")
    parser.add_argument("--eeg_domain", default="all",
                        choices=["all", "ch", "ts", "fq"],
                        help="EEG feature domain (spatial/temporal/spectral).")
    parser.add_argument("--out_csv",    default=None,
                        help="Output CSV path. Defaults to <domain>_area_metrics.csv.")
    # kept for backward compat with audio/experiment/area.py --drop_first flag;
    # audio always drops step 0 internally so this flag is a no-op.
    parser.add_argument("--drop_first", action="store_true",
                        help=argparse.SUPPRESS)
    args = parser.parse_args()

    dispatch = {"image": _run_image, "audio": _run_audio, "eeg": _run_eeg}
    dispatch[args.domain](args)


if __name__ == "__main__":
    main()
