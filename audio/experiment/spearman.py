# import os
# import argparse
# import pickle
# import numpy as np
# from scipy.stats import rankdata

# # =========================
# # Argument parser
# # =========================
# parser = argparse.ArgumentParser(description="Audio AIM Spearman rank consistency")
# parser.add_argument("--model", type=str, default="res1dnet31", choices=["alexnet", "audionet", "res1dnet31", "cnn14"])
# parser.add_argument("--dataset", type=str, default="esc50", choices=["esc50", "msos", "audiomnist"])
# parser.add_argument("--mask_type", type=str, default="zero", choices=["zero", "pgd", "road"],)
# args = parser.parse_args()

# # =========================
# # Path (from arguments)
# # =========================
# PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# DATA_ROOT = os.path.join(
#     PROJECT_ROOT,
#     "morf_lerf",
#     args.model,
#     args.dataset,
#     args.mask_type,
# )

# # =========================
# # Config (ONLY existing methods)
# # =========================
# METHOD_MAP = {
#     "gradient": "GD",
#     "gradinput": "GI",
#     "integrad": "IG",
#     "smoothgrad": "SG",
#     "random": "RD",
# }

# METHOD_MAP_ABS = {
#     "gradients_abs": "GDA",
#     "gradinput_abs": "GIA",
#     "integrad_abs": "IGA",
#     "smoothgrad_abs": "SGA",
#     "smoothgrad_sq": "SGS",
#     "vargrad": "VG",
# }

# # =========================
# # Utils
# # =========================
# def load_pkl(path):
#     with open(path, "rb") as f:
#         return pickle.load(f)


# def _as_curve(obj):
#     """
#     Robustly extract a 1D performance curve
#     """
#     if isinstance(obj, dict):
#         for k in ["acc", "accuracy", "acc_curve", "curve", "accs", "acc_list"]:
#             if k in obj:
#                 obj = obj[k]
#                 break
#         else:
#             for v in obj.values():
#                 if isinstance(v, (list, tuple, np.ndarray)):
#                     obj = v
#                     break

#     arr = np.asarray(obj)
#     if arr.ndim == 1:
#         return arr.astype(float)
#     if arr.ndim >= 2:
#         return arr.mean(axis=0).astype(float)

#     raise ValueError("Cannot extract curve.")


# def parse_fname(fname):
#     """
#     Expected filename examples:
#       gradcam_morf.pkl
#       gradcam_lerf.pkl
#       gradients_abs_morf.pkl
#       gradients_abs_lerf.pkl
#     """
#     name = fname.replace(".pkl", "")
#     parts = name.split("_")
#     if len(parts) < 2:
#         return None

#     kind = parts[-1]
#     if kind not in ("morf", "lerf"):
#         return None

#     tokens = parts[:-1]
#     is_abs = "abs" in tokens
#     method_tokens = [t for t in tokens if t != "abs"]
#     method_raw = "_".join(method_tokens).lower()

#     return {
#         "method_raw": method_raw,
#         "is_abs": is_abs,
#         "kind": kind,
#     }


# def method_short(method_raw, is_abs):
#     if is_abs:
#         return METHOD_MAP_ABS.get(method_raw, None)
#     return METHOD_MAP.get(method_raw, None)


# def spearman_rank(x, y):
#     """
#     MoRF: lower is better
#     LeRF: higher is better
#     """
#     rx = rankdata(x)
#     ry = rankdata(-y)
#     n = len(rx)
#     return 1 - (6 * np.sum((rx - ry) ** 2)) / (n * (n * n - 1))


# # =========================
# # Scan flat folder
# # =========================
# def scan_pairs(mask_dir):
#     rows = []

#     for fname in os.listdir(mask_dir):
#         if not fname.endswith(".pkl"):
#             continue

#         meta = parse_fname(fname)
#         if meta is None:
#             continue

#         ms = method_short(meta["method_raw"], meta["is_abs"])
#         if ms is None:
#             continue

#         meta["method"] = ms
#         meta["path"] = os.path.join(mask_dir, fname)
#         rows.append(meta)

#     # method -> {morf, lerf}
#     tmp = {}
#     for r in rows:
#         tmp.setdefault(r["method"], {})[r["kind"]] = r["path"]

#     pairs = {}
#     for m, v in tmp.items():
#         if "morf" in v and "lerf" in v:
#             pairs[m] = (v["morf"], v["lerf"])

#     return pairs


# # =========================
# # Main
# # =========================
# if __name__ == "__main__":

#     print("PROJECT_ROOT =", PROJECT_ROOT)
#     print("DATA_ROOT    =", DATA_ROOT)

#     if not os.path.isdir(DATA_ROOT):
#         raise FileNotFoundError(f"DATA_ROOT not found: {DATA_ROOT}")

#     pairs = scan_pairs(DATA_ROOT)
#     if len(pairs) == 0:
#         raise RuntimeError(
#             "No morf/lerf pairs found. "
#             "Check filenames like *_morf.pkl and *_lerf.pkl."
#         )

#     methods = sorted(pairs.keys())
#     print("[INFO] Methods:", methods)

#     curvesM, curvesL = {}, {}
#     for m in methods:
#         pm, pl = pairs[m]
#         curveM = _as_curve(load_pkl(pm))
#         curveL = _as_curve(load_pkl(pl))

#         # drop 0% masked
#         curvesM[m] = curveM[1:] if len(curveM) >= 2 else curveM
#         curvesL[m] = curveL[1:] if len(curveL) >= 2 else curveL

#     K = min(len(curvesM[m]) for m in methods)
#     if K < 2:
#         raise RuntimeError("Curve too short for Spearman computation.")

#     rhos = []
#     for r in range(K):
#         perfM = np.array([curvesM[m][r] for m in methods], dtype=float)
#         perfL = np.array([curvesL[m][r] for m in methods], dtype=float)
#         rhos.append(spearman_rank(perfM, perfL))

#     rhos = np.asarray(rhos)
#     rhos = rhos[:-1]
#     print(
#         f"Audio | {args.dataset} | {args.mask_type} | {args.model} : "
#         f"ρ = {rhos.mean():.3f} ± {rhos.std():.3f} (K={K})"
#     )
#     print("ρ per step:", ", ".join([f"{v:.3f}" for v in rhos]))

import os
import glob
import pickle
import argparse
import numpy as np
import pandas as pd
from scipy.stats import rankdata


DATASETS = ["audiomnist", "esc50", "msos"]
MODELS = ["alexnet", "cnn14"]
MASKS = ["pgd", "zero", "road"]

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


def spearman_rank(r1, r2):
    rank1 = rankdata(r1) * 1.0
    rank2 = rankdata(np.asarray(r2) * -1) * 1.0

    rankq1, rankq2 = rank1.copy(), rank2.copy()

    uq1, cnt1 = np.unique(rank1, return_counts=True)
    uq2, cnt2 = np.unique(rank2, return_counts=True)

    for i in range(len(cnt1)):
        if cnt1[i] > 1 and uq1[i] % 1 > 0:
            rankq1[np.where(rank1 == uq1[i])[0]] += 0.5

    for i in range(len(cnt2)):
        if cnt2[i] > 1 and uq2[i] % 1 > 0:
            rankq2[np.where(rank2 == uq2[i])[0]] += 0.5

    numer = np.corrcoef(rankq1, rankq2)[0, 1]
    denom = (rankq1.std(axis=0) * rankq2.std(axis=0)) + 1e-10
    rho_corr = numer / denom

    n = len(rankq1)
    numerq = 6 * np.sum((rankq1 - rankq2) ** 2)
    denomq = n * (n * n - 1)
    rho_diff = 1 - (numerq / denomq)

    return rankq1, rankq2, rho_diff, rho_corr


def load_curve(pkl_path, drop_first=True):
    with open(pkl_path, "rb") as f:
        arr = pickle.load(f)

    arr = np.asarray(arr, dtype=float).squeeze()

    if arr.ndim != 1:
        raise ValueError(f"{pkl_path} is not 1D, got shape {arr.shape}")

    if drop_first and len(arr) > 1:
        arr = arr[1:]

    return arr


def compute_one_folder(folder, drop_first=True):
    valid_methods = []
    morf_curves = []
    lerf_curves = []

    for method in METHOD_ORDER:
        morf_path = os.path.join(folder, f"{method}_morf.pkl")
        lerf_path = os.path.join(folder, f"{method}_lerf.pkl")

        if not (os.path.isfile(morf_path) and os.path.isfile(lerf_path)):
            continue

        m = load_curve(morf_path, drop_first=drop_first)
        l = load_curve(lerf_path, drop_first=drop_first)

        L = min(len(m), len(l))
        m = m[:L]
        l = l[:L]

        valid_methods.append(method)
        morf_curves.append(m)
        lerf_curves.append(l)

    if len(valid_methods) < 2:
        raise ValueError(f"Not enough methods in {folder}. Found: {valid_methods}")

    min_len = min(len(x) for x in morf_curves)
    alignM = np.stack([x[:min_len] for x in morf_curves], axis=0)
    alignL = np.stack([x[:min_len] for x in lerf_curves], axis=0)

    ratio_rows = []
    for r in range(min_len):
        _, _, rho_diff, rho_corr = spearman_rank(alignM[:, r], alignL[:, r])
        ratio_rows.append({
            "ratio_index": r + 1,
            "rho_diff": float(rho_diff),
            "rho_corr": float(rho_corr),
        })

    df_ratio = pd.DataFrame(ratio_rows)
    return valid_methods, df_ratio


def get_target_folders(root, dataset, model, mask):
    if dataset == "esc50":
        pattern = os.path.join(root, dataset, model, mask, "fold_*")
        folders = sorted([p for p in glob.glob(pattern) if os.path.isdir(p)])
    else:
        folder = os.path.join(root, dataset, model, mask)
        folders = [folder] if os.path.isdir(folder) else []
    return folders


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="morf_lerf")
    parser.add_argument("--drop_first", action="store_true")
    parser.add_argument("--out_summary", type=str, default="audio_spearman_summary.csv")
    parser.add_argument("--out_ratio", type=str, default="audio_spearman_by_ratio.csv")
    parser.add_argument("--out_fold", type=str, default="audio_spearman_by_fold.csv")
    args = parser.parse_args()

    summary_rows = []
    ratio_rows_all = []
    fold_rows_all = []

    for dataset in DATASETS:
        for model in MODELS:
            for mask in MASKS:
                folders = get_target_folders(args.root, model, dataset, mask)

                if len(folders) == 0:
                    print(f"[SKIP] no folder: {dataset}/{model}/{mask}")
                    continue

                fold_ratio_tables = []
                fold_names = []

                for folder in folders:
                    fold_name = os.path.basename(folder) if dataset == "esc50" else "no_fold"

                    try:
                        methods, df_ratio = compute_one_folder(folder, drop_first=args.drop_first)
                    except Exception as e:
                        print(f"[SKIP] {folder}: {e}")
                        continue

                    fold_names.append(fold_name)
                    fold_ratio_tables.append(df_ratio.copy())

                    fold_rows_all.append({
                        "dataset": dataset,
                        "model": model,
                        "mask": mask,
                        "fold": fold_name,
                        "folder": folder,
                        "n_methods": len(methods),
                        "methods": ",".join(methods),
                        "n_ratios": len(df_ratio),
                        "rho_diff_mean": float(df_ratio["rho_diff"].mean()),
                        "rho_diff_std": float(df_ratio["rho_diff"].std(ddof=0)),
                        "rho_corr_mean": float(df_ratio["rho_corr"].mean()),
                        "rho_corr_std": float(df_ratio["rho_corr"].std(ddof=0)),
                        "rho_diff_pretty": f"{df_ratio['rho_diff'].mean():.3f}±{df_ratio['rho_diff'].std(ddof=0):.3f}",
                        "rho_corr_pretty": f"{df_ratio['rho_corr'].mean():.3f}±{df_ratio['rho_corr'].std(ddof=0):.3f}",
                    })

                if len(fold_ratio_tables) == 0:
                    continue

                min_len = min(len(df) for df in fold_ratio_tables)
                rho_diff_mat = np.stack(
                    [df["rho_diff"].values[:min_len] for df in fold_ratio_tables],
                    axis=0
                )  # (n_fold, n_ratio)
                rho_corr_mat = np.stack(
                    [df["rho_corr"].values[:min_len] for df in fold_ratio_tables],
                    axis=0
                )

                # per-ratio mean/std across folds
                for r in range(min_len):
                    ratio_rows_all.append({
                        "dataset": dataset,
                        "model": model,
                        "mask": mask,
                        "ratio_index": r + 1,
                        "n_folds": rho_diff_mat.shape[0],
                        "rho_diff_mean": float(rho_diff_mat[:, r].mean()),
                        "rho_diff_std": float(rho_diff_mat[:, r].std(ddof=0)),
                        "rho_corr_mean": float(rho_corr_mat[:, r].mean()),
                        "rho_corr_std": float(rho_corr_mat[:, r].std(ddof=0)),
                    })

                # overall mean/std across all folds x all ratios
                flat_diff = rho_diff_mat.reshape(-1)
                flat_corr = rho_corr_mat.reshape(-1)

                summary_rows.append({
                    "dataset": dataset,
                    "model": model,
                    "mask": mask,
                    "n_folds": rho_diff_mat.shape[0],
                    "n_ratios": min_len,
                    "rho_diff_mean": float(flat_diff.mean()),
                    "rho_diff_std": float(flat_diff.std(ddof=0)),
                    "rho_corr_mean": float(flat_corr.mean()),
                    "rho_corr_std": float(flat_corr.std(ddof=0)),
                    "rho_diff_pretty": f"{flat_diff.mean():.3f}±{flat_diff.std(ddof=0):.3f}",
                    "rho_corr_pretty": f"{flat_corr.mean():.3f}±{flat_corr.std(ddof=0):.3f}",
                })

                print(f"[OK] {dataset}/{model}/{mask}")
                print(f"  folds     : {fold_names}")
                print(f"  rho_diff  : {flat_diff.mean():.3f} ± {flat_diff.std(ddof=0):.3f}")
                print(f"  rho_corr  : {flat_corr.mean():.3f} ± {flat_corr.std(ddof=0):.3f}")

    df_summary = pd.DataFrame(summary_rows)
    df_ratio = pd.DataFrame(ratio_rows_all)
    df_fold = pd.DataFrame(fold_rows_all)

    df_summary.to_csv(args.out_summary, index=False)
    df_ratio.to_csv(args.out_ratio, index=False)
    df_fold.to_csv(args.out_fold, index=False)

    print("\nSaved:")
    print(f"  {args.out_summary}")
    print(f"  {args.out_ratio}")
    print(f"  {args.out_fold}")

    if len(df_summary) > 0:
        print("\n=== compact table ===")
        for _, row in df_summary.iterrows():
            print(
                f"{row['dataset']:10s}  "
                f"{row['model']:11s}  "
                f"{row['mask']:5s}  "
                f"{row['rho_diff_pretty']}"
            )


if __name__ == "__main__":
    main()