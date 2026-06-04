import os
import pickle
import numpy as np
import pandas as pd

# =========================
# Config (IMAGE)
# =========================
DATASETS = ["brain_mri", "imagenet", "oxford_pet"]
MODELS = ["efficientnet_b0", "repvgg_b0", "resnet_50"]
MASKS = ["zero", "pgd", "road"]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
BASE = os.path.join(PROJECT_ROOT, "morf_lerf_image")


MASK_RENAME = {
    "pgd": "mdAR",
    "road": "mdROAD",
    "zero": "Zeroing",
}

METHOD_ORDER = [
    "gradients",
    "smoothgrad",
    "integrad",
    "gradinput",
    "gradcam",
    "gradcampp",
    "scorecam",
    "smoothgradcampp",
    "random",
]

METHOD_RENAME = {
    "gradients": "GD",
    "smoothgrad": "SG",
    "integrad": "IG",
    "gradinput": "GI",
    "gradcam": "GC",
    "gradcampp": "GC++",
    "scorecam": "SC",
    "smoothgradcampp": "SGC++",
    "random": "RD",
}

MASK_ORDER = ["mdROAD", "mdAR", "Zeroing"]
METRIC_ORDER = ["AOC", "ABC", "AUC"]


# =========================
# Utils
# =========================
def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def extract_curve(obj):
    if isinstance(obj, dict):
        for k in ["acc", "accuracy", "acc_curve", "curve", "accs", "acc_list"]:
            if k in obj:
                obj = obj[k]
                break
        else:
            for v in obj.values():
                if isinstance(v, (list, tuple, np.ndarray)):
                    obj = v
                    break

    arr = np.asarray(obj, dtype=float)
    arr = np.squeeze(arr)

    if arr.ndim > 1:
        arr = arr.mean(axis=0)

    return arr.astype(float)


def scan_methods(folder):
    pairs = {}
    for f in os.listdir(folder):
        if not f.endswith(".pkl"):
            continue

        name = f[:-4]

        if name.endswith("_morf"):
            m = name[:-5]
            pairs.setdefault(m, {})["morf"] = os.path.join(folder, f)

        elif name.endswith("_lerf"):
            m = name[:-5]
            pairs.setdefault(m, {})["lerf"] = os.path.join(folder, f)

    return {
        m: (v["morf"], v["lerf"])
        for m, v in pairs.items()
        if "morf" in v and "lerf" in v
    }


def format_mean_std(mean_val, std_val):
    if pd.isna(mean_val):
        return ""
    return f"{mean_val:.3f}±{std_val:.3f}"


# =========================
# Area metrics
# =========================
def compute_area(morf, lerf):
    morf = np.asarray(morf, dtype=float)
    lerf = np.asarray(lerf, dtype=float)

    if len(morf) != len(lerf):
        L = min(len(morf), len(lerf))
        morf = morf[:L]
        lerf = lerf[:L]

    acc0 = morf[0]
    accL_K = lerf[-1]
    accM_K = morf[-1]

    denom_L = acc0 - accL_K
    denom_M = acc0 - accM_K

    if np.isclose(denom_L, 0.0) or np.isclose(denom_M, 0.0):
        return np.nan, np.nan, np.nan

    K = len(morf) - 1

    aoc = 0.0
    abc = 0.0
    auc = 0.0

    for k in range(K + 1):
        aoc += min((acc0 - morf[k]) / denom_M, 1.0)
        abc += min((lerf[k] - morf[k]) / denom_M, 1.0)
        auc += min((lerf[k] - accL_K) / denom_L, 1.0)

    return aoc / (K + 1), abc / (K + 1), auc / (K + 1)


# =========================
# MAIN
# =========================
results = []

for dataset in DATASETS:
    for model in MODELS:
        for mask in MASKS:
            folder = os.path.join(BASE, model, dataset, mask)
            print(f"[CHECK] {folder}")

            if not os.path.isdir(folder):
                print(f"[SKIP] folder not found: {folder}")
                continue

            pairs = scan_methods(folder)
            print(f"[INFO] found {len(pairs)} method pairs in {folder}")

            for method, (pm, pl) in pairs.items():
                morf = extract_curve(load_pkl(pm))
                lerf = extract_curve(load_pkl(pl))

                aoc, abc, auc = compute_area(morf, lerf)

                results.append({
                    "dataset": dataset,
                    "model": model,
                    "domain": "image",
                    "mask": mask,
                    "mask_show": MASK_RENAME.get(mask, mask),
                    "method": method,
                    "method_show": METHOD_RENAME.get(method, method),
                    "AOC": aoc,
                    "ABC": abc,
                    "AUC": auc,
                })

df = pd.DataFrame(results)


print("\n=== RAW RESULT ===")
print(df.to_string(index=False) if len(df) > 0 else "Empty DataFrame")

if df.empty:
    print("\n[ERROR] No results found.")
    print(f"BASE = {BASE}")
    raise SystemExit(1)

# =========================
# Summary: mean ± std over dataset+model
# =========================
summary = (
    df.groupby(["mask_show", "method", "method_show"], as_index=False)
      .agg(
          AOC_mean=("AOC", "mean"),
          AOC_std=("AOC", "std"),
          ABC_mean=("ABC", "mean"),
          ABC_std=("ABC", "std"),
          AUC_mean=("AUC", "mean"),
          AUC_std=("AUC", "std"),
      )
)

summary["method"] = pd.Categorical(summary["method"], categories=METHOD_ORDER, ordered=True)
summary["mask_show"] = pd.Categorical(summary["mask_show"], categories=MASK_ORDER, ordered=True)
summary = summary.sort_values(["method", "mask_show"]).reset_index(drop=True)


print("\n=== SUMMARY (mean ± std) ===")
print(summary.to_string(index=False))

# =========================
# Compact table
# =========================
rows = []
for _, r in summary.iterrows():
    rows.append({
        "method": r["method_show"],
        "mask": r["mask_show"],
        "AOC": format_mean_std(r["AOC_mean"], r["AOC_std"]),
        "ABC": format_mean_std(r["ABC_mean"], r["ABC_std"]),
        "AUC": format_mean_std(r["AUC_mean"], r["AUC_std"]),
    })

compact = pd.DataFrame(rows)

pivot = compact.pivot_table(
    index="method",
    columns="mask",
    values=["AOC", "ABC", "AUC"],
    aggfunc="first"
)

wanted_cols = []
for mask in MASK_ORDER:
    for metric in METRIC_ORDER:
        col = (metric, mask)
        if col in pivot.columns:
            wanted_cols.append(col)

pivot = pivot[wanted_cols]
pivot = pivot.swaplevel(0, 1, axis=1)
pivot = pivot.sort_index(axis=1, level=0)

print("\n=== IMAGE TABLE ===")
print(pivot.to_string())







# =========================
# Per-dataset summary tables
# =========================
print("\n==============================")
print("=== PER DATASET TABLES ===")
print("==============================")

for dataset in DATASETS:
    print(f"\n===== DATASET: {dataset} =====")

    df_ds = df[df["dataset"] == dataset]

    if df_ds.empty:
        print("[SKIP] No data")
        continue

    summary_ds = (
        df_ds.groupby(["mask_show", "method", "method_show"], as_index=False)
        .agg(
            AOC_mean=("AOC", "mean"),
            AOC_std=("AOC", "std"),
            ABC_mean=("ABC", "mean"),
            ABC_std=("ABC", "std"),
            AUC_mean=("AUC", "mean"),
            AUC_std=("AUC", "std"),
        )
    )

    # 排序
    summary_ds["method"] = pd.Categorical(summary_ds["method"], categories=METHOD_ORDER, ordered=True)
    summary_ds["mask_show"] = pd.Categorical(summary_ds["mask_show"], categories=MASK_ORDER, ordered=True)
    summary_ds = summary_ds.sort_values(["method", "mask_show"]).reset_index(drop=True)

    # compact format
    rows = []
    for _, r in summary_ds.iterrows():
        rows.append({
            "method": r["method_show"],
            "mask": r["mask_show"],
            "AOC": format_mean_std(r["AOC_mean"], r["AOC_std"]),
            "ABC": format_mean_std(r["ABC_mean"], r["ABC_std"]),
            "AUC": format_mean_std(r["AUC_mean"], r["AUC_std"]),
        })

    compact_ds = pd.DataFrame(rows)

    pivot_ds = compact_ds.pivot_table(
        index="method",
        columns="mask",
        values=["AOC", "ABC", "AUC"],
        aggfunc="first"
    )

    # column order
    wanted_cols = []
    for mask in MASK_ORDER:
        for metric in METRIC_ORDER:
            col = (metric, mask)
            if col in pivot_ds.columns:
                wanted_cols.append(col)

    pivot_ds = pivot_ds[wanted_cols]
    pivot_ds = pivot_ds.swaplevel(0, 1, axis=1)
    pivot_ds = pivot_ds.sort_index(axis=1, level=0)

    print(pivot_ds.to_string())