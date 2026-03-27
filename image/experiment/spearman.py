import os
import argparse
import pickle
import numpy as np
from scipy.stats import rankdata

# =========================
# Argument parser
# =========================
parser = argparse.ArgumentParser(description="Image AIM Spearman rank consistency")
parser.add_argument("--model", type=str, default="resnet_50", choices=["resnet_50", "efficientnet_b0", "repvgg_b0"])
parser.add_argument("--dataset", type=str, default="imagenet", choices=["brain_mri", "oxford_pet", "imagenet"])
parser.add_argument("--mask_type", type=str, default="pgd", choices=["zero", "pgd", "road"],)
args = parser.parse_args()

# =========================
# Path (from arguments)
# =========================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(
    PROJECT_ROOT,
    "morf_lerf_image",
    args.model,
    args.dataset,
    args.mask_type,
)

# =========================
# Config (ONLY existing methods)
# =========================
METHOD_MAP = {
    "gradcam": "GC",
    "gradcampp": "GCPP",
    "gradients": "GD",
    "smoothgrad": "SG",
    "random": "RD",
}

METHOD_MAP_ABS = {
    "gradients": "GDA",
}

# =========================
# Utils
# =========================
def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _as_curve(obj):
    """
    Robustly extract a 1D performance curve
    """
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

    arr = np.asarray(obj)
    if arr.ndim == 1:
        return arr.astype(float)
    if arr.ndim >= 2:
        return arr.mean(axis=0).astype(float)

    raise ValueError("Cannot extract curve.")


def parse_fname(fname):
    """
    Expected filename examples:
      gradcam_morf.pkl
      gradcam_lerf.pkl
      gradients_abs_morf.pkl
      gradients_abs_lerf.pkl
    """
    name = fname.replace(".pkl", "")
    parts = name.split("_")
    if len(parts) < 2:
        return None

    kind = parts[-1]
    if kind not in ("morf", "lerf"):
        return None

    tokens = parts[:-1]
    is_abs = "abs" in tokens
    method_tokens = [t for t in tokens if t != "abs"]
    method_raw = "_".join(method_tokens).lower()

    return {
        "method_raw": method_raw,
        "is_abs": is_abs,
        "kind": kind,
    }


def method_short(method_raw, is_abs):
    if is_abs:
        return METHOD_MAP_ABS.get(method_raw, None)
    return METHOD_MAP.get(method_raw, None)


def spearman_rank(x, y):
    """
    MoRF: lower is better
    LeRF: higher is better
    """
    rx = rankdata(x)
    ry = rankdata(-y)
    n = len(rx)
    return 1 - (6 * np.sum((rx - ry) ** 2)) / (n * (n * n - 1))


# =========================
# Scan flat folder
# =========================
def scan_pairs(mask_dir):
    rows = []

    for fname in os.listdir(mask_dir):
        if not fname.endswith(".pkl"):
            continue

        meta = parse_fname(fname)
        if meta is None:
            continue

        ms = method_short(meta["method_raw"], meta["is_abs"])
        if ms is None:
            continue

        meta["method"] = ms
        meta["path"] = os.path.join(mask_dir, fname)
        rows.append(meta)

    # method -> {morf, lerf}
    tmp = {}
    for r in rows:
        tmp.setdefault(r["method"], {})[r["kind"]] = r["path"]

    pairs = {}
    for m, v in tmp.items():
        if "morf" in v and "lerf" in v:
            pairs[m] = (v["morf"], v["lerf"])

    return pairs


# =========================
# Main
# =========================
if __name__ == "__main__":

    print("PROJECT_ROOT =", PROJECT_ROOT)
    print("DATA_ROOT    =", DATA_ROOT)

    if not os.path.isdir(DATA_ROOT):
        raise FileNotFoundError(f"DATA_ROOT not found: {DATA_ROOT}")

    pairs = scan_pairs(DATA_ROOT)
    if len(pairs) == 0:
        raise RuntimeError(
            "No morf/lerf pairs found. "
            "Check filenames like *_morf.pkl and *_lerf.pkl."
        )

    methods = sorted(pairs.keys())
    print("[INFO] Methods:", methods)

    curvesM, curvesL = {}, {}
    for m in methods:
        pm, pl = pairs[m]
        curveM = _as_curve(load_pkl(pm))
        curveL = _as_curve(load_pkl(pl))

        # drop 0% masked
        curvesM[m] = curveM[1:] if len(curveM) >= 2 else curveM
        curvesL[m] = curveL[1:] if len(curveL) >= 2 else curveL

    K = min(len(curvesM[m]) for m in methods)
    if K < 2:
        raise RuntimeError("Curve too short for Spearman computation.")

    rhos = []
    for r in range(K):
        perfM = np.array([curvesM[m][r] for m in methods], dtype=float)
        perfL = np.array([curvesL[m][r] for m in methods], dtype=float)
        rhos.append(spearman_rank(perfM, perfL))

    rhos = np.asarray(rhos)
    rhos = rhos[:-1]
    print(
        f"Image | {args.dataset} | {args.mask_type} | {args.model} : "
        f"ρ = {rhos.mean():.3f} ± {rhos.std():.3f} (K={K})"
    )
    print("ρ per step:", ", ".join([f"{v:.3f}" for v in rhos]))