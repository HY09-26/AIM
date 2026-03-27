import os
import re
import pickle
import numpy as np

# =========================
# Path (project-root safe)
# =========================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJECT_ROOT, "experiment_AudioMNIST", "morf_lerf_audio")

# AudioMNIST
CHANCE = 0.1

# Your method naming (filename -> short name)
METHOD_MAP = {
    "gradient": "GD",
    "gradinput": "GI",
    "smoothgrad": "SG",
    "smoothgrad_sq": "SS",
    "vargrad": "VG",
    "integrad": "IG",
    "random": "RD",
}
METHOD_MAP_ABS = {
    "gradient": "GDA",
    "gradinput": "GIA",
    "smoothgrad": "SGA",
    "integrad": "IGA",
}

# -------------------------
# Helpers
# -------------------------
def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def _as_curve(obj):
    """
    Try to extract an accuracy curve from a loaded pickle object.
    Expected: (N, K+1) or (K+1,) or list-like.
    Returns: np.ndarray of shape (K+1,) in float.
    """
    # common cases
    if isinstance(obj, dict):
        # try common keys first
        for k in ["acc", "accuracy", "acc_curve", "curve", "accs", "acc_list"]:
            if k in obj:
                obj = obj[k]
                break
        else:
            # fallback: pick the first array-like value
            for v in obj.values():
                if isinstance(v, (list, tuple, np.ndarray)):
                    obj = v
                    break

    arr = np.asarray(obj)
    if arr.ndim == 0:
        raise ValueError("Cannot extract curve: scalar found.")
    if arr.ndim == 1:
        return arr.astype(float)
    if arr.ndim >= 2:
        # average across samples / batches
        return arr.mean(axis=0).astype(float)

def parse_fname(fname):
    """
    Parse names like:
      alexnet_gradient_morf.pkl
      alexnet_gradient_abs_lerf.pkl
      audionet_integrated_grad_pgd_morf.pkl
      audionet_smoothgrad_waveform_zero_lerf.pkl
    Returns dict with fields:
      model, method_raw, is_abs, masking, input, kind(morf/lerf)
    """
    name = fname.replace(".pkl", "")
    parts = name.split("_")
    if len(parts) < 3:
        return None

    model = parts[0]
    kind = parts[-1]  # morf or lerf
    if kind not in ("morf", "lerf"):
        return None

    tokens = parts[1:-1]  # middle tokens

    # detect flags
    is_abs = "abs" in tokens
    masking = "pgd" if "pgd" in tokens else ("zero" if "zero" in tokens else "unknown")
    inp = "waveform" if "waveform" in tokens else ("spectrogram" if "spectrogram" in tokens else "unknown")

    # method token(s): remove known flags
    flag_set = {"abs", "pgd", "zero", "waveform", "spectrogram"}
    method_tokens = [t for t in tokens if t not in flag_set]

    # recover multi-token method names
    method_raw = "_".join(method_tokens).lower()

    # sometimes files are like alexnet_gradient_abs_morf (method_tokens=["gradient"])
    # or alexnet_integrated_grad_morf (method_tokens=["integrated","grad"])
    if method_raw == "integrad":
        pass

    return {
        "model": model,
        "method_raw": method_raw,
        "is_abs": is_abs,
        "masking": masking,
        "input": inp,
        "kind": kind,
    }

def method_short(method_raw, is_abs):
    if is_abs:
        if method_raw in METHOD_MAP_ABS:
            return METHOD_MAP_ABS[method_raw]
        # some abs variants not defined (e.g. smoothgrad_sq_abs) -> skip safely
        return None
    else:
        return METHOD_MAP.get(method_raw, None)

def compute_metrics(curveM, curveL):
    # curves are already normalized to [chance, top_acc] in original AIM;
    # for safety, we clip again relative to unmasked mean.
    top_acc = max(curveM[0], curveL[0]) if len(curveM) > 0 else 1.0
    curveM = np.clip(curveM, CHANCE, top_acc)
    curveL = np.clip(curveL, CHANCE, top_acc)

    # standard AIM area-based metrics
    aoc = np.mean(1.0 - curveM)
    abc = np.mean(np.clip(curveL - curveM, 0, None))
    auc = np.mean(curveL)
    return aoc, abc, auc

# -------------------------
# Main scan
# -------------------------
def scan_all_pickles():
    rows = []
    for model in ["alexnet", "audionet"]:
        model_dir = os.path.join(DATA_ROOT, model)
        if not os.path.isdir(model_dir):
            continue
        for fname in os.listdir(model_dir):
            if not fname.endswith(".pkl"):
                continue
            meta = parse_fname(fname)
            if meta is None:
                continue
            meta["path"] = os.path.join(model_dir, fname)
            rows.append(meta)
    return rows

def pair_morf_lerf(rows):
    """
    Build pairs keyed by (model, input, masking, method_short) -> (morf_path, lerf_path)
    """
    pairs = {}
    for r in rows:
        ms = method_short(r["method_raw"], r["is_abs"])
        if ms is None:
            continue
        key = (r["model"], r["input"], r["masking"], ms)
        pairs.setdefault(key, {})[r["kind"]] = r["path"]

    # keep only complete pairs
    out = {}
    for k, v in pairs.items():
        if "morf" in v and "lerf" in v:
            out[k] = (v["morf"], v["lerf"])
    return out

def summarize_table(pairs):
    """
    Aggregate over all available pairs (could include multiple files per config if you have repeats).
    Output grouped by (model, input, masking)
    """
    # group -> method -> list(metrics)
    grouped = {}
    for (model, inp, masking, method), (pm, pl) in pairs.items():
        grouped.setdefault((model, inp, masking), {}).setdefault(method, []).append((pm, pl))

    for gkey in sorted(grouped.keys()):
        model, inp, masking = gkey
        print(f"\n==============================")
        print(f"{model.upper()} | {inp.upper()} | {masking.upper()}")
        print(f"==============================")
        print("Method |   AOC   |   ABC   |   AUC")

        methods_sorted = sorted(grouped[gkey].keys(), key=lambda x: x)  # alphabetical short names
        for m in methods_sorted:
            aocs, abcs, aucs = [], [], []
            for pm, pl in grouped[gkey][m]:
                morf_obj = load_pkl(pm)
                lerf_obj = load_pkl(pl)

                curveM = _as_curve(morf_obj)
                curveL = _as_curve(lerf_obj)

                # common convention: element 0 = unmasked; use [1:] as masking trajectory
                if curveM.ndim != 1 or curveL.ndim != 1:
                    raise ValueError("Curves must be 1D after extraction.")
                if len(curveM) >= 2:
                    trajM = curveM[1:]
                else:
                    trajM = curveM
                if len(curveL) >= 2:
                    trajL = curveL[1:]
                else:
                    trajL = curveL

                aoc, abc, auc = compute_metrics(trajM, trajL)
                aocs.append(aoc); abcs.append(abc); aucs.append(auc)

            # mean±std over whatever files you have for that config
            print(
                f"{m:>4} | "
                f"{np.mean(aocs):.3f}±{np.std(aocs):.3f} | "
                f"{np.mean(abcs):.3f}±{np.std(abcs):.3f} | "
                f"{np.mean(aucs):.3f}±{np.std(aucs):.3f}"
            )

if __name__ == "__main__":
    print("PROJECT_ROOT =", PROJECT_ROOT)
    print("DATA_ROOT    =", DATA_ROOT)
    if not os.path.isdir(DATA_ROOT):
        raise FileNotFoundError(f"DATA_ROOT not found: {DATA_ROOT}")

    rows = scan_all_pickles()
    if len(rows) == 0:
        raise RuntimeError(f"No .pkl files found under: {DATA_ROOT}")

    pairs = pair_morf_lerf(rows)
    if len(pairs) == 0:
        raise RuntimeError(
            "No morf/lerf pairs found. Check filenames end with _morf.pkl and _lerf.pkl."
        )

    summarize_table(pairs)
