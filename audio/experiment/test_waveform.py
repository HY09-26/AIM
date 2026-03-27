
# Unified rank-based (non-interval) MoRF/LeRF for waveform datasets:
# - AudioMNIST / ESC50 / MSOS
# - model: res1dnet31 | audionet
# - mask_type: zero | pgd | road
#
# Unified spec:
#   * mask schedule: per-point cumulative (m_list)
#   * MoRF: mask top-m points
#   * LeRF: mask bottom-m points
#   * zero baseline: 0.0
#   * PGD: compute once on full waveform; masked points replaced by X_adv
#   * ROAD: points-only replacement; segment = shortest cover; cache across k
#
# Notes:
# - AudioMNIST uses Res1dNet31Lite when model=res1dnet31
# - ESC50/MSOS uses Res1dNet31 (full) when model=res1dnet31
# - AudioNet expects (B,1,T) -> we adapt input
#
import os
import sys
import argparse
import pickle
from pathlib import Path

import numpy as np
import h5py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ============================================================
# Path
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Dataset loaders (keep original dataset-specific logic)
# ============================================================
def load_h5_wave_audiomnist(path: str):
    with h5py.File(path, "r") as f:
        if "data" in f:
            x = f["data"][:]
        elif "waveform" in f:
            x = f["waveform"][:]
        elif "x" in f:
            x = f["x"][:]
        else:
            raise KeyError(f"Cannot find waveform key in {path}. Keys={list(f.keys())}")

        if "label" in f:
            lab = f["label"][:]
            y = int(lab[0][0]) if np.asarray(lab).ndim >= 2 else int(lab[0])
        elif "y" in f:
            lab = f["y"][:]
            y = int(lab[0][0]) if np.asarray(lab).ndim >= 2 else int(lab[0])
        else:
            raise KeyError(f"Cannot find label key in {path}. Keys={list(f.keys())}")

    x = np.asarray(x, np.float32).reshape(-1)
    return x, y


def load_test_set_audiomnist(split_txt: str, data_dir: str):
    xs, ys = [], []
    with open(split_txt, "r") as f:
        for line in f:
            p = line.strip()
            if not p or p.startswith("#"):
                continue
            full = os.path.join(data_dir, p)
            if not os.path.exists(full):
                print(f"[WARN] missing file: {full}")
                continue
            x, y = load_h5_wave_audiomnist(full)
            xs.append(x)
            ys.append(y)

    if len(xs) == 0:
        raise ValueError(f"No files loaded. Check split_txt={split_txt} and data_dir={data_dir}")

    return np.stack(xs, 0).astype(np.float32), np.array(ys, np.int64)


def load_h5_wave_esc50(path: str):
    with h5py.File(path, "r") as f:
        x = f["data"][:]          # (1,1,1,T) in your pipeline
        y = int(f["label"][0][0])
    x = np.asarray(x, np.float32).reshape(-1)
    return x, y


def load_test_set_esc50(split_txt: str, data_dir: str):
    xs, ys = [], []
    with open(split_txt, "r") as f:
        for line in f:
            p = line.strip()
            if not p or "waveform" not in p:
                continue
            full = os.path.join(data_dir, p)
            x, y = load_h5_wave_esc50(full)
            xs.append(x)
            ys.append(y)
    return np.stack(xs, 0).astype(np.float32), np.array(ys, np.int64)


def load_h5_wave_msos(path: str):
    with h5py.File(path, "r") as f:
        x = f["data"][:]          # could be (1,1,1,T)
        y = int(f["label"][0][0])
    x = np.asarray(x, np.float32).reshape(-1)
    return x, y


def load_test_set_msos(split_txt: str, data_dir: str):
    xs, ys = [], []
    with open(split_txt, "r") as f:
        for line in f:
            p = line.strip()
            if not p or "waveform" not in p:
                continue
            full = os.path.join(data_dir, p)
            x, y = load_h5_wave_msos(full)
            xs.append(x)
            ys.append(y)
    return np.stack(xs, 0).astype(np.float32), np.array(ys, np.int64)


def load_test_set(dataset: str, split_txt: str, data_dir: str):
    if dataset == "audiomnist":
        return load_test_set_audiomnist(split_txt, data_dir)
    if dataset == "esc50":
        return load_test_set_esc50(split_txt, data_dir)
    if dataset == "msos":
        return load_test_set_msos(split_txt, data_dir)
    raise ValueError(dataset)


# ============================================================
# Model input adapter
# ============================================================
def adapt_wave_input(x: torch.Tensor, model_name: str) -> torch.Tensor:
    """
    x: (B,T)
    - res1dnet31*: keep (B,T)
    - audionet: use (B,1,T)
    """
    if model_name == "audionet" and x.ndim == 2:
        x = x.unsqueeze(1)
    return x


# ============================================================
# Evaluation
# ============================================================
@torch.no_grad()
def eval_epoch(model, loader, model_name: str):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x = adapt_wave_input(x, model_name)
        out = model(x)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / max(1, total)


# ============================================================
# PGD (compute ONCE on full waveform, batch-wise)
# ============================================================
def pgd_attack_waveform(
    model,
    X: np.ndarray,          # (N,T)
    y: np.ndarray,          # (N,)
    model_name: str,
    eps=0.03,
    steps=20,
    batch_size=32,
):
    model.eval()
    X_adv = X.copy()
    step = eps / max(1, steps)
    loss_fn = nn.CrossEntropyLoss()

    for i in range(0, len(X), batch_size):
        xb = torch.tensor(X_adv[i:i + batch_size], device=device, requires_grad=True)
        yb = torch.tensor(y[i:i + batch_size], device=device)
        x0 = torch.tensor(X[i:i + batch_size], device=device)

        for _ in range(steps):
            out = model(adapt_wave_input(xb, model_name))
            loss = loss_fn(out, yb)
            loss.backward()

            xb = xb + step * xb.grad.sign()
            xb = torch.max(torch.min(xb, x0 + eps), x0 - eps)
            xb = xb.detach().requires_grad_(True)

        X_adv[i:i + batch_size] = xb.detach().cpu().numpy()

    return X_adv.astype(np.float32)


# ============================================================
# Orders + prefix/suffix min/max (fast segment bounds)
# ============================================================
def compute_orders_desc(saliency: np.ndarray, cache_path: Path | None):
    """
    orders_desc[i] = argsort(-saliency[i])  (most important first)
    shape: (N,T)
    """
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            print(f"[ORDER] Loading cached orders from {cache_path}")
            return np.load(cache_path)

    print("[ORDER] Computing per-sample ranking (desc)...")
    orders_desc = np.argsort(-saliency, axis=1).astype(np.int32)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, orders_desc)
        print(f"[ORDER] Saved to {cache_path}")

    return orders_desc


def prefix_minmax(orders_desc: np.ndarray):
    """
    For top-m points:
      pref_min[i,m-1] = min(orders_desc[i,:m])
      pref_max[i,m-1] = max(orders_desc[i,:m])
    """
    pref_min = np.minimum.accumulate(orders_desc, axis=1).astype(np.int32)
    pref_max = np.maximum.accumulate(orders_desc, axis=1).astype(np.int32)
    return pref_min, pref_max


def suffix_minmax(orders_desc: np.ndarray):
    """
    For bottom-m points (tail of orders_desc):
      suff_min[i, T-m] = min(orders_desc[i, T-m:])
      suff_max[i, T-m] = max(orders_desc[i, T-m:])
    We'll build arrays of shape (N,T) where index s is "start position".
    """
    # reverse accumulate then reverse back
    rev = orders_desc[:, ::-1]
    rev_min = np.minimum.accumulate(rev, axis=1)
    rev_max = np.maximum.accumulate(rev, axis=1)
    suff_min = rev_min[:, ::-1].astype(np.int32)
    suff_max = rev_max[:, ::-1].astype(np.int32)
    return suff_min, suff_max


# ============================================================
# ROAD: fast & robust MFBB-like segment generator (points-only replacement)
# ============================================================
def _linear_fill_three_points(x0, xmid, x1, n):
    if n <= 1:
        return np.array([x0], dtype=np.float32)
    mid = (n - 1) // 2
    out = np.empty(n, dtype=np.float32)

    # left 0..mid
    if mid == 0:
        out[0] = x0
    else:
        t = np.linspace(0.0, 1.0, mid + 1, dtype=np.float32)
        out[: mid + 1] = (1 - t) * x0 + t * xmid

    # right mid..n-1
    if n - 1 - mid == 0:
        out[-1] = x1
    else:
        t = np.linspace(0.0, 1.0, n - mid, dtype=np.float32)
        out[mid:] = (1 - t) * xmid + t * x1

    return out


def _mfbb_noise_fast(n: int, original_std: float, rng: np.random.Generator):
    """
    Very fast smooth noise:
    white -> cumsum -> cumsum, normalize, scale to 5% of original std
    """
    if n <= 1:
        return np.zeros(n, dtype=np.float32)

    w = rng.standard_normal(n).astype(np.float32)
    s = np.cumsum(w)
    s = np.cumsum(s)
    s = s - s.mean()
    sd = float(s.std() + 1e-8)
    target = float(original_std) * 0.05
    return (s / sd * target).astype(np.float32)


def road_generate_segment_robust(x_full: np.ndarray, t0: int, t1: int, rng: np.random.Generator):
    seg_len = int(t1 - t0)
    if seg_len < 4:
        return None

    orig = x_full[t0:t1].astype(np.float32, copy=False)
    mu_o = float(orig.mean())
    sd_o = float(orig.std() + 1e-8)

    mid = (t0 + t1) // 2
    base = _linear_fill_three_points(float(x_full[t0]), float(x_full[mid]), float(x_full[t1 - 1]), seg_len)
    rep = base + _mfbb_noise_fast(seg_len, sd_o, rng)

    if not np.all(np.isfinite(rep)):
        return None

    # match mean/std to original segment
    mu_r = float(rep.mean())
    sd_r = float(rep.std() + 1e-8)
    rep = (rep - mu_r) / sd_r * sd_o + mu_o

    # conservative clamp
    lo = mu_o - 8.0 * sd_o
    hi = mu_o + 8.0 * sd_o
    rep = np.clip(rep, lo, hi).astype(np.float32)
    return rep


def road_apply_points_only(
    x_work: np.ndarray,          # (T,) will be modified in-place
    x_orig: np.ndarray,          # (T,) reference
    idx_mask: np.ndarray,        # (m,)
    seg_cache: dict,             # key=(t0,t1) -> rep segment
    rng: np.random.Generator,
):
    if idx_mask.size == 0:
        return

    t0 = int(idx_mask.min())
    t1 = int(idx_mask.max()) + 1
    seg_len = t1 - t0
    if seg_len < 4:
        return

    key = (t0, t1)
    if key not in seg_cache:
        seg_cache[key] = road_generate_segment_robust(x_orig, t0, t1, rng)

    rep = seg_cache[key]
    pos = (idx_mask - t0).astype(np.int32)

    if rep is None:
        # fallback: linear baseline only
        mid = (t0 + t1) // 2
        base = _linear_fill_three_points(float(x_orig[t0]), float(x_orig[mid]), float(x_orig[t1 - 1]), seg_len)
        x_work[idx_mask] = base[pos]
    else:
        x_work[idx_mask] = rep[pos]

# ============================================================
# PGD eps defaults by (model, dataset)
# ============================================================
PGD_EPS_DEFAULT = {
    ("audionet", "audiomnist"): 0.045,
    ("audionet", "esc50"):      0.07,
    ("audionet", "msos"):       0.0025,
    ("res1dnet31", "audiomnist"): 0.065,
    ("res1dnet31", "esc50"):      0.02,
    ("res1dnet31", "msos"):       0.0015,
}

def get_default_eps(model_name: str, dataset: str) -> float:
    key = (model_name, dataset)
    if key not in PGD_EPS_DEFAULT:
        raise KeyError(f"No default eps for model={model_name}, dataset={dataset}")
    return float(PGD_EPS_DEFAULT[key])

# ============================================================
# MoRF / LeRF (rank-based, per-point cumulative)
# ============================================================
def run_morf_lerf_ranked(
    model,
    X: np.ndarray,              # (N,T)
    y: np.ndarray,              # (N,)
    orders_desc: np.ndarray,    # (N,T)
    pref_min: np.ndarray,       # (N,T)
    pref_max: np.ndarray,       # (N,T)
    suff_min: np.ndarray,       # (N,T)
    suff_max: np.ndarray,       # (N,T)
    n_steps: int,
    batch_size: int,
    model_name: str,
    mode: str,                  # "morf" | "lerf"
    mask_type: str,             # "zero" | "pgd" | "road"
    X_adv: np.ndarray | None = None,
    road_seed: int = 0,
):
    N, T = X.shape
    acc_curve = []

    # global ROAD cache across k:
    # cache per sample to avoid mixing segments across samples
    road_cache_per_sample = [dict() for _ in range(N)]
    rng = np.random.default_rng(road_seed)

    # per-point cumulative schedule
    m_list = [int(round(k * (T / max(1, n_steps)))) for k in range(n_steps + 1)]
    m_list = [max(0, min(m, T)) for m in m_list]

    for k, m in enumerate(m_list):
        x = X.copy()

        if m > 0:
            if mode == "morf":
                # top-m points
                for i in range(N):
                    idx_mask = orders_desc[i, :m]

                    if mask_type == "zero":
                        x[i, idx_mask] = 0.0

                    elif mask_type == "pgd":
                        if X_adv is None:
                            raise ValueError("X_adv is required for mask_type='pgd'")
                        x[i, idx_mask] = X_adv[i, idx_mask]

                    elif mask_type == "road":
                        # shortest cover segment via prefix min/max at m-1
                        t0 = int(pref_min[i, m - 1])
                        t1 = int(pref_max[i, m - 1]) + 1
                        # we still replace only idx_mask points, but segment is for generation
                        # caching by (t0,t1) is already done inside seg_cache
                        _ = (t0, t1)
                        road_apply_points_only(
                            x_work=x[i],
                            x_orig=X[i],
                            idx_mask=idx_mask,
                            seg_cache=road_cache_per_sample[i],
                            rng=rng,
                        )
                    else:
                        raise ValueError(mask_type)

            elif mode == "lerf":
                # bottom-m points
                start = T - m
                for i in range(N):
                    idx_mask = orders_desc[i, start:]

                    if mask_type == "zero":
                        x[i, idx_mask] = 0.0

                    elif mask_type == "pgd":
                        if X_adv is None:
                            raise ValueError("X_adv is required for mask_type='pgd'")
                        x[i, idx_mask] = X_adv[i, idx_mask]

                    elif mask_type == "road":
                        # shortest cover segment via suffix min/max at start
                        t0 = int(suff_min[i, start])
                        t1 = int(suff_max[i, start]) + 1
                        _ = (t0, t1)
                        road_apply_points_only(
                            x_work=x[i],
                            x_orig=X[i],
                            idx_mask=idx_mask,
                            seg_cache=road_cache_per_sample[i],
                            rng=rng,
                        )
                    else:
                        raise ValueError(mask_type)
            else:
                raise ValueError(mode)

        loader = DataLoader(
            TensorDataset(
                torch.tensor(x, dtype=torch.float32),
                torch.tensor(y, dtype=torch.long),
            ),
            batch_size=batch_size,
            shuffle=False,
        )
        acc = eval_epoch(model, loader, model_name=model_name)
        acc_curve.append(acc)

        print(
            f"[{mode.upper()}-RANK] step {k:02d}/{n_steps} "
            f"| masked={100*k/n_steps:5.1f}% | points={m:5d}/{T} | acc={acc:.4f}"
        )

    return np.array(acc_curve, dtype=np.float32)


# ============================================================
# Helpers: default paths (match your original patterns)
# ============================================================
def default_data_dir(dataset: str):
    if dataset == "audiomnist":
        return str(Path(PROJECT_ROOT) / "experiment_utils" / "data" / "audiomnist" / "preprocessed_data")
    if dataset == "esc50":
        return str(Path(PROJECT_ROOT) / "experiment_utils" / "data" / "ESC50" / "preprocessed_data")
    if dataset == "msos":
        return str(Path(PROJECT_ROOT) / "experiment_utils" / "data" / "MSoS" / "preprocessed_data")
    raise ValueError(dataset)


def default_split_txt(dataset: str, data_dir: str, fold: int):
    if dataset == "audiomnist":
        return str(Path(data_dir) / "AudioNet_digit_0_test.txt")
    if dataset == "esc50":
        return str(Path(data_dir) / f"ESC50_fold{fold}_test.txt")
    if dataset == "msos":
        return str(Path(data_dir) / "MSOS_test.txt")
    raise ValueError(dataset)


def default_ckpt(dataset: str, model: str, fold: int):
    ckpt_root = Path(PROJECT_ROOT) / "experiment_utils" / "checkpoints"
    if dataset == "audiomnist":
        return str(ckpt_root / "audiomnist" / model / "best_model.pth")
    if dataset == "esc50":
        return str(ckpt_root / "esc50" / model / f"fold_{fold}" / "best_model.pth")
    if dataset == "msos":
        return str(ckpt_root / "msos" / model / "best_model.pth")
    raise ValueError(dataset)


def default_expl_dir(dataset: str, model: str, fold: int):
    if dataset == "audiomnist":
        return str(Path(PROJECT_ROOT) / "expl_audiomnist" / model)
    if dataset == "esc50":
        return str(Path(PROJECT_ROOT) / "expl_esc50" / model / f"fold_{fold}")
    if dataset == "msos":
        return str(Path(PROJECT_ROOT) / "expl_msos" / model)
    raise ValueError(dataset)


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", choices=["audiomnist", "esc50", "msos"], default="audiomnist")
    parser.add_argument("--model", choices=["res1dnet31", "audionet"], default="audionet")
    parser.add_argument("--num_classes", type=int, default=None)

    parser.add_argument("--fold", type=int, default=1, help="ESC50 fold only")

    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--split_txt", default=None)

    parser.add_argument("--ckpt", default=None)

    parser.add_argument("--expl_dir", default=None)
    parser.add_argument("--expl_method", default="gradient_abs")

    parser.add_argument("--mask_type", choices=["zero", "pgd", "road"], default="pgd")
    parser.add_argument("--eps", type=float, default=None, help="PGD epsilon. If omitted, use default by (model,dataset).")
    parser.add_argument("--pgd_steps", type=int, default=20)

    parser.add_argument("--n_steps", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)

    parser.add_argument("--road_seed", type=int, default=0)

    parser.add_argument("--out_root", default=str(Path(PROJECT_ROOT) / "morf_lerf"))
    parser.add_argument("--cache_root", default=str(Path(PROJECT_ROOT) / "morf_lerf" / "_cache"))

    args = parser.parse_args()



    # ---------- defaults ----------
    if args.data_dir is None:
        args.data_dir = default_data_dir(args.dataset)
    if args.split_txt is None:
        args.split_txt = default_split_txt(args.dataset, args.data_dir, args.fold)
    if args.ckpt is None:
        args.ckpt = default_ckpt(args.dataset, args.model, args.fold)
    if args.expl_dir is None:
        args.expl_dir = default_expl_dir(args.dataset, args.model, args.fold)
    if args.mask_type == "pgd" and args.eps is None:
        args.eps = get_default_eps(args.model, args.dataset)
        print(f"[PGD] Using default eps for ({args.model} x {args.dataset}) = {args.eps}")
    if args.num_classes is None:
        # sensible defaults
        if args.dataset == "audiomnist":
            args.num_classes = 10
        elif args.dataset == "esc50":
            args.num_classes = 50
        else:
            args.num_classes = 5

    # ---------- data ----------
    X_wave, y = load_test_set(args.dataset, args.split_txt, args.data_dir)
    print(f"[DATA] dataset={args.dataset} X_wave={X_wave.shape} y={y.shape}")

    # ---------- model imports + build ----------
    if args.model == "res1dnet31":
        if args.dataset == "audiomnist":
            from experiment_utils.model.res1dnet31 import Res1dNet31Lite
            model = Res1dNet31Lite(num_classes=args.num_classes).to(device)
        else:
            from experiment_utils.model.res1dnet31 import Res1dNet31
            model = Res1dNet31(num_classes=args.num_classes).to(device)
    else:
        from experiment_utils.model.audionet import AudioNet
        model = AudioNet(num_classes=args.num_classes).to(device)

    print("[MODEL] loading ckpt:", args.ckpt)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt)
    model.eval()

    # ---------- saliency ----------
    expl_path = Path(args.expl_dir) / f"{args.expl_method}.npy"
    print("[EXPL] loading:", expl_path)
    saliency = np.load(expl_path).squeeze()

    if saliency.ndim == 1:
        saliency = saliency[None, :]
    else:
        saliency = saliency.reshape(saliency.shape[0], -1)

    print("[EXPL] saliency:", saliency.shape)

    # enforce alignment
    if saliency.shape[0] != X_wave.shape[0]:
        raise ValueError(f"saliency N mismatch: saliency={saliency.shape[0]} vs X={X_wave.shape[0]}")
    if saliency.shape[1] != X_wave.shape[1]:
        raise ValueError(f"saliency T mismatch: saliency={saliency.shape[1]} vs X={X_wave.shape[1]}")

    # ---------- ranking cache ----------
    cache_root = Path(args.cache_root) / args.dataset / args.model
    orders_path = cache_root / "orders" / f"{args.expl_method}_desc.npy"
    orders_desc = compute_orders_desc(saliency, orders_path)

    pref_min, pref_max = prefix_minmax(orders_desc)
    suff_min, suff_max = suffix_minmax(orders_desc)

    # ---------- PGD cache (once) ----------
    X_adv = None
    if args.mask_type == "pgd":
        pgd_dir = cache_root / "pgd" / (f"fold_{args.fold}" if args.dataset == "esc50" else "nofold")
        pgd_dir.mkdir(parents=True, exist_ok=True)
        pgd_path = pgd_dir / f"pgd_eps{args.eps}_steps{args.pgd_steps}.npy"

        if pgd_path.exists():
            print(f"[PGD] Loading cached PGD from {pgd_path}")
            X_adv = np.load(pgd_path)
        else:
            print("[PGD] Running PGD once on FULL waveform...")
            X_adv = pgd_attack_waveform(
                model=model,
                X=X_wave,
                y=y,
                model_name=args.model,
                eps=args.eps,
                steps=args.pgd_steps,
                batch_size=args.batch_size,
            )
            np.save(pgd_path, X_adv)
            print(f"[PGD] Saved to {pgd_path}")

    # ---------- MoRF / LeRF ----------
    morf = run_morf_lerf_ranked(
        model=model,
        X=X_wave,
        y=y,
        orders_desc=orders_desc,
        pref_min=pref_min,
        pref_max=pref_max,
        suff_min=suff_min,
        suff_max=suff_max,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        model_name=args.model,
        mode="morf",
        mask_type=args.mask_type,
        X_adv=X_adv,
        road_seed=args.road_seed,
    )

    lerf = run_morf_lerf_ranked(
        model=model,
        X=X_wave,
        y=y,
        orders_desc=orders_desc,
        pref_min=pref_min,
        pref_max=pref_max,
        suff_min=suff_min,
        suff_max=suff_max,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        model_name=args.model,
        mode="lerf",
        mask_type=args.mask_type,
        X_adv=X_adv,
        road_seed=args.road_seed,
    )

    # ---------- save ----------
    out_dir = Path(args.out_root) / args.dataset / args.model / args.mask_type
    if args.dataset == "esc50":
        out_dir = out_dir / f"fold_{args.fold}"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / f"{args.expl_method}_morf.pkl", "wb") as f:
        pickle.dump(morf, f)
    with open(out_dir / f"{args.expl_method}_lerf.pkl", "wb") as f:
        pickle.dump(lerf, f)

    print("Saved to:", out_dir)


if __name__ == "__main__":
    main()