"""
test_waveform_interval.py - Interval-based MoRF/LeRF for waveform datasets.

Supported datasets : AudioMNIST, ESC-50, MSoS
Supported models   : res1dnet31, audionet
Supported masking  : zero | pgd | road

Masking schedule (interval-based):
  At step k, the masked region is a contiguous interval of length L_k = round(k/n_steps * T).
  MoRF: choose the interval with the highest total saliency.
  LeRF: choose the interval with the lowest total saliency.
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)

from experiment.utils import (
    DATASET_NUM_CLASSES,
    default_ckpt,
    default_data_dir,
    default_expl_dir,
    default_split_txt,
    device,
    eval_epoch,
    get_default_eps,
    load_test_set,
    pgd_attack_waveform,
    road_generate_segment,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# Interval selection
# ============================================================

def best_interval_by_sum(s: np.ndarray, L: int, mode: str) -> tuple[int, int]:
    """
    Find the contiguous interval of length L that maximises (MoRF) or
    minimises (LeRF) the sum of saliency values.

    Args:
        s:    Saliency array for a single sample, shape (T,).
        L:    Interval length.
        mode: "morf" (max sum) or "lerf" (min sum).

    Returns:
        (t0, t1): Half-open interval [t0, t1).
    """
    T = s.shape[0]
    if L <= 0:
        return (0, 0)
    if L >= T:
        return (0, T)

    pref = np.empty(T + 1, dtype=np.float64)
    pref[0] = 0.0
    np.cumsum(s, out=pref[1:])
    window_sums = pref[L:] - pref[:-L]  # shape: (T - L + 1,)

    t0 = int(np.argmax(window_sums) if mode == "morf" else np.argmin(window_sums))
    return (t0, t0 + L)


# ============================================================
# ROAD interval application
# ============================================================

def road_apply_interval(
    x_work: np.ndarray,
    x_orig: np.ndarray,
    t0: int,
    t1: int,
    seg_cache: dict,
    rng: np.random.Generator,
) -> None:
    """
    Replace a contiguous interval x_work[t0:t1] with an in-distribution segment.

    Args:
        x_work:    Waveform to modify in-place, shape (T,).
        x_orig:    Original reference waveform, shape (T,).
        t0, t1:    Half-open interval [t0, t1) to replace.
        seg_cache: Persistent cache keyed by (t0, t1) -> replacement segment.
        rng:       NumPy random generator for reproducibility.
    """
    if t1 - t0 < 4:
        return
    key = (int(t0), int(t1))
    if key not in seg_cache:
        seg_cache[key] = road_generate_segment(x_orig, t0, t1, rng)

    rep = seg_cache[key]
    if rep is None:
        # Fallback: linear interpolation through three anchor points
        from experiment.utils import _linear_fill_three_points
        mid = (t0 + t1) // 2
        x_work[t0:t1] = _linear_fill_three_points(
            float(x_orig[t0]), float(x_orig[mid]), float(x_orig[t1 - 1]), t1 - t0
        )
    else:
        x_work[t0:t1] = rep


# ============================================================
# Interval MoRF / LeRF runner
# ============================================================

def run_morf_lerf_interval(
    model,
    X: np.ndarray,
    y: np.ndarray,
    saliency: np.ndarray,
    n_steps: int,
    batch_size: int,
    model_name: str,
    mode: str,
    mask_type: str,
    X_adv: Optional[np.ndarray] = None,
    road_seed: int = 0,
) -> np.ndarray:
    """
    Run interval-based MoRF or LeRF faithfulness evaluation.

    Args:
        model:      Trained classifier.
        X:          Waveforms of shape (N, T).
        y:          Labels of shape (N,).
        saliency:   Attribution scores of shape (N, T).
        n_steps:    Number of masking steps.
        batch_size: Inference batch size.
        model_name: One of "res1dnet31", "audionet".
        mode:       "morf" or "lerf".
        mask_type:  "zero", "pgd", or "road".
        X_adv:      Adversarial waveforms for PGD masking, shape (N, T).
        road_seed:  Random seed for ROAD segment generation.

    Returns:
        acc_curve: Accuracy at each masking step, shape (n_steps + 1,).
    """
    N, T = X.shape
    acc_curve = []
    road_cache_per_sample = [dict() for _ in range(N)]
    rng = np.random.default_rng(road_seed)
    L_list = [max(0, min(int(round(k * T / n_steps)), T)) for k in range(n_steps + 1)]

    for k, L in enumerate(L_list):
        x = X.copy()

        if L > 0:
            for i in range(N):
                t0, t1 = best_interval_by_sum(saliency[i], L, mode=mode)
                if t1 <= t0:
                    continue

                if mask_type == "zero":
                    x[i, t0:t1] = 0.0
                elif mask_type == "pgd":
                    if X_adv is None:
                        raise ValueError("X_adv is required for mask_type='pgd'")
                    x[i, t0:t1] = X_adv[i, t0:t1]
                elif mask_type == "road":
                    road_apply_interval(
                        x_work=x[i], x_orig=X[i],
                        t0=t0, t1=t1,
                        seg_cache=road_cache_per_sample[i],
                        rng=rng,
                    )
                else:
                    raise ValueError(f"Unknown mask_type: {mask_type!r}")

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
        logger.info(
            "[%s-INTERVAL] step %02d/%d | masked=%5.1f%% | L=%5d/%d | acc=%.4f",
            mode.upper(), k, n_steps, 100 * k / n_steps, L, T, acc,
        )

    return np.array(acc_curve, dtype=np.float32)


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interval-based MoRF/LeRF faithfulness evaluation on waveform datasets."
    )
    parser.add_argument("--dataset",    choices=["audiomnist", "esc50", "msos"], default="audiomnist")
    parser.add_argument("--model",      choices=["res1dnet31", "audionet"],       default="audionet")
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--fold",       type=int,  default=1, help="ESC-50 fold index.")
    parser.add_argument("--data_dir",   default=None)
    parser.add_argument("--split_txt",  default=None)
    parser.add_argument("--ckpt",       default=None)
    parser.add_argument("--expl_dir",   default=None)
    parser.add_argument("--expl_method", default="gradient_abs")
    parser.add_argument("--mask_type",  choices=["zero", "pgd", "road"], default="pgd")
    parser.add_argument("--eps",        type=float, default=None,
                        help="PGD epsilon. Defaults to per-(model, dataset) value.")
    parser.add_argument("--pgd_steps",  type=int, default=20)
    parser.add_argument("--n_steps",    type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--road_seed",  type=int, default=0)
    parser.add_argument("--out_root",   default=str(Path(PROJECT_ROOT) / "morf_lerf_interval"))
    parser.add_argument("--cache_root", default=str(Path(PROJECT_ROOT) / "morf_lerf_interval" / "_cache"))
    args = parser.parse_args()

    # Fill in defaults
    if args.data_dir   is None: args.data_dir   = default_data_dir(args.dataset, PROJECT_ROOT)
    if args.split_txt  is None: args.split_txt  = default_split_txt(args.dataset, args.data_dir, args.fold)
    if args.ckpt       is None: args.ckpt       = default_ckpt(args.dataset, args.model, args.fold, PROJECT_ROOT)
    if args.expl_dir   is None: args.expl_dir   = default_expl_dir(args.dataset, args.model, args.fold, PROJECT_ROOT)
    if args.num_classes is None: args.num_classes = DATASET_NUM_CLASSES[args.dataset]
    if args.mask_type == "pgd" and args.eps is None:
        args.eps = get_default_eps(args.model, args.dataset)
        logger.info("Using default PGD eps for (%s, %s): %.4f", args.model, args.dataset, args.eps)

    # Data
    X_wave, y = load_test_set(args.dataset, args.split_txt, args.data_dir)
    logger.info("Dataset=%s  X=%s  y=%s", args.dataset, X_wave.shape, y.shape)

    # Model
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

    logger.info("Loading checkpoint: %s", args.ckpt)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt.get("state_dict", ckpt))
    model.eval()

    # Saliency
    expl_path = Path(args.expl_dir) / f"{args.expl_method}.npy"
    logger.info("Loading saliency: %s", expl_path)
    saliency = np.load(expl_path).squeeze()
    saliency = saliency[None, :] if saliency.ndim == 1 else saliency.reshape(saliency.shape[0], -1)
    logger.info("Saliency shape: %s", saliency.shape)

    if saliency.shape[0] != X_wave.shape[0]:
        raise ValueError(f"Saliency/data N mismatch: {saliency.shape[0]} vs {X_wave.shape[0]}")
    if saliency.shape[1] != X_wave.shape[1]:
        raise ValueError(f"Saliency/data T mismatch: {saliency.shape[1]} vs {X_wave.shape[1]}")

    # PGD (cached)
    X_adv = None
    if args.mask_type == "pgd":
        cache_root = Path(args.cache_root) / args.dataset / args.model
        fold_tag = f"fold_{args.fold}" if args.dataset == "esc50" else "nofold"
        pgd_dir = cache_root / "pgd" / fold_tag
        pgd_dir.mkdir(parents=True, exist_ok=True)
        pgd_path = pgd_dir / f"pgd_eps{args.eps}_steps{args.pgd_steps}.npy"

        if pgd_path.exists():
            logger.info("Loading cached PGD from %s", pgd_path)
            X_adv = np.load(pgd_path)
        else:
            logger.info("Running PGD (eps=%.4f, steps=%d)...", args.eps, args.pgd_steps)
            X_adv = pgd_attack_waveform(
                model=model, X=X_wave, y=y, model_name=args.model,
                eps=args.eps, steps=args.pgd_steps, batch_size=args.batch_size,
            )
            np.save(pgd_path, X_adv)
            logger.info("Saved PGD cache to %s", pgd_path)

    # MoRF and LeRF
    shared_kwargs = dict(
        model=model, X=X_wave, y=y, saliency=saliency,
        n_steps=args.n_steps, batch_size=args.batch_size,
        model_name=args.model, mask_type=args.mask_type,
        X_adv=X_adv, road_seed=args.road_seed,
    )
    morf = run_morf_lerf_interval(**shared_kwargs, mode="morf")
    lerf = run_morf_lerf_interval(**shared_kwargs, mode="lerf")

    # Save
    out_dir = Path(args.out_root) / args.dataset / args.model / args.mask_type
    if args.dataset == "esc50":
        out_dir /= f"fold_{args.fold}"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / f"{args.expl_method}_morf.pkl", "wb") as f:
        pickle.dump(morf, f)
    with open(out_dir / f"{args.expl_method}_lerf.pkl", "wb") as f:
        pickle.dump(lerf, f)
    logger.info("Results saved to: %s", out_dir)


if __name__ == "__main__":
    main()
