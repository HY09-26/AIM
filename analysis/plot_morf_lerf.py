"""Unified MoRF / LeRF accuracy-curve plots — image and audio domains.

Usage examples
--------------
# from project root:
python analysis/plot_morf_lerf.py --domain image
python analysis/plot_morf_lerf.py --domain audio
python analysis/plot_morf_lerf.py --domain audio --datasets esc50 --esc50_folds 1 2 3

# or via per-domain thin wrappers (original paths still work):
python image/experiment/plot_morf_lerf.py
python audio/experiment/plot_morf_lerf.py --esc50_folds 1 2

Note: the EEG plot.py is a saliency-map visualisation script (topomaps,
channel-time heatmaps) and is NOT equivalent to MoRF/LeRF curve plots —
it is therefore not merged here.

Output
------
For each (dataset, model, mask_type) [and ESC-50 fold], plots are saved
inside a plots/ subdirectory next to the result pickle files:
  {result_folder}/plots/{method}_morf_lerf.png   (per-method MoRF+LeRF)
  {result_folder}/plots/all_morf.png             (all methods, MoRF only)
  {result_folder}/plots/all_lerf.png             (all methods, LeRF only)
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ANALYSIS_DIR)


class _CompatUnpickler(pickle.Unpickler):
    """Remap numpy._core → numpy.core for cross-version compatibility."""
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core")
        return super().find_class(module, name)

# Attribution methods whose curves are shown with a dashed line (audio only).
# These are the non-absolute-value variants, which tend to produce weaker
# attributions than their |·| counterparts.
_NON_ABS_METHODS = {
    "gradient", "gradinput", "smoothgrad", "integrad", "vargrad",
}


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_curve(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        return np.asarray(_CompatUnpickler(f).load(), dtype=float)


def _pct_masked(n_steps: int) -> np.ndarray:
    """X-axis ticks: 0 % … 100 % for n_steps + 1 points."""
    return np.linspace(0, 100, n_steps + 1)


def _discover_methods(folder: str) -> list[str]:
    """Return sorted list of methods with both *_morf.pkl and *_lerf.pkl."""
    morf_set: set[str] = set()
    lerf_set: set[str] = set()
    for fname in os.listdir(folder):
        if fname.endswith("_morf.pkl"):
            morf_set.add(fname[: -len("_morf.pkl")])
        elif fname.endswith("_lerf.pkl"):
            lerf_set.add(fname[: -len("_lerf.pkl")])
    return sorted(morf_set & lerf_set)


# ─────────────────────────────────────────────────────────────────────────────
# Per-folder plotting
# ─────────────────────────────────────────────────────────────────────────────

def _plot_one_method(
    folder: str,
    method: str,
    plot_dir: str,
    dashed_nabs: bool,
) -> None:
    """One figure: MoRF and LeRF curves for a single attribution method."""
    mp = os.path.join(folder, f"{method}_morf.pkl")
    lp = os.path.join(folder, f"{method}_lerf.pkl")
    if not (os.path.exists(mp) and os.path.exists(lp)):
        logger.warning("Missing MoRF or LeRF for %s, skipping.", method)
        return

    morf = _load_curve(mp)
    lerf = _load_curve(lp)
    x    = _pct_masked(len(morf) - 1)
    ls   = "--" if (dashed_nabs and method in _NON_ABS_METHODS) else "-"

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(x, morf, marker="o", linestyle=ls, label="MoRF")
    ax.plot(x, lerf, marker="s", linestyle=ls, label="LeRF")
    ax.set_xlabel("% Masked")
    ax.set_ylabel("Accuracy")
    ax.set_title(method)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, f"{method}_morf_lerf.png"), dpi=200)
    plt.close(fig)


def _plot_all_methods(
    folder: str,
    methods: list[str],
    mode: str,
    plot_dir: str,
    dashed_nabs: bool,
) -> None:
    """One figure: all methods' MoRF (or LeRF) curves overlaid."""
    fig, ax = plt.subplots(figsize=(6, 5))
    plotted = False
    for method in methods:
        path = os.path.join(folder, f"{method}_{mode}.pkl")
        if not os.path.exists(path):
            continue
        curve = _load_curve(path)
        x     = _pct_masked(len(curve) - 1)
        ls    = "--" if (dashed_nabs and method in _NON_ABS_METHODS) else "-"
        ax.plot(x, curve, linewidth=2, linestyle=ls, label=method)
        plotted = True

    if not plotted:
        logger.warning("No curves for %s, skipping combined plot.", mode.upper())
        plt.close(fig)
        return

    ax.set_xlabel("% Masked")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{mode.upper()} – All Methods")
    ax.legend(fontsize=8)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, f"all_{mode}.png"), dpi=200)
    plt.close(fig)


def _process_folder(folder: str, dashed_nabs: bool = False) -> None:
    """Discover methods and generate all plots for one result folder."""
    methods = _discover_methods(folder)
    if not methods:
        logger.warning("No MoRF/LeRF pairs in %s, skipping.", folder)
        return

    plot_dir = os.path.join(folder, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    logger.info("Found %d methods: %s", len(methods), methods)
    logger.info("Saving plots to: %s", plot_dir)

    for method in methods:
        _plot_one_method(folder, method, plot_dir, dashed_nabs=dashed_nabs)
    _plot_all_methods(folder, methods, mode="morf", plot_dir=plot_dir, dashed_nabs=dashed_nabs)
    _plot_all_methods(folder, methods, mode="lerf", plot_dir=plot_dir, dashed_nabs=dashed_nabs)


# ─────────────────────────────────────────────────────────────────────────────
# Domain runners
# ─────────────────────────────────────────────────────────────────────────────

_IMG_DATASETS = ["brain_mri", "imagenet", "oxford_pet"]
_IMG_MODELS   = ["resnet_50", "efficientnet_b0", "repvgg_b0"]
_IMG_MASKS    = ["zero", "pgd", "road"]

_AUD_DATASETS = ["audiomnist", "esc50", "msos"]
_AUD_MODELS   = ["audionet", "res1dnet31", "alexnet", "cnn14"]
_AUD_MASKS    = ["zero", "pgd", "road"]


def _run_image(args: argparse.Namespace) -> None:
    root     = args.root or os.path.join(PROJECT_ROOT, "image", "morf_lerf_image")
    datasets = args.datasets if args.datasets else _IMG_DATASETS
    models   = args.models   if args.models   else _IMG_MODELS
    masks    = args.masks    if args.masks    else _IMG_MASKS

    for model in models:
        for dataset in datasets:
            for mask in masks:
                folder = os.path.join(root, model, dataset, mask)
                if not os.path.isdir(folder):
                    logger.info("Not found, skipping: %s", folder)
                    continue
                logger.info("Processing: model=%s dataset=%s mask=%s", model, dataset, mask)
                _process_folder(folder, dashed_nabs=False)


def _run_audio(args: argparse.Namespace) -> None:
    root     = args.root or os.path.join(PROJECT_ROOT, "audio", "morf_lerf")
    datasets = args.datasets if args.datasets else _AUD_DATASETS
    models   = args.models   if args.models   else _AUD_MODELS
    masks    = args.masks    if args.masks    else _AUD_MASKS
    folds    = args.esc50_folds if args.esc50_folds else [1, 2, 3, 4, 5]

    for dataset in datasets:
        for model in models:
            for mask in masks:
                logger.info("Processing: dataset=%s model=%s mask=%s", dataset, model, mask)
                if dataset == "esc50":
                    for fold in folds:
                        folder = os.path.join(root, dataset, model, mask, f"fold_{fold}")
                        if not os.path.isdir(folder):
                            logger.info("Not found, skipping: %s", folder)
                            continue
                        _process_folder(folder, dashed_nabs=True)
                else:
                    folder = os.path.join(root, dataset, model, mask)
                    if not os.path.isdir(folder):
                        logger.info("Not found, skipping: %s", folder)
                        continue
                    _process_folder(folder, dashed_nabs=True)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate MoRF/LeRF accuracy-curve plots (image / audio).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--domain", required=True, choices=["image", "audio"])
    parser.add_argument("--root",     default=None,
                        help="Override result root directory.")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Dataset(s) to process; defaults to all.")
    parser.add_argument("--models",   nargs="+", default=None,
                        help="Model(s) to process; defaults to all.")
    parser.add_argument("--masks",    nargs="+", default=None,
                        help="Mask type(s) to process; defaults to all.")
    parser.add_argument("--esc50_folds", nargs="+", type=int, default=None,
                        help="(audio) ESC-50 fold numbers (default: 1–5).")
    args = parser.parse_args()

    dispatch = {"image": _run_image, "audio": _run_audio}
    dispatch[args.domain](args)
    logger.info("All done.")


if __name__ == "__main__":
    main()
