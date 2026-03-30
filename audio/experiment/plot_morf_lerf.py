"""
plot_morf_lerf.py - Visualise MoRF and LeRF accuracy curves.

For each (dataset, model, mask_type) combination, generates:
  - One figure per attribution method showing MoRF and LeRF on the same axes.
  - One figure showing all MoRF curves together.
  - One figure showing all LeRF curves together.

Plots are saved as PNG files inside a "plots/" subdirectory next to the result
pickle files.
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

# ============================================================
# Constants
# ============================================================

# Non-absolute methods are plotted with dashed lines to visually distinguish
# them from absolute-value variants (which tend to be stronger attributions).
NON_ABS_METHODS = {
    "gradient",
    "gradinput",
    "smoothgrad",
    "integrad",
    "vargrad",      # was misspelled "vargard" in original
}

DATASETS   = ["audiomnist", "esc50", "msos"]
MODELS     = ["audionet", "res1dnet31"]
MASK_TYPES = ["zero", "pgd", "road"]


# ============================================================
# Helpers
# ============================================================

def load_curve(path: str) -> np.ndarray:
    """Load a MoRF or LeRF accuracy curve from a pickle file."""
    with open(path, "rb") as f:
        return np.asarray(pickle.load(f), dtype=float)


def percent_masked(n_steps: int) -> np.ndarray:
    """Return evenly-spaced x-axis ticks from 0% to 100% for n_steps+1 points."""
    return np.linspace(0, 100, n_steps + 1)


def discover_methods(base_dir: str) -> list[str]:
    """
    Find all attribution method names that have both *_morf.pkl and *_lerf.pkl.

    Args:
        base_dir: Directory containing *_morf.pkl and *_lerf.pkl files.

    Returns:
        Sorted list of method names.
    """
    morf_methods = set()
    lerf_methods = set()
    for fname in os.listdir(base_dir):
        if fname.endswith("_morf.pkl"):
            morf_methods.add(fname[: -len("_morf.pkl")])
        elif fname.endswith("_lerf.pkl"):
            lerf_methods.add(fname[: -len("_lerf.pkl")])
    return sorted(morf_methods & lerf_methods)


# ============================================================
# Plotting
# ============================================================

def plot_one_method(
    base_dir: str,
    method: str,
    save_dir: str,
) -> None:
    """
    Plot MoRF and LeRF curves for a single attribution method on one figure.

    Args:
        base_dir: Directory containing the .pkl files.
        method:   Attribution method name (stem of pickle filename).
        save_dir: Directory to save the PNG figure.
    """
    morf_path = os.path.join(base_dir, f"{method}_morf.pkl")
    lerf_path = os.path.join(base_dir, f"{method}_lerf.pkl")

    if not (os.path.exists(morf_path) and os.path.exists(lerf_path)):
        logger.warning("Missing MoRF or LeRF for %s, skipping.", method)
        return

    morf = load_curve(morf_path)
    lerf = load_curve(lerf_path)
    x    = percent_masked(len(morf) - 1)

    linestyle = "--" if method in NON_ABS_METHODS else "-"
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(x, morf, marker="o", linestyle=linestyle, label="MoRF")
    ax.plot(x, lerf, marker="s", linestyle=linestyle, label="LeRF")
    ax.set_xlabel("% Masked")
    ax.set_ylabel("Accuracy")
    ax.set_title(method)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"{method}_morf_lerf.png"), dpi=200)
    plt.close(fig)


def plot_all_methods(
    base_dir: str,
    methods: list[str],
    mode: str,
    save_dir: str,
) -> None:
    """
    Plot all attribution methods' MoRF (or LeRF) curves on one figure.

    Args:
        base_dir: Directory containing the .pkl files.
        methods:  List of method names to include.
        mode:     "morf" or "lerf".
        save_dir: Directory to save the PNG figure.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    plotted = False

    for method in methods:
        path = os.path.join(base_dir, f"{method}_{mode}.pkl")
        if not os.path.exists(path):
            continue
        curve     = load_curve(path)
        x         = percent_masked(len(curve) - 1)
        linestyle = "--" if method in NON_ABS_METHODS else "-"
        ax.plot(x, curve, linewidth=2, linestyle=linestyle, label=method)
        plotted = True

    if not plotted:
        logger.warning("No curves found for %s, skipping combined plot.", mode.upper())
        plt.close(fig)
        return

    ax.set_xlabel("% Masked")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{mode.upper()} – All Methods")
    ax.legend(fontsize=8)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"all_{mode}.png"), dpi=200)
    plt.close(fig)


# ============================================================
# Per-folder entry point
# ============================================================

def process_folder(base_dir: str) -> None:
    """Discover methods in base_dir and generate all plots."""
    methods = discover_methods(base_dir)
    if not methods:
        logger.warning("No MoRF/LeRF pairs found in %s, skipping.", base_dir)
        return

    plot_dir = os.path.join(base_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    logger.info("Found %d methods: %s", len(methods), methods)
    logger.info("Saving plots to: %s", plot_dir)

    for method in methods:
        plot_one_method(base_dir, method, plot_dir)

    plot_all_methods(base_dir, methods, mode="morf", save_dir=plot_dir)
    plot_all_methods(base_dir, methods, mode="lerf", save_dir=plot_dir)


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate MoRF/LeRF accuracy curve plots."
    )
    parser.add_argument("--root",      default=None,
                        help="Root directory of morf_lerf result folders. "
                             "Defaults to <project_root>/morf_lerf.")
    parser.add_argument("--datasets",  nargs="+", default=DATASETS,
                        choices=DATASETS)
    parser.add_argument("--models",    nargs="+", default=MODELS,
                        choices=MODELS)
    parser.add_argument("--masks",     nargs="+", default=MASK_TYPES,
                        choices=MASK_TYPES)
    parser.add_argument("--esc50_folds", nargs="+", type=int,
                        default=[1, 2, 3, 4, 5])
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base_root    = args.root or os.path.join(project_root, "morf_lerf")

    for dataset in args.datasets:
        for model in args.models:
            for mask_type in args.masks:
                logger.info("Processing: dataset=%s model=%s mask=%s",
                            dataset, model, mask_type)
                if dataset == "esc50":
                    for fold in args.esc50_folds:
                        base_dir = os.path.join(
                            base_root, dataset, model, mask_type, f"fold_{fold}"
                        )
                        if not os.path.isdir(base_dir):
                            logger.info("Folder not found, skipping: %s", base_dir)
                            continue
                        process_folder(base_dir)
                else:
                    base_dir = os.path.join(base_root, dataset, model, mask_type)
                    if not os.path.isdir(base_dir):
                        logger.info("Folder not found, skipping: %s", base_dir)
                        continue
                    process_folder(base_dir)

    logger.info("All done.")


if __name__ == "__main__":
    main()
