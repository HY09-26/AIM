import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


NON_ABS_METHODS = {
    "gradient",
    "gradinput",
    "smoothgrad",
    "integrad",
    "vargard",
}

# --------------------------------------------------
# Utilities
# --------------------------------------------------
def load_pkl(path):
    with open(path, "rb") as f:
        return np.asarray(pickle.load(f))


def percent_masked(n_steps):
    """step index → % pixels masked"""
    return np.linspace(0, 100, n_steps + 1)


def discover_methods(base_dir):
    """
    Discover methods from filenames:
      <method>_morf.pkl
      <method>_lerf.pkl
    """
    methods = set()
    for fname in os.listdir(base_dir):
        if fname.endswith("_morf.pkl"):
            methods.add(fname.replace("_morf.pkl", ""))
        elif fname.endswith("_lerf.pkl"):
            methods.add(fname.replace("_lerf.pkl", ""))
    return sorted(list(methods))


# --------------------------------------------------
# Plotting
# --------------------------------------------------
def plot_method_morf_lerf(base_dir, method, save_dir):
    """
    Plot MoRF + LeRF in ONE figure for a single method
    """
    morf_path = os.path.join(base_dir, f"{method}_morf.pkl")
    lerf_path = os.path.join(base_dir, f"{method}_lerf.pkl")

    if not os.path.exists(morf_path) or not os.path.exists(lerf_path):
        print(f"[WARN] Missing MoRF or LeRF for {method}, skipping")
        return

    morf = load_pkl(morf_path)
    lerf = load_pkl(lerf_path)

    x = percent_masked(len(morf) - 1)

    plt.figure(figsize=(5.5, 4.5))
    linestyle = "--" if method in NON_ABS_METHODS else "-"

    plt.plot(x, morf, marker="o", linestyle=linestyle, label="MoRF")
    plt.plot(x, lerf, marker="s", linestyle=linestyle, label="LeRF")

    plt.xlabel("% Pixels Masked")
    plt.ylabel("Accuracy")
    plt.title(method)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    out_path = os.path.join(save_dir, f"{method}_morf_lerf.png")
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_all_methods(base_dir, methods, mode, save_dir):
    """
    Plot all methods together for MoRF or LeRF
    """
    plt.figure(figsize=(6, 5))

    plotted = False
    for method in methods:
        path = os.path.join(base_dir, f"{method}_{mode}.pkl")
        if not os.path.exists(path):
            continue

        curve = load_pkl(path)
        x = percent_masked(len(curve) - 1)
        linestyle = "--" if method in NON_ABS_METHODS else "-"
        plt.plot(x, curve, linewidth=2, linestyle=linestyle, label=method)
        plotted = True

    if not plotted:
        print(f"[WARN] No curves found for {mode.upper()}, skipping")
        plt.close()
        return

    plt.xlabel("% Pixels Masked")
    plt.ylabel("Accuracy")
    plt.title(f"{mode.upper()} – All Methods")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    out_path = os.path.join(save_dir, f"all_{mode}.png")
    plt.savefig(out_path, dpi=200)
    plt.close()


# --------------------------------------------------
# Main (ALL COMBINATIONS)
# --------------------------------------------------
if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    DATASETS = ["audiomnist", "esc50", "msos"]
    MODELS = ["audionet", "res1dnet31"]
    MASK_TYPES = ["zero", "pgd", "road"]

    BASE_ROOT = os.path.join(PROJECT_ROOT, "morf_lerf")

    ESC50_FOLDS = [1] 

    for dataset in DATASETS:
        for model in MODELS:
            for mask_type in MASK_TYPES:

                # -------------------------
                # ESC50: .../<mask_type>/fold_1/
                # -------------------------
                if dataset == "esc50":
                    mask_dir = os.path.join(BASE_ROOT, dataset, model, mask_type)
                    if not os.path.isdir(mask_dir):
                        print(f"[SKIP] Not found: {mask_dir}")
                        continue

                    for fold in ESC50_FOLDS:
                        base_dir = os.path.join(mask_dir, f"fold_{fold}")

                        if not os.path.isdir(base_dir):
                            print(f"[SKIP] Not found: {base_dir}")
                            continue

                        plot_dir = os.path.join(base_dir, "plots")
                        os.makedirs(plot_dir, exist_ok=True)

                        print("\n==============================")
                        print(f"[INFO] dataset={dataset} model={model} mask={mask_type} fold=fold_{fold}")
                        print(f"[INFO] Scanning folder: {base_dir}")

                        methods = discover_methods(base_dir)
                        if len(methods) == 0:
                            print("[WARN] No *_morf.pkl or *_lerf.pkl found, skipping")
                            continue

                        print("[INFO] Found methods:", methods)

                        for method in methods:
                            plot_method_morf_lerf(base_dir, method, plot_dir)

                        plot_all_methods(base_dir, methods, mode="morf", save_dir=plot_dir)
                        plot_all_methods(base_dir, methods, mode="lerf", save_dir=plot_dir)

                        print("[DONE] Saved to:", plot_dir)

                # -------------------------
                # Others: .../<mask_type>/
                # -------------------------
                else:
                    base_dir = os.path.join(BASE_ROOT, dataset, model, mask_type)

                    if not os.path.isdir(base_dir):
                        print(f"[SKIP] Not found: {base_dir}")
                        continue

                    plot_dir = os.path.join(base_dir, "plots")
                    os.makedirs(plot_dir, exist_ok=True)

                    print("\n==============================")
                    print(f"[INFO] dataset={dataset} model={model} mask={mask_type}")
                    print(f"[INFO] Scanning folder: {base_dir}")

                    methods = discover_methods(base_dir)
                    if len(methods) == 0:
                        print("[WARN] No *_morf.pkl or *_lerf.pkl found, skipping")
                        continue

                    print("[INFO] Found methods:", methods)

                    for method in methods:
                        plot_method_morf_lerf(base_dir, method, plot_dir)

                    plot_all_methods(base_dir, methods, mode="morf", save_dir=plot_dir)
                    plot_all_methods(base_dir, methods, mode="lerf", save_dir=plot_dir)

                    print("[DONE] Saved to:", plot_dir)

    print("\n[ALL DONE] Finished all combinations.")