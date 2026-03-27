import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


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
    plt.plot(x, morf, marker="o", label="MoRF")
    plt.plot(x, lerf, marker="s", label="LeRF")

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
        plt.plot(x, curve, marker="o", label=method)
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
# Main
# --------------------------------------------------
if __name__ == "__main__":

    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    BASE_DIR = os.path.join(
        PROJECT_ROOT,
        "morf_lerf_image",
        "repvgg_b0",
        "imagenet",
        "pgd",
    )

    if not os.path.isdir(BASE_DIR):
        raise RuntimeError(f"Base directory not found: {BASE_DIR}")

   
    PLOT_DIR = os.path.join(BASE_DIR, "plots")
    os.makedirs(PLOT_DIR, exist_ok=True)

    print("[INFO] Scanning folder:", BASE_DIR)

    methods = discover_methods(BASE_DIR)

    if len(methods) == 0:
        raise RuntimeError("No *_morf.pkl or *_lerf.pkl found.")

    print("[INFO] Found methods:", methods)

    # 1) Each method: MoRF + LeRF in one figure
    for method in methods:
        plot_method_morf_lerf(BASE_DIR, method, PLOT_DIR)

    # 2) All methods MoRF
    plot_all_methods(BASE_DIR, methods, mode="morf", save_dir=PLOT_DIR)

    # 3) All methods LeRF
    plot_all_methods(BASE_DIR, methods, mode="lerf", save_dir=PLOT_DIR)

    print("[DONE] All plots saved to:", PLOT_DIR)
