import os
import pickle
import numpy as np
import pandas as pd
from scipy.stats import rankdata

# =========================
# CONFIG
# =========================
BASE_DIR = "/mnt/left/home/2025/tony/hsinyuan/AIM_eeg/experiment"

DATASETS = ["MI", "ERN", "SSVEP"]
MODELS   = ["eegnet", "icnn", "sccnet"]
DOMAINS  = ["ch", "ts", "fq"]

DOMAIN_NAME = {
    "ch": "Spatial",
    "ts": "Temporal",
    "fq": "Spectral",
}

SUBDIR_SUFFIX = {
    "ch": "",
    "ts": "_sto",
    "fq": "_si",
}

LEG_NABS = ["GD", "GI", "SG", "SS", "VG", "IG", "RD"]
LEG_ABS  = ["GDA", "GIA", "SGA", "IGA", "RD"]
LEG_ALL  = ["GD", "GI", "SG", "SS", "VG", "IG",
            "GDA", "GIA", "SGA", "IGA", "RD"]

# =========================
# LOADER
# =========================
class CompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core")
        return super().find_class(module, name)

def load_pickle(path):
    with open(path, "rb") as f:
        return CompatUnpickler(f).load()

# =========================
# SPEARMAN
# =========================
def spearman_rank(morf_scores, lerf_scores):
    rank_morf = rankdata(morf_scores)
    rank_lerf = rankdata(-lerf_scores)

    if np.std(rank_morf) < 1e-10 or np.std(rank_lerf) < 1e-10:
        return np.nan

    return np.corrcoef(rank_morf, rank_lerf)[0, 1]


def get_method_curves(morf_nabs, lerf_nabs, morf_abs, lerf_abs):
    method_curves = {}

    for i in range(len(morf_nabs) - 1):
        method_curves[LEG_NABS[i]] = (
            morf_nabs[i]["acc"],
            lerf_nabs[i]["acc"]
        )

    for i in range(len(morf_abs) - 1):
        method_curves[LEG_ABS[i]] = (
            morf_abs[i]["acc"],
            lerf_abs[i]["acc"]
        )

    method_curves["RD"] = (
        (morf_nabs[-1]["acc"] + morf_abs[-1]["acc"]) / 2,
        (lerf_nabs[-1]["acc"] + lerf_abs[-1]["acc"]) / 2
    )

    return method_curves


# =========================
# MAIN
# =========================
rows = []

for dataset in DATASETS:
    for model in MODELS:
        for domain in DOMAINS:

            all_rhos = []

            for rep in range(1):

                suffix = SUBDIR_SUFFIX[domain]
                data_dir = os.path.join(
                    BASE_DIR,
                    f"repeat{rep}/{domain}_test_zero/{dataset}{suffix}"
                )

                try:
                    morf_nabs = load_pickle(os.path.join(data_dir, f"{model}_{domain}_nabs_morf.pickle"))
                    lerf_nabs = load_pickle(os.path.join(data_dir, f"{model}_{domain}_nabs_lerf.pickle"))

                    morf_abs = load_pickle(os.path.join(data_dir, f"{model}_{domain}_abs_morf.pickle"))
                    lerf_abs = load_pickle(os.path.join(data_dir, f"{model}_{domain}_abs_lerf.pickle"))

                except:
                    continue

                method_curves = get_method_curves(
                    morf_nabs, lerf_nabs,
                    morf_abs, lerf_abs
                )

                morf_stack = []
                lerf_stack = []

                for method in LEG_ALL:
                    morf, lerf = method_curves[method]

                    morf_stack.append(np.mean(morf, axis=0))
                    lerf_stack.append(np.mean(lerf, axis=0))

                morf_stack = np.array(morf_stack)
                lerf_stack = np.array(lerf_stack)

                K = morf_stack.shape[1]

                # same rule as paper
                if domain == "ch" and dataset != "ERN":
                    use_K = K // 2
                else:
                    use_K = K

                for k in range(use_K):
                    rho = spearman_rank(
                        morf_stack[:, k],
                        lerf_stack[:, k]
                    )
                    all_rhos.append(rho)

            all_rhos = np.array(all_rhos)

            rows.append({
                "dataset": dataset,
                "model": model,
                "domain": DOMAIN_NAME[domain],
                "mean": np.nanmean(all_rhos),
                "std": np.nanstd(all_rhos)
            })


df = pd.DataFrame(rows)

# =========================
# PRINT TABLE (paper style)
# =========================
def fmt(mu, sd):
    return f"{mu:.3f}±{sd:.3f}".replace("0.", ".")

def get_val(dataset, model, domain):
    tmp = df[
        (df["dataset"] == dataset) &
        (df["model"] == model) &
        (df["domain"] == domain)
    ]
    if len(tmp) == 0:
        return "-"
    r = tmp.iloc[0]
    return fmt(r["mean"], r["std"])


# column width
W = 12

def cell(x):
    return f"{x:<{W}}"


print("\n" + "="*120)

# ===== HEADER =====
print(
    cell("Framework") + cell("Domain") +
    cell("SMR-EEGNet") + cell("SMR-ICNN") + cell("SMR-SCCNet") +
    cell("ERN-EEGNet") + cell("ERN-ICNN") + cell("ERN-SCCNet") +
    cell("SSVEP-EEGNet") + cell("SSVEP-ICNN") + cell("SSVEP-SCCNet")
)

print("-"*120)

# ===== BODY =====
for i, domain in enumerate(["Spatial", "Temporal", "Spectral"]):

    framework = "Zeroing" if i == 0 else ""

    row = [
        cell(framework),
        cell(domain)
    ]

    for dataset in ["MI", "ERN", "SSVEP"]:
        for model in ["eegnet", "icnn", "sccnet"]:
            row.append(cell(get_val(dataset, model, domain)))

    print("".join(row))

print("="*120)