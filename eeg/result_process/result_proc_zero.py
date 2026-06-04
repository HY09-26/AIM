import os
import numpy as np
import pickle
import pandas as pd

# =========================
# CONFIG
# =========================
BASE_DIR = "/mnt/left/home/2025/tony/hsinyuan/AIM_eeg/experiment"
MODEL_ACC_BASE = "/mnt/right/alumni/2023/irishsieh/atk"

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
# PICKLE LOADER
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
# METRICS
# =========================
def compute_metrics_per_subject(morf_raw, lerf_raw, baseline):
    """
    morf_raw, lerf_raw: (n_subjects, K), 不含 x0
    baseline: (n_subjects,)
    return: AOC, ABC, AUC for each subject
    """
    morf = np.concatenate([baseline[:, None], morf_raw], axis=1)
    lerf = np.concatenate([baseline[:, None], lerf_raw], axis=1)

    n_sub, Kp1 = morf.shape
    K = Kp1 - 1

    AOC = np.zeros(n_sub)
    ABC = np.zeros(n_sub)
    AUC = np.zeros(n_sub)

    for i in range(n_sub):
        x0 = morf[i, 0]
        xa_m = morf[i, -1]
        xa_l = lerf[i, -1]

        denom_M = (x0 - xa_m) + 1e-8
        denom_L = (x0 - xa_l) + 1e-8


        for k in range(1, K + 1):
            aoc_k = (x0 - morf[i, k]) / denom_M
            abc_k = (lerf[i, k] - morf[i, k]) / denom_L 
            auc_k = (lerf[i, k] - lerf[i, -1]) / denom_L

            AOC[i] += np.clip(aoc_k, 0, 1)
            ABC[i] += np.clip(abc_k, 0, 1)
            AUC[i] += np.clip(auc_k, 0, 1)

        AOC[i] /= K
        ABC[i] /= K
        AUC[i] /= K

    return AOC, ABC, AUC


# =========================
# COLLECT ALL RESULTS
# =========================
rows = []

for dataset in DATASETS:
    model_acc_path = os.path.join(MODEL_ACC_BASE, f"model_accs_{dataset}.npy")
    model_accs = np.load(model_acc_path)

    print(f"\nDataset: {dataset}")
    print("model_accs shape:", model_accs.shape)

    n_reps = model_accs.shape[0]

    for rep in range(n_reps):
        for mid, model in enumerate(MODELS):
            baseline = model_accs[rep, mid]

            for domain in DOMAINS:
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

                except FileNotFoundError:
                    print(f"Missing: {dataset} | rep{rep} | {model} | {domain}")
                    continue

                method_curves = {}

                # signed: GD, GI, SG, SS, VG, IG
                for i in range(len(morf_nabs) - 1):
                    method_curves[LEG_NABS[i]] = (
                        morf_nabs[i]["acc"],
                        lerf_nabs[i]["acc"]
                    )

                # abs: GDA, GIA, SGA, IGA
                for i in range(len(morf_abs) - 1):
                    method_curves[LEG_ABS[i]] = (
                        morf_abs[i]["acc"],
                        lerf_abs[i]["acc"]
                    )

                # random: average nabs + abs random
                method_curves["RD"] = (
                    (morf_nabs[-1]["acc"] + morf_abs[-1]["acc"]) / 2,
                    (lerf_nabs[-1]["acc"] + lerf_abs[-1]["acc"]) / 2
                )

                for method in LEG_ALL:
                    if method not in method_curves:
                        continue

                    morf, lerf = method_curves[method]

                    AOC, ABC, AUC = compute_metrics_per_subject(
                        morf, lerf, baseline
                    )

                    # 先平均 subject，得到一個 dataset × model × repeat 的值
                    rows.append({
                        "dataset": dataset,
                        "model": model,
                        "rep": rep,
                        "domain": DOMAIN_NAME[domain],
                        "method": method,
                        "AOC": np.mean(AOC),
                        "ABC": np.mean(ABC),
                        "AUC": np.mean(AUC),
                    })


df = pd.DataFrame(rows)
df.to_csv("zeroing_raw_dataset_model_repeat.csv", index=False)


# =========================
# MEAN ± STD ACROSS ALL dataset × model × repeat
# =========================
summary = (
    df.groupby(["domain", "method"])
      .agg(
          AOC_mean=("AOC", "mean"),
          AOC_std=("AOC", "std"),
          ABC_mean=("ABC", "mean"),
          ABC_std=("ABC", "std"),
          AUC_mean=("AUC", "mean"),
          AUC_std=("AUC", "std"),
          n=("AOC", "count"),
      )
      .reset_index()
)

summary.to_csv("zeroing_mean_std_across_dataset_model_repeat.csv", index=False)


# =========================
# PRINT TABLE LIKE PAPER
# =========================
def fmt(mu, sd):
    return f"{mu:.3f}±{sd:.3f}".replace("0.", ".")

def get_val(method, domain, metric):
    tmp = summary[
        (summary["method"] == method) &
        (summary["domain"] == domain)
    ]
    if len(tmp) == 0:
        return "-"

    r = tmp.iloc[0]
    return fmt(r[f"{metric}_mean"], r[f"{metric}_std"])

W = 13

def cell(x):
    return f"{x:<{W}}"

print("\n" + "=" * 150)

# header 1
print(
    cell("Method") +
    cell("Spatial") + cell("") + cell("") +
    cell("Temporal") + cell("") + cell("") +
    cell("Spectral") + cell("") + cell("")
)

# header 2
print(
    cell("") +
    cell("AOC") + cell("ABC") + cell("AUC") +
    cell("AOC") + cell("ABC") + cell("AUC") +
    cell("AOC") + cell("ABC") + cell("AUC")
)

print("-" * 150)

for method in LEG_ALL:
    row = [cell(method)]

    for domain in ["Spatial", "Temporal", "Spectral"]:
        for metric in ["AOC", "ABC", "AUC"]:
            row.append(cell(get_val(method, domain, metric)))

    print("".join(row))

print("=" * 150)
