# experiment_utils/expl_vis_esc50.py
import os
import random
import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch

from model.cnn14 import Cnn14
from model.res1dnet31 import Res1dNet31


# ------------------------------------------------------------
# utils
# ------------------------------------------------------------
def load_h5(path):
    with h5py.File(path, "r") as f:
        x = f["data"][:]
        y = int(f["label"][0][0])
    return x, y

def read_test_list(txt, key):
    return [l.strip() for l in open(txt) if key in l]

def cnn14_forward_logmel(model, waveform):

    """
    waveform: (T,) numpy
    return: log-mel (mel_bins=64, time)
    """
    model.eval()
    with torch.no_grad():
        x = torch.tensor(waveform, dtype=torch.float32)
        x = x.unsqueeze(0).to(next(model.parameters()).device)  # (1, T)

        spec = model.spectrogram_extractor(x)   # (1, 1, time, freq)
        logmel = model.logmel_extractor(spec)   # (1, 1, time, mel)

        logmel = logmel.squeeze(0).squeeze(0)   # (time, mel)
        logmel = logmel.transpose(0, 1)         # (mel, time)

    return logmel.cpu().numpy()

def norm(x):
    x_mean = x.mean()
    x_std = x.std()
    return (x - x_mean) / (x_std + 1e-8)

def overlay_saliency(ax, base, sal, title):
    """
    base: log-mel or spectrogram (2D)
    sal : signed saliency (same shape)
    """

    # background (gray log-mel)
    ax.imshow(
        base,
        origin="lower",
        aspect="auto",
        cmap="gray"
    )

    # symmetric color scaling
    vmax = np.percentile(np.abs(sal), 99)
    if vmax < 1e-8:
        vmax = 1e-8

    ax.imshow(
        sal,
        origin="lower",
        aspect="auto",
        cmap="seismic",
        alpha=0.3,
        vmin=-vmax,
        vmax= vmax
    )

    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Mel bins")


# ------------------------------------------------------------
# plotters
# ------------------------------------------------------------
def plot_cnn14(waveform, saliency_dict, model, save_path):
    """
    waveform: (T,)
    saliency_dict: {method: (64, T')}
    """

    logmel = cnn14_forward_logmel(model, waveform)  # (64, T)

    # --------------------------------------------------
    # layout definition (6 x 2)
    # --------------------------------------------------
    layout = [
        ("input", "random"),
        ("gradient", "gradient_abs"),
        ("gradinput", "gradinput_abs"),
        ("smoothgrad", "smoothgrad_abs"),
        ("smoothgrad_sq", "vargrad"),
        ("integrad", "integrad_abs"),
    ]

    fig, axes = plt.subplots(6, 2, figsize=(14, 18), sharex=True, sharey=True)

    for r, (m1, m2) in enumerate(layout):
        # ---- Col 1 ----
        if m1 == "input":
            axes[r, 0].imshow(
                logmel,
                origin="lower",
                aspect="auto",
                cmap="magma"
            )
            axes[r, 0].set_title("Input log-mel")
        else:
            overlay_saliency(
                axes[r, 0],
                logmel,
                saliency_dict[m1],
                title=m1
            )

        # ---- Col 2 ----
        overlay_saliency(
            axes[r, 1],
            logmel,
            saliency_dict[m2],
            title=m2
        )

    for ax in axes.flat:
        ax.set_xlabel("Time")
        ax.set_ylabel("Mel bins")

   
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, dpi=200)
    plt.close()



def plot_res1d(waveform, saliency_dict, title, save_path):
    n = 1 + len(saliency_dict)
    fig, axes = plt.subplots(
        n, 1, figsize=(14, 1.8 * n), sharex=True
    )

    # --- waveform ---
    axes[0].plot(waveform, linewidth=0.8)
    axes[0].set_title(title)
    axes[0].set_ylabel("Amplitude")
    axes[0].set_xlabel("Sample index")
    axes[0].tick_params(labelsize=8)

    # --- saliency ---
    for i, (k, v) in enumerate(saliency_dict.items(), 1):
        axes[i].plot(norm(v), linewidth=0.8)
        axes[i].set_title(k)
        axes[i].set_ylabel("Saliency")
        axes[i].set_xlabel("Sample index")
        axes[i].tick_params(labelsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()



# ------------------------------------------------------------
# main
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--models", nargs="+", default=["cnn14", "res1dnet31"])
    args = ap.parse_args()

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA = os.path.join(ROOT, "experiment_utils/data/ESC50/preprocessed_data")
    EXPL = os.path.join(ROOT, "expl_esc50")
    SAVE_DIR = os.path.join(EXPL, "visualization")
    os.makedirs(SAVE_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --------------------------------------------------------
    # load models once
    # --------------------------------------------------------
    models = {}
    if "cnn14" in args.models:
        m = Cnn14(num_classes=50).to(device)
        ckpt = torch.load(
            os.path.join(
                ROOT,
                "experiment_utils/checkpoints/esc50/cnn14",
                f"fold_{args.fold}",
                "best_model.pth"
            ),
            map_location="cpu"
        )
        m.load_state_dict(ckpt["state_dict"], strict=False)
        m.eval()
        models["cnn14"] = m

    if "res1dnet31" in args.models:
        m = Res1dNet31(num_classes=50).to(device)
        ckpt = torch.load(
            os.path.join(
                ROOT,
                "experiment_utils/checkpoints/esc50/res1dnet31",
                f"fold_{args.fold}",
                "best_model.pth"
            ),
            map_location="cpu"
        )
        m.load_state_dict(ckpt["state_dict"], strict=False)
        m.eval()
        models["res1dnet31"] = m

    test_txt = os.path.join(DATA, f"ESC50_fold{args.fold}_test.txt")

    
    n = 1
    for model_name in args.models:
        is_cnn = model_name == "cnn14"
        key = "waveform"   


        if n==1:
            files = read_test_list(test_txt, key)
            idx = random.randrange(len(files))
            fname = files[idx]
            n += 1
        else:
            fname = files[idx]

        x_raw, y = load_h5(os.path.join(DATA, fname))
        waveform = x_raw.squeeze()   # (T,)

        # ----------------------------------------------------
        # load saliency
        # ----------------------------------------------------
        expl_dir = os.path.join(EXPL, model_name, f"fold_{args.fold}")
        methods = sorted(f[:-4] for f in os.listdir(expl_dir) if f.endswith(".npy"))

        sal = {}
        for m in methods:
            sal[m] = np.load(os.path.join(expl_dir, f"{m}.npy"))[idx].squeeze()

        print(f"\n[{model_name}] waveform {waveform.shape}")
        for k, v in sal.items():
            print(f"  {k}: {v.shape}")

        save_path = os.path.join(
            SAVE_DIR,
            f"{model_name}_fold{args.fold}_{fname.replace('.hdf5','')}.png"
        )

        if is_cnn:
            plot_cnn14(waveform, sal, models["cnn14"], save_path)
        else:
            plot_res1d(waveform, sal, model_name, save_path)

        print(f"[Saved] {save_path}")


if __name__ == "__main__":
    main()
