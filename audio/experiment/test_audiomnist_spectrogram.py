# # test_audiomnist_spectrogram.py
# import os
# import sys
# import argparse
# import pickle
# from pathlib import Path

# import numpy as np
# import h5py
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader, TensorDataset
# import torch.nn.functional as F
# from tqdm import tqdm

# # ============================================================
# # Path
# # ============================================================
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
# sys.path.append(PROJECT_ROOT)

# from experiment_utils.model import alexnet

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# # ============================================================
# # AudioMNIST spectrogram loader (self-contained)
# # ============================================================
# def load_h5_spectrogram(path):
#     """
#     Expect h5 format:
#       data: (1,1,H,W)
#       label: scalar
#     """
#     with h5py.File(path, "r") as f:
#         x = f["data"][:]            # (1,1,H,W)
#         y = int(f["label"][0][0])
#     x = np.asarray(x, np.float32)[0]   # -> (1,H,W)
#     return x, y


# def load_test_set(split_txt, data_dir):
#     xs, ys = [], []
#     with open(split_txt, "r") as f:
#         for line in f:
#             rel = line.strip()
#             if not rel:
#                 continue

#             full = os.path.join(data_dir, rel)

        
#             # print("Loading:", full)

#             x, y = load_h5_spectrogram(full)
#             xs.append(x)
#             ys.append(y)

#     if len(xs) == 0:
#         raise RuntimeError(
#             f"No samples loaded from {split_txt}. "
#             f"Check split file content or filtering logic."
#         )

#     X = np.stack(xs, axis=0).astype(np.float32)
#     y = np.asarray(ys, dtype=np.int64)
#     return X, y



# def build_loader_from_np(X_np, y_np, batch_size):
#     x = torch.tensor(X_np, dtype=torch.float32)
#     y = torch.tensor(y_np, dtype=torch.long)
#     ds = TensorDataset(x, y)
#     return DataLoader(ds, batch_size=batch_size, shuffle=False)


# # ============================================================
# # Evaluation (ESC50-style)
# # ============================================================
# @torch.no_grad()
# def eval_epoch(model, loader):
#     model.eval()
#     correct, total = 0, 0
#     for x, y in loader:
#         x, y = x.to(device), y.to(device)
#         out = model(x)
#         correct += (out.argmax(1) == y).sum().item()
#         total += y.size(0)
#     return correct / total


# # ============================================================
# # PGD attack (spectrogram space)
# # ============================================================
# def pgd_attack_spectrogram(
#     model,
#     X,
#     y,
#     eps=1.0,
#     steps=20,
#     batch_size=64,
# ):
#     """
#     X: (N,1,H,W)
#     """
#     model.eval()
#     X_adv = X.copy()
#     step = eps / steps
#     N = X.shape[0]

#     for i in range(0, N, batch_size):
#         xb = torch.tensor(
#             X_adv[i:i+batch_size],
#             device=device,
#             requires_grad=True
#         )
#         yb = torch.tensor(
#             y[i:i+batch_size],
#             device=device
#         )

#         for _ in range(steps):
#             out = model(xb)
#             loss = nn.CrossEntropyLoss()(out, yb)
#             loss.backward()

#             xb = xb + step * xb.grad.sign()
#             xb = torch.max(
#                 torch.min(
#                     xb,
#                     torch.tensor(X[i:i+batch_size], device=device) + eps
#                 ),
#                 torch.tensor(X[i:i+batch_size], device=device) - eps
#             )
#             xb = xb.detach().requires_grad_(True)

#         X_adv[i:i+batch_size] = xb.detach().cpu().numpy()

#     return X_adv


# # ============================================================
# # ROAD (spectrogram)
# # ============================================================
# @torch.no_grad()
# def road_spectrogram(loader, noise_std=0.2):
#     """
#     Input loader yields (x,y), x:(B,1,H,W)
#     Return: (N,1,H,W)
#     """
#     X_all = []
#     for xb, _ in tqdm(loader, desc="[ROAD-SPEC] collect X"):
#         X_all.append(xb.cpu().numpy())

#     X = torch.from_numpy(
#         np.concatenate(X_all, axis=0).astype(np.float32)
#     )

#     N, C, H, W = X.shape
#     wd, wi = 1/6, 1/12

#     Xp = F.pad(X, (1,1,1,1), mode="reflect")

#     up    = Xp[:, :, 0:H,     1:W+1]
#     down  = Xp[:, :, 2:H+2,   1:W+1]
#     left  = Xp[:, :, 1:H+1,   0:W]
#     right = Xp[:, :, 1:H+1,   2:W+2]

#     ul = Xp[:, :, 0:H,     0:W]
#     ur = Xp[:, :, 0:H,     2:W+2]
#     dl = Xp[:, :, 2:H+2,   0:W]
#     dr = Xp[:, :, 2:H+2,   2:W+2]

#     interp = wd*(up+down+left+right) + wi*(ul+ur+dl+dr)
#     noise = noise_std * torch.randn_like(interp)

#     return (interp + noise).numpy().astype(np.float32)


# # ============================================================
# # MoRF / LeRF (ESC50-style)
# # ============================================================
# def run_morf_lerf(
#     model,
#     X,
#     y,
#     saliency,
#     n_steps,
#     batch_size,
#     mode,
#     X_adv=None,
# ):
#     """
#     X: (N,1,H,W)
#     saliency: (N,1,H,W) or (N,H,W)
#     """
#     N, _, H, W = X.shape
#     HW = H * W

#     x = X.reshape(N, 1, HW).copy()
#     s = saliency.reshape(N, -1)

#     order = np.argsort(-s, axis=1) if mode == "morf" else np.argsort(s, axis=1)
#     step = max(1, HW // n_steps)

#     zero = float(X.min())
#     acc_curve = []

#     print(f"\n[{mode.upper()}] n_steps={n_steps}, step_size={step}")

#     for k in range(n_steps + 1):

#         if k > 0:
#             for i in range(N):
#                 idx = order[i, (k-1)*step : k*step]
#                 if X_adv is None:
#                     x[i, :, idx] = zero
#                 else:
#                     x[i, :, idx] = X_adv.reshape(N,1,HW)[i,:,idx]

#         loader = build_loader_from_np(
#             x.reshape(N,1,H,W),
#             y,
#             batch_size
#         )

#         acc = eval_epoch(model, loader)
#         acc_curve.append(acc)

#         print(
#             f"[{mode.upper()}] step {k:02d}/{n_steps} | "
#             f"masked={100*k/n_steps:5.1f}% | acc={acc:.4f}"
#         )

#     return np.array(acc_curve)


# # ============================================================
# # Main
# # ============================================================
# def main():
#     parser = argparse.ArgumentParser()

#     parser.add_argument("--data_dir",
#                         default="../experiment_utils/data/audiomnist/preprocessed_data")
#     parser.add_argument("--split_txt",
#                         default="../experiment_utils/data/audiomnist/preprocessed_data/AlexNet_digit_0_test.txt")
#     parser.add_argument("--ckpt",
#                         default="../experiment_utils/checkpoints/audiomnist/alexnet/repeat0/best_model.pth")
#     parser.add_argument("--expl_dir",
#                         default="../expl_audiomnist/alexnet")
#     parser.add_argument("--expl_method", default="gradient")

#     parser.add_argument("--mask_type",
#                         choices=["zero", "pgd", "road"],
#                         default="road")

#     parser.add_argument("--epsilon", type=float, default=1.0)
#     parser.add_argument("--pgd_steps", type=int, default=20)
#     parser.add_argument("--n_steps", type=int, default=20)
#     parser.add_argument("--batch_size", type=int, default=64)

#     parser.add_argument("--out_dir",
#                         default=Path(PROJECT_ROOT) / "morf_lerf" / "alexnet")

#     args = parser.parse_args()

#     print("Device:", device)

#     # ---------- data ----------
#     X_test, y_test = load_test_set(args.split_txt, args.data_dir)
#     print("Test data:", X_test.shape, y_test.shape)

#     # ---------- model ----------
#     model = alexnet(num_classes=10).to(device)
#     ckpt = torch.load(args.ckpt, map_location=device)
#     model.load_state_dict(ckpt["state_dict"])
#     model.eval()

#     # ---------- saliency ----------
#     saliency = np.load(
#         os.path.join(args.expl_dir, f"{args.expl_method}.npy")
#     )

#     # ---------- replacement ----------
#     X_adv = None

#     if args.mask_type == "pgd":
#         print("[PGD] running attack...")
#         X_adv = pgd_attack_spectrogram(
#             model, X_test, y_test,
#             eps=args.epsilon,
#             steps=args.pgd_steps,
#             batch_size=args.batch_size
#         )

#     elif args.mask_type == "road":
#         noise_std = 0.1
#         road_dir = Path(args.out_dir) / "road"
#         road_dir.mkdir(parents=True, exist_ok=True)
#         road_path = road_dir / f"road_noise{noise_std:.3f}.npy"

#         if road_path.exists():
#             print("[ROAD] loading cache")
#             X_adv = np.load(road_path)
#         else:
#             print("[ROAD] computing ROAD...")
#             road_loader = build_loader_from_np(
#                 X_test, y_test, args.batch_size
#             )
#             X_adv = road_spectrogram(road_loader, noise_std=noise_std)
#             np.save(road_path, X_adv)

#     # ---------- MoRF / LeRF ----------
#     morf = run_morf_lerf(
#         model, X_test, y_test, saliency,
#         args.n_steps, args.batch_size, "morf", X_adv
#     )
#     lerf = run_morf_lerf(
#         model, X_test, y_test, saliency,
#         args.n_steps, args.batch_size, "lerf", X_adv
#     )

#     # ---------- save ----------
#     out_dir = Path(args.out_dir) / args.mask_type
#     out_dir.mkdir(parents=True, exist_ok=True)

#     with open(out_dir / f"{args.expl_method}_morf.pkl", "wb") as f:
#         pickle.dump(morf, f)
#     with open(out_dir / f"{args.expl_method}_lerf.pkl", "wb") as f:
#         pickle.dump(lerf, f)

#     print("Saved to:", out_dir)


# if __name__ == "__main__":
#     main()















# test_audiomnist_spectrogram.py
import os
import sys
import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from tqdm import tqdm

# ============================================================
# Path
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)

from experiment_utils.train_audiomnist import (
    AudioMNISTWaveDataset,
    AudioMNISTSpecDataset,
    ensure_wave_batch,
    ensure_spec_batch,
    MODEL_INPUT,
)

from experiment_utils.model.alexnet import AlexNet_Audio
from experiment_utils.model.cnn14 import Cnn14

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Loader
# ============================================================
def get_loader(split_txt, data_dir, model_name, batch_size):
    if MODEL_INPUT[model_name] == "spec":
        ds = AudioMNISTSpecDataset(split_txt, data_dir)
    else:
        ds = AudioMNISTWaveDataset(split_txt, data_dir)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )


# ============================================================
# CNN14 logmel wrapper 
# ============================================================
def _call_conv_block(conv_block, x, pool_size):
    try:
        return conv_block(x, pool_size=pool_size, pool_type="avg")
    except TypeError:
        return conv_block(x, pool_size=pool_size)


class CNN14LogmelWrapper(nn.Module):
    def __init__(self, cnn14):
        super().__init__()
        self.m = cnn14

    @torch.no_grad()
    def waveform_to_logmel(self, wav):

        # train 时 CNN14 forward 吃 (B,1,T)
        # 但 spectrogram_extractor 需要 (B,T)

        if wav.dim() == 3:          # (B,1,T)
            wav = wav.squeeze(1)
        elif wav.dim() == 4:        # (B,1,1,T)
            wav = wav.squeeze(1).squeeze(1)

        spec = self.m.spectrogram_extractor(wav)
        logmel = self.m.logmel_extractor(spec)

        return logmel.permute(0,1,3,2)  # (B,1,mel,time)



    def forward_from_logmel(self, logmel):
        x = logmel.permute(0,1,3,2)  # (B,1,time,mel)

        x = x.transpose(1,3)
        x = self.m.bn0(x)
        x = x.transpose(1,3)

        x = _call_conv_block(self.m.conv_block1, x, (2,2))
        x = _call_conv_block(self.m.conv_block2, x, (2,2))
        x = _call_conv_block(self.m.conv_block3, x, (2,2))
        x = _call_conv_block(self.m.conv_block4, x, (2,2))
        x = _call_conv_block(self.m.conv_block5, x, (2,2))
        x = _call_conv_block(self.m.conv_block6, x, (1,1))

        x = torch.mean(x, dim=3)
        x1,_ = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2

        x = torch.relu_(self.m.fc1(x))
        return self.m.fc_out(x)

    def forward(self, x):
        return self.forward_from_logmel(x)


# ============================================================
# Eval
# ============================================================
def eval_epoch(model, loader):

    model.eval()
    correct, total = 0, 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        out = model(x)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)

    return correct / total



# ============================================================
# PGD (2D)
# ============================================================
def pgd_attack_2d(model, X, y, eps=0.1, steps=20, batch_size=64):
    model.eval()
    X_adv = X.copy().astype(np.float32)
    step = eps / steps
    N = X.shape[0]

    for i in range(0, N, batch_size):
        xb0 = torch.tensor(X[i:i+batch_size], device=device)
        xb = torch.tensor(X_adv[i:i+batch_size], device=device, requires_grad=True)
        yb = torch.tensor(y[i:i+batch_size], device=device)

        for _ in range(steps):
            out = model(xb)
            loss = nn.CrossEntropyLoss()(out, yb)
            loss.backward()

            xb = xb + step * xb.grad.sign()
            xb = torch.max(torch.min(xb, xb0 + eps), xb0 - eps)
            xb = xb.detach().requires_grad_(True)

        X_adv[i:i+batch_size] = xb.detach().cpu().numpy()

    return X_adv


# ============================================================
# ROAD (2D)
# ============================================================
@torch.no_grad()
def road_spectrogram(loader, noise_std=0.1):

    X_all = []
    for xb, _ in tqdm(loader, desc="[ROAD] collect X"):
        X_all.append(xb.cpu().numpy())

    X_all = np.concatenate(X_all, axis=0).astype(np.float32)
    X = torch.from_numpy(X_all)

    N, C, H, W = X.shape
    wd, wi = 1/6, 1/12

    Xp = F.pad(X, (1,1,1,1), mode="reflect")

    up    = Xp[:,:,0:H,1:W+1]
    down  = Xp[:,:,2:H+2,1:W+1]
    left  = Xp[:,:,1:H+1,0:W]
    right = Xp[:,:,1:H+1,2:W+2]

    ul = Xp[:,:,0:H,0:W]
    ur = Xp[:,:,0:H,2:W+2]
    dl = Xp[:,:,2:H+2,0:W]
    dr = Xp[:,:,2:H+2,2:W+2]

    interp = wd*(up+down+left+right) + wi*(ul+ur+dl+dr)
    noise = noise_std * torch.randn_like(interp)

    return (interp + noise).numpy().astype(np.float32)


# ============================================================
# MoRF / LeRF
# ============================================================
def run_morf_lerf(model, X, y, saliency,
                  n_steps, batch_size, mode, X_adv=None):


    if saliency.ndim == 3:
        saliency = saliency[:,None,:,:]

    N, _, H, W = X.shape
    HW = H*W

    x = X.reshape(N,1,HW).copy()
    s = saliency.reshape(N,HW)

    order = np.argsort(-s,1) if mode=="morf" else np.argsort(s,1)
    step = max(1, HW//n_steps)

    zero = float(X.min())
    print("Replacement value:", zero, "x_mean:", X.mean(), "x_std:", X.std(), "x_min:", X.min(), "x_max:", X.max())
    adv_flat = X_adv.reshape(N,1,HW) if X_adv is not None else None

    acc_curve = []

    print(f"\n[{mode.upper()}] n_steps={n_steps}")

    for k in range(n_steps+1):

        if k>0:
            for i in range(N):
                idx = order[i,(k-1)*step:k*step]
                if adv_flat is None:
                    x[i,:,idx] = zero
                else:
                    x[i,:,idx] = adv_flat[i,:,idx]

        loader = DataLoader(
            TensorDataset(
                torch.tensor(x.reshape(N,1,H,W),dtype=torch.float32),
                torch.tensor(y,dtype=torch.long),
            ),
            batch_size=batch_size,
            shuffle=False,
        )

        acc = eval_epoch(model, loader)
        acc_curve.append(acc)

        print(f"[{mode}] step {k:02d}/{n_steps} acc={acc:.4f}")

    return np.array(acc_curve)


# ============================================================
# Main
# ============================================================
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", choices=["alexnet","cnn14"], default="alexnet")
    parser.add_argument("--data_dir",
        default="../experiment_utils/data/audiomnist/preprocessed_data")
    parser.add_argument("--split_txt",
        default="../experiment_utils/data/audiomnist/preprocessed_data/AudioNet_digit_0_test.txt")
    parser.add_argument("--ckpt",
        default="../experiment_utils/checkpoints/audiomnist/alexnet/best_model.pth")

    parser.add_argument("--expl_method", default="gradient_abs")
    parser.add_argument("--mask_type", choices=["zero","pgd","road"], default="zero")
    parser.add_argument("--eps", type=float, default=1)
    parser.add_argument("--pgd_steps", type=int, default=20)
    parser.add_argument("--n_steps", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)

    args = parser.parse_args()

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------
    test_loader = get_loader(
        args.split_txt,
        args.data_dir,
        args.model,
        args.batch_size,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    if args.model == "alexnet":
        model = AlexNet_Audio(num_classes=10)
    else:
        model = Cnn14(num_classes=10)

    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(state)

    model = model.to(device)
    model.eval()

    # --------------------------------------------------------
    # Collect input
    # --------------------------------------------------------
    if args.model == "cnn14":

        wrapper = CNN14LogmelWrapper(model).to(device)

        X_all, y_all = [], []
        for x, y in test_loader:
            x = ensure_wave_batch(x).to(device)
            with torch.no_grad():
                logmel = wrapper.waveform_to_logmel(x)
            X_all.append(logmel.cpu())
            y_all.append(y)

        X_input = torch.cat(X_all).numpy()
        y_input = torch.cat(y_all).numpy()

        model = wrapper

    else:
        X_all, y_all = [], []
        for x, y in test_loader:
            x = ensure_spec_batch(x)
            X_all.append(x)
            y_all.append(y)

        X_input = torch.cat(X_all).numpy()
        y_input = torch.cat(y_all).numpy()

    print("Input shape:", X_input.shape)

    # --------------------------------------------------------
    # Load saliency
    # --------------------------------------------------------
    sal_path = Path(PROJECT_ROOT) / "expl_audiomnist" / args.model / f"{args.expl_method}.npy"
    saliency = np.load(sal_path)
    print("Saliency shape:", saliency.shape)

    # --------------------------------------------------------
    # Mask type
    # --------------------------------------------------------
    X_adv = None
    cache_root = Path(PROJECT_ROOT) / "morf_lerf" / args.model / "audiomnist"


    if args.mask_type == "pgd":

        pgd_dir = cache_root / "pgd"
        pgd_dir.mkdir(parents=True, exist_ok=True)

        pgd_path = pgd_dir / f"pgd_eps{args.eps}.npy"

        if pgd_path.exists():
            print(f"[PGD] Loading cached PGD from {pgd_path}")
            X_adv = np.load(pgd_path)

        else:
            print("[PGD] Cache not found, running PGD attack once...")

            X_adv = pgd_attack_2d(
                model,
                X_input,
                y_input,
                eps=args.eps,
                steps=args.pgd_steps,
                batch_size=args.batch_size,
            )

            np.save(pgd_path, X_adv)
            print(f"[PGD] Saved PGD cache to {pgd_path}")

    elif args.mask_type == "road":

        noise_std = 0.1

        road_dir = cache_root / "road"
        road_dir.mkdir(parents=True, exist_ok=True)

        road_path = road_dir / f"road_{noise_std}.npy"

        if road_path.exists():
            print(f"[ROAD] Loading cached ROAD from {road_path}")
            X_adv = np.load(road_path)

        else:
            print("[ROAD] Cache not found, running ROAD once...")

            road_loader = DataLoader(
                TensorDataset(
                    torch.tensor(X_input, dtype=torch.float32),
                    torch.tensor(y_input, dtype=torch.long),
                ),
                batch_size=args.batch_size,
                shuffle=False,
            )

            X_adv = road_spectrogram(
                road_loader,
                noise_std=noise_std,
            )

            np.save(road_path, X_adv)
            print(f"[ROAD] Saved ROAD cache to {road_path}")
    # --------------------------------------------------------
    # MoRF / LeRF
    # --------------------------------------------------------
    morf = run_morf_lerf(
        model, X_input, y_input,
        saliency, args.n_steps, args.batch_size,
        "morf", X_adv
    )

    lerf = run_morf_lerf(
        model, X_input, y_input,
        saliency, args.n_steps, args.batch_size,
        "lerf", X_adv
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------
    out_dir = Path(PROJECT_ROOT) / "morf_lerf" / args.model / "audiomnist" / args.mask_type
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / f"{args.expl_method}_morf.pkl","wb") as f:
        pickle.dump(morf,f)

    with open(out_dir / f"{args.expl_method}_lerf.pkl","wb") as f:
        pickle.dump(lerf,f)

    print("Saved to:", out_dir)


if __name__ == "__main__":
    main()
