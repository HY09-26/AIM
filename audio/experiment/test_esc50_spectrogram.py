# #test_esc50_spectrogram.py
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

# from experiment_utils.model.cnn14 import Cnn14

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# # ============================================================
# # ESC-50 waveform loader
# # ============================================================
# def load_h5_wave(path):
#     with h5py.File(path, "r") as f:
#         x = f["data"][:]
#         y = int(f["label"][0][0])
#     x = np.asarray(x, np.float32).reshape(-1)
#     return x, y


# def load_test_set(split_txt, data_dir):
#     xs, ys = [], []
#     with open(split_txt, "r") as f:
#         for line in f:
#             p = line.strip()
#             if not p or "waveform" not in p:
#                 continue
#             full = os.path.join(data_dir, p)
#             x, y = load_h5_wave(full)
#             xs.append(x)
#             ys.append(y)
#     return np.stack(xs, 0), np.array(ys, np.int64)


# # ============================================================
# # CNN14 log-mel helper 
# # ============================================================
# def _call_conv_block(conv_block, x, pool_size):
#     try:
#         return conv_block(x, pool_size=pool_size, pool_type="avg")
#     except TypeError:
#         return conv_block(x, pool_size=pool_size)


# class CNN14LogmelHelper:
#     def __init__(self, model):
#         self.m = model

#     @torch.no_grad()
#     def waveform_to_logmel(self, wav):
#         spec = self.m.spectrogram_extractor(wav)
#         logmel = self.m.logmel_extractor(spec)
#         return logmel.permute(0, 1, 3, 2)  # (B,1,mel,time)

#     def forward_from_logmel(self, logmel_mt):
#         x = logmel_mt.permute(0, 1, 3, 2)  # -> (B,1,time,mel)

#         x = x.transpose(1, 3)
#         x = self.m.bn0(x)
#         x = x.transpose(1, 3)

#         x = _call_conv_block(self.m.conv_block1, x, (2, 2))
#         x = _call_conv_block(self.m.conv_block2, x, (2, 2))
#         x = _call_conv_block(self.m.conv_block3, x, (2, 2))
#         x = _call_conv_block(self.m.conv_block4, x, (2, 2))
#         x = _call_conv_block(self.m.conv_block5, x, (2, 2))
#         x = _call_conv_block(self.m.conv_block6, x, (1, 1))

#         x = torch.mean(x, dim=3)
#         x1, _ = torch.max(x, dim=2)
#         x2 = torch.mean(x, dim=2)
#         x = x1 + x2

#         x = torch.relu_(self.m.fc1(x))
#         return self.m.fc_out(x)


# class CNN14SpectrogramWrapper(nn.Module):
#     def __init__(self, cnn14):
#         super().__init__()
#         self.helper = CNN14LogmelHelper(cnn14)

#     def forward(self, x):
#         return self.helper.forward_from_logmel(x)


# # ============================================================
# # Evaluation
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
# # PGD/Road(log-mel space)
# # ============================================================
# def pgd_attack_logmel(
#     model,
#     X,
#     y,
#     eps=0.01,
#     steps=20,
#     batch_size=32,
# ):
#     """
#     X: (N,1,mel,time)
#     """
#     model.eval()
#     X_adv = X.copy()

#     step = eps / steps
#     N = X.shape[0]

#     for i in range(0, N, batch_size):
#         xb = torch.tensor(X_adv[i:i+batch_size], device=device, requires_grad=True)
#         yb = torch.tensor(y[i:i+batch_size], device=device)

#         for _ in range(steps):
#             out = model(xb)
#             loss = nn.CrossEntropyLoss()(out, yb)
#             loss.backward()

#             xb = xb + step * xb.grad.sign()
#             xb = torch.max(torch.min(xb, torch.tensor(X[i:i+batch_size], device=device) + eps),
#                            torch.tensor(X[i:i+batch_size], device=device) - eps)
#             xb = xb.detach().requires_grad_(True)

#         X_adv[i:i+batch_size] = xb.detach().cpu().numpy()

#     return X_adv

# @torch.no_grad()
# def road_spectrogram(
#     loader,        # yields (x, y), x: (B,1,mel,time)
#     noise_std=0.2,
# ):
#     """
#     Spectrogram ROAD (Noisy Linear Imputation)
#     Return: (N,1,mel,time) np.float32
#     """

#     X_all = []
#     for xb, _ in tqdm(loader, desc="[ROAD-SPEC] collect X"):
#         X_all.append(xb.cpu().numpy())

#     X_all = np.concatenate(X_all, axis=0).astype(np.float32)
#     X = torch.from_numpy(X_all)   # CPU tensor

#     N, C, M, T = X.shape

#     wd, wi = 1 / 6, 1 / 12

#     # pad on (mel, time)
#     Xp = F.pad(X, (1, 1, 1, 1), mode="reflect")

#     up    = Xp[:, :, 0:M,     1:T+1]
#     down  = Xp[:, :, 2:M+2,   1:T+1]
#     left  = Xp[:, :, 1:M+1,   0:T]
#     right = Xp[:, :, 1:M+1,   2:T+2]

#     ul = Xp[:, :, 0:M,     0:T]
#     ur = Xp[:, :, 0:M,     2:T+2]
#     dl = Xp[:, :, 2:M+2,   0:T]
#     dr = Xp[:, :, 2:M+2,   2:T+2]

#     interp = wd * (up + down + left + right) \
#            + wi * (ul + ur + dl + dr)

#     noise = noise_std * torch.randn_like(interp)

#     X_road = (interp + noise).numpy().astype(np.float32)
#     return X_road

# # ============================================================
# # MoRF / LeRF
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
#     N, _, M, T = X.shape
#     HW = M * T

#     x = X.reshape(N, 1, HW).copy()
#     s = saliency.reshape(N, HW)

#     order = np.argsort(-s, 1) if mode == "morf" else np.argsort(s, 1)
#     step = max(1, HW // n_steps)
#     zero = float(X.min())
#     print('zero value:', zero)

#     acc_curve = []

#     print(f"\n[{mode.upper()}] n_steps={n_steps}, step_size={step}")

#     for k in range(n_steps + 1):

#         if k > 0:
#             for i in range(N):
#                 idx = order[i, (k-1)*step:k*step]
#                 if X_adv is None:
#                     x[i, :, idx] = zero
#                 else:
#                     x[i, :, idx] = X_adv.reshape(N, 1, HW)[i, :, idx]

#         loader = DataLoader(
#             TensorDataset(
#                 torch.tensor(x.reshape(N, 1, M, T), dtype=torch.float32),
#                 torch.tensor(y, dtype=torch.long),
#             ),
#             batch_size=batch_size,
#             shuffle=False,
#         )

#         acc = eval_epoch(model, loader)
#         acc_curve.append(acc)

#         masked_pct = 100 * k / n_steps
#         print(f"[{mode.upper()}] step {k:02d}/{n_steps} | masked={masked_pct:5.1f}% | acc={acc:.4f}")

#     return np.array(acc_curve)




# # ============================================================
# # Main
# # ============================================================
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--data_dir", default="../experiment_utils/data/ESC50/preprocessed_data")
#     parser.add_argument("--split_txt", default="../experiment_utils/data/ESC50/preprocessed_data/ESC50_fold1_test.txt")
#     parser.add_argument("--ckpt", default="../experiment_utils/checkpoints/esc50/cnn14/fold_1/best_model.pth")
#     parser.add_argument("--fold", type=int, default=1)
#     parser.add_argument("--expl_method", default="gradient_abs")
#     parser.add_argument("--mask_type", choices=["zero", "pgd", "road"], default="road")
#     parser.add_argument("--eps", type=float, default=0.75)
#     parser.add_argument("--pgd_steps", type=int, default=20)
#     parser.add_argument("--n_steps", type=int, default=20)
#     parser.add_argument("--batch_size", type=int, default=32)
#     args = parser.parse_args()

#     # ---------- data ----------
#     X_wave, y = load_test_set(args.split_txt, args.data_dir)
#     X_wave = torch.tensor(X_wave, device=device)

#     # ---------- model ----------
#     cnn14 = Cnn14(num_classes=50).to(device)
#     ckpt = torch.load(args.ckpt, map_location="cpu")
#     cnn14.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)
#     cnn14.eval()

#     wrapper = CNN14SpectrogramWrapper(cnn14).to(device)

#     # ---------- log-mel ----------
#     helper = wrapper.helper
#     with torch.no_grad():
#         X_logmel = helper.waveform_to_logmel(X_wave).cpu().numpy()

#     # ---------- saliency ----------
#     saliency = np.load(
#         Path(PROJECT_ROOT) / "expl_esc50" / "cnn14" / f"fold_{args.fold}"/f"{args.expl_method}.npy"
#     )

#     # ---------- PGD (optional) ----------
#     X_adv = None

#     if args.mask_type == "pgd":
#         pgd_dir = Path(PROJECT_ROOT) / "morf_lerf" / "cnn14" / "esc50" / "pgd"
#         pgd_dir.mkdir(parents=True, exist_ok=True)

#         pgd_path = pgd_dir / f"pgd_eps{args.eps}.npy"

#         if pgd_path.exists():
#             print(f"[PGD] Loading cached PGD from {pgd_path}")
#             X_adv = np.load(pgd_path)

#         else:
#             print("[PGD] Cache not found, running PGD attack once...")
#             X_adv = pgd_attack_logmel(
#                 wrapper,
#                 X_logmel,
#                 y,
#                 eps=args.eps,
#                 steps=args.pgd_steps,
#                 batch_size=args.batch_size,
#             )
#             np.save(pgd_path, X_adv)
#             print(f"[PGD] Saved PGD cache to {pgd_path}")

#     # ---------- ROAD (optional) ----------
#     elif args.mask_type == "road":
#         noise_std=0.1
#         road_dir = Path(PROJECT_ROOT) / "morf_lerf" / "cnn14" / "esc50" / "road"
#         road_dir.mkdir(parents=True, exist_ok=True)

#         road_path = road_dir / f"road_{noise_std}.npy"

#         if road_path.exists():
#             print(f"[ROAD] Loading cached ROAD from {road_path}")
#             X_road = np.load(road_path)

#         else:
#             print("[ROAD] Cache not found, running ROAD once...")
#             road_loader = DataLoader(
#                 TensorDataset(
#                     torch.tensor(X_logmel, dtype=torch.float32),
#                     torch.tensor(y, dtype=torch.long),
#                 ),
#                 batch_size=args.batch_size,
#                 shuffle=False,
#             )

#             X_road = road_spectrogram(
#                 road_loader,
#                 noise_std=noise_std,
#             )
#             np.save(road_path, X_road)
#             print(f"[ROAD] Saved ROAD cache to {road_path}")

#         X_adv = X_road

#     # ---------- MoRF / LeRF ----------
#     morf = run_morf_lerf(
#         wrapper, X_logmel, y, saliency,
#         args.n_steps, args.batch_size, "morf", X_adv
#     )
#     lerf = run_morf_lerf(
#         wrapper, X_logmel, y, saliency,
#         args.n_steps, args.batch_size, "lerf", X_adv
#     )

#     # ---------- save ----------
#     out_dir = Path(PROJECT_ROOT) / "morf_lerf" / "cnn14" / "esc50" / args.mask_type
#     out_dir.mkdir(parents=True, exist_ok=True)

#     with open(out_dir / f"{args.expl_method}_morf.pkl", "wb") as f:
#         pickle.dump(morf, f)
#     with open(out_dir / f"{args.expl_method}_lerf.pkl", "wb") as f:
#         pickle.dump(lerf, f)

#     print("Saved to:", out_dir)


# if __name__ == "__main__":
#     main()








# test_esc50_spectrogram.py
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
import torch.nn.functional as F
from tqdm import tqdm


# ============================================================
# Path
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)

from experiment_utils.model.cnn14 import Cnn14
from experiment_utils.model.alexnet import AlexNet_Audio

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# H5 loaders
# ============================================================
def load_h5_wave(path):
    with h5py.File(path, "r") as f:
        x = f["data"][:]            # (1,1,1,T)
        y = int(f["label"][0][0])
    x = np.asarray(x, np.float32).reshape(-1)  # (T,)
    return x, y


def load_h5_spec(path):
    with h5py.File(path, "r") as f:
        x = f["data"][0, 0]         # (F,T)
        y = int(f["label"][0][0])
    x = np.asarray(x, np.float32)

    # normalize (match your train/expl_gen behavior)
    mean = x.mean()
    std = x.std()
    x = (x - mean) / (std + 1e-6)

    return x, y


def load_test_set_wave(split_txt, data_dir):
    xs, ys = [], []
    with open(split_txt, "r") as f:
        for line in f:
            p = line.strip()
            if not p or "waveform" not in p:
                continue
            full = os.path.join(data_dir, p)
            x, y = load_h5_wave(full)
            xs.append(x)
            ys.append(y)
    return np.stack(xs, 0), np.array(ys, np.int64)  # (N,T), (N,)


def load_test_set_spec(split_txt, data_dir):
    xs, ys = [], []
    with open(split_txt, "r") as f:
        for line in f:
            p = line.strip()
            if not p or "spectrogram" not in p:
                continue
            full = os.path.join(data_dir, p)
            x, y = load_h5_spec(full)  # (F,T)
            xs.append(x)
            ys.append(y)
    X = np.stack(xs, 0).astype(np.float32)            # (N,F,T)
    X = X[:, None, :, :]                              # (N,1,F,T)
    return X, np.array(ys, np.int64)


# ============================================================
# CNN14 log-mel helper
# ============================================================
def _call_conv_block(conv_block, x, pool_size):
    try:
        return conv_block(x, pool_size=pool_size, pool_type="avg")
    except TypeError:
        return conv_block(x, pool_size=pool_size)


class CNN14LogmelHelper:
    def __init__(self, model):
        self.m = model

    @torch.no_grad()
    def waveform_to_logmel(self, wav):
        # wav: (N,T) torch on device
        spec = self.m.spectrogram_extractor(wav)
        logmel = self.m.logmel_extractor(spec)
        return logmel.permute(0, 1, 3, 2)  # (B,1,mel,time)

    def forward_from_logmel(self, logmel_mt):
        # logmel_mt: (B,1,mel,time)
        x = logmel_mt.permute(0, 1, 3, 2)  # -> (B,1,time,mel)

        x = x.transpose(1, 3)
        x = self.m.bn0(x)
        x = x.transpose(1, 3)

        x = _call_conv_block(self.m.conv_block1, x, (2, 2))
        x = _call_conv_block(self.m.conv_block2, x, (2, 2))
        x = _call_conv_block(self.m.conv_block3, x, (2, 2))
        x = _call_conv_block(self.m.conv_block4, x, (2, 2))
        x = _call_conv_block(self.m.conv_block5, x, (2, 2))
        x = _call_conv_block(self.m.conv_block6, x, (1, 1))

        x = torch.mean(x, dim=3)
        x1, _ = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2

        x = torch.relu_(self.m.fc1(x))
        return self.m.fc_out(x)


class CNN14SpectrogramWrapper(nn.Module):
    def __init__(self, cnn14):
        super().__init__()
        self.helper = CNN14LogmelHelper(cnn14)

    def forward(self, x):
        # x: (B,1,mel,time)
        return self.helper.forward_from_logmel(x)


# ============================================================
# Evaluation
# ============================================================
@torch.no_grad()
def eval_epoch(model, loader):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = model(x)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / total


# ============================================================
# PGD (2D input space, works for logmel/spec)
# ============================================================
def pgd_attack_2d(
    model,
    X,
    y,
    eps=0.01,
    steps=20,
    batch_size=32,
):
    """
    X: (N,1,H,W) numpy
    y: (N,) numpy
    """
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
def road_spectrogram(
    loader,        # yields (x, y), x: (B,1,H,W)
    noise_std=0.2,
):
    """
    2D ROAD (Noisy Linear Imputation) for spectrogram/log-mel
    Return: (N,1,H,W) np.float32
    """
    X_all = []
    for xb, _ in tqdm(loader, desc="[ROAD-2D] collect X"):
        X_all.append(xb.cpu().numpy())

    X_all = np.concatenate(X_all, axis=0).astype(np.float32)
    X = torch.from_numpy(X_all)   # CPU tensor

    N, C, H, W = X.shape
    wd, wi = 1 / 6, 1 / 12

    Xp = F.pad(X, (1, 1, 1, 1), mode="reflect")

    up    = Xp[:, :, 0:H,     1:W+1]
    down  = Xp[:, :, 2:H+2,   1:W+1]
    left  = Xp[:, :, 1:H+1,   0:W]
    right = Xp[:, :, 1:H+1,   2:W+2]

    ul = Xp[:, :, 0:H,     0:W]
    ur = Xp[:, :, 0:H,     2:W+2]
    dl = Xp[:, :, 2:H+2,   0:W]
    dr = Xp[:, :, 2:H+2,   2:W+2]

    interp = wd * (up + down + left + right) + wi * (ul + ur + dl + dr)
    noise = noise_std * torch.randn_like(interp)

    X_road = (interp + noise).numpy().astype(np.float32)
    return X_road


# ============================================================
# MoRF / LeRF
# ============================================================
def run_morf_lerf(
    model,
    X,          # (N,1,H,W)
    y,          # (N,)
    saliency,   # (N,1,H,W) or (N,H,W)
    n_steps,
    batch_size,
    mode,
    X_adv=None, # (N,1,H,W) if provided
):
    if saliency.ndim == 4:
        s = saliency
    elif saliency.ndim == 3:
        s = saliency[:, None, :, :]
    else:
        raise ValueError(f"Unexpected saliency shape: {saliency.shape}")

    N, _, H, W = X.shape
    HW = H * W

    x = X.reshape(N, 1, HW).copy()
    s = s.reshape(N, HW)

    order = np.argsort(-s, 1) if mode == "morf" else np.argsort(s, 1)
    step = max(1, HW // n_steps)

    # for "zero" masking baseline
    zero = float(X.min())
    print(f"\nzero_value={zero:.4f}", f"x_mean={X.mean():.4f}", f"x_std={X.std():.4f}", f"x_max={X.max():.4f}")

    acc_curve = []
    print(f"\n[{mode.upper()}] n_steps={n_steps}, step_size={step}")

    adv_flat = None
    if X_adv is not None:
        adv_flat = X_adv.reshape(N, 1, HW)

    for k in range(n_steps + 1):
        if k > 0:
            for i in range(N):
                idx = order[i, (k-1)*step:k*step]
                if adv_flat is None:
                    x[i, :, idx] = zero
                else:
                    x[i, :, idx] = adv_flat[i, :, idx]

        loader = DataLoader(
            TensorDataset(
                torch.tensor(x.reshape(N, 1, H, W), dtype=torch.float32),
                torch.tensor(y, dtype=torch.long),
            ),
            batch_size=batch_size,
            shuffle=False,
        )

        acc = eval_epoch(model, loader)
        acc_curve.append(acc)

        masked_pct = 100 * k / n_steps
        print(f"[{mode.upper()}] step {k:02d}/{n_steps} | masked={masked_pct:5.1f}% | acc={acc:.4f}")

    return np.array(acc_curve)


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["cnn14", "alexnet"], default="cnn14")

    parser.add_argument("--data_dir", default="../experiment_utils/data/ESC50/preprocessed_data")
    parser.add_argument("--split_txt", default="../experiment_utils/data/ESC50/preprocessed_data/ESC50_fold1_test.txt")

    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--expl_method", default="gradient_abs")
    parser.add_argument("--mask_type", choices=["zero", "pgd", "road"], default="pgd")

    # ckpt paths
    parser.add_argument("--ckpt_cnn14", default="../experiment_utils/checkpoints/esc50/cnn14/fold_1/best_model.pth")
    parser.add_argument("--ckpt_alexnet", default="../experiment_utils/checkpoints/esc50/alexnet/fold_1/best_model.pth")

    # mask params
    parser.add_argument("--eps", type=float, default=0.75)
    parser.add_argument("--pgd_steps", type=int, default=20)
    parser.add_argument("--n_steps", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    # =======================================================
    # Build model + input X + saliency path
    # =======================================================
    if args.model == "cnn14":
        # ---------- data ----------
        X_wave, y = load_test_set_wave(args.split_txt, args.data_dir)   # (N,T)
        X_wave = torch.tensor(X_wave, device=device)

        # ---------- model ----------
        cnn14 = Cnn14(num_classes=50).to(device)
        ckpt = torch.load(args.ckpt_cnn14, map_location="cpu")
        cnn14.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)
        cnn14.eval()

        wrapper = CNN14SpectrogramWrapper(cnn14).to(device)

        # ---------- log-mel ----------
        helper = wrapper.helper
        with torch.no_grad():
            X_2d = helper.waveform_to_logmel(X_wave).cpu().numpy()  # (N,1,mel,time)

        eval_model = wrapper

        sal_path = Path(PROJECT_ROOT) / "expl_esc50" / "cnn14" / f"fold_{args.fold}" / f"{args.expl_method}.npy"

        cache_root = Path(PROJECT_ROOT) / "morf_lerf" / "cnn14" / "esc50"

    else:
        # ---------- data ----------
        X_2d, y = load_test_set_spec(args.split_txt, args.data_dir)     # (N,1,F,T)

        # ---------- model ----------
        alex = AlexNet_Audio(num_classes=50).to(device)
        ckpt = torch.load(args.ckpt_alexnet, map_location="cpu")
        alex.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)
        alex.eval()

        eval_model = alex

        sal_path = Path(PROJECT_ROOT) / "expl_esc50" / "alexnet" / f"fold_{args.fold}" / f"{args.expl_method}.npy"

        cache_root = Path(PROJECT_ROOT) / "morf_lerf" / "alexnet" / "esc50"

    # ---------- saliency ----------
    saliency = np.load(sal_path)
    # allow either (N,1,H,W) or (N,H,W)
    if saliency.ndim == 3:
        saliency = saliency[:, None, :, :]

    # =======================================================
    # Mask cache: PGD / ROAD
    # =======================================================
    X_adv = None

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
                eval_model, X_2d, y,
                eps=args.eps, steps=args.pgd_steps, batch_size=args.batch_size
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
                    torch.tensor(X_2d, dtype=torch.float32),
                    torch.tensor(y, dtype=torch.long),
                ),
                batch_size=args.batch_size,
                shuffle=False,
            )
            X_adv = road_spectrogram(road_loader, noise_std=noise_std)
            np.save(road_path, X_adv)
            print(f"[ROAD] Saved ROAD cache to {road_path}")

    # =======================================================
    # MoRF / LeRF
    # =======================================================
    morf = run_morf_lerf(
        eval_model, X_2d, y, saliency,
        args.n_steps, args.batch_size, "morf", X_adv
    )
    lerf = run_morf_lerf(
        eval_model, X_2d, y, saliency,
        args.n_steps, args.batch_size, "lerf", X_adv
    )

    # =======================================================
    # Save
    # =======================================================
    out_dir = cache_root / args.mask_type
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / f"{args.expl_method}_morf.pkl", "wb") as f:
        pickle.dump(morf, f)
    with open(out_dir / f"{args.expl_method}_lerf.pkl", "wb") as f:
        pickle.dump(lerf, f)

    print("Saved to:", out_dir)


if __name__ == "__main__":
    main()

