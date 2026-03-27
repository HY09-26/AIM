# #test_msos_spectrogram.py
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
# # MSOS waveform loader
# # ============================================================
# def load_h5_wave(path):
#     with h5py.File(path, "r") as f:
#         x = f["data"][:]               # (1,1,1,T)
#         y = int(f["label"][0][0])
#     x = np.asarray(x, np.float32).reshape(-1)  # (T,)
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
# # CNN14 log-mel helper (same as ESC50)
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
# # PGD/Road (log-mel space)
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
#             xb = torch.max(
#                 torch.min(xb, torch.tensor(X[i:i+batch_size], device=device) + eps),
#                 torch.tensor(X[i:i+batch_size], device=device) - eps
#             )
#             xb = xb.detach().requires_grad_(True)

#         X_adv[i:i+batch_size] = xb.detach().cpu().numpy()

#     return X_adv


# @torch.no_grad()
# def road_spectrogram(
#     loader,        # yields (x, y), x: (B,1,mel,time)
#     noise_std=0.05,
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
#     parser.add_argument("--data_dir", default="../experiment_utils/data/MSoS/preprocessed_data")
#     parser.add_argument("--split_txt", default="../experiment_utils/data/MSoS/preprocessed_data/MSOS_test.txt")
#     parser.add_argument("--ckpt", default="../experiment_utils/checkpoints/msos/cnn14/best_model.pth")
#     parser.add_argument("--expl_method", default="gradinput") 
#     parser.add_argument("--mask_type", choices=["zero", "pgd", "road"], default="road")
#     parser.add_argument("--eps", type=float, default=0.45)
#     parser.add_argument("--pgd_steps", type=int, default=20)
#     parser.add_argument("--n_steps", type=int, default=20)
#     parser.add_argument("--batch_size", type=int, default=32)
#     args = parser.parse_args()

#     # ---------- data ----------
#     X_wave, y = load_test_set(args.split_txt, args.data_dir)
#     X_wave = torch.tensor(X_wave, device=device)

#     # ---------- model ----------
#     cnn14 = Cnn14(num_classes=5).to(device)
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
#         Path(PROJECT_ROOT) / "expl_msos" / "cnn14" / f"{args.expl_method}.npy"
#     )

#     # ---------- PGD (optional) ----------
#     X_adv = None

#     if args.mask_type == "pgd":
#         pgd_dir = Path(PROJECT_ROOT) / "morf_lerf" / "cnn14" / "msos" / "pgd"
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
#         road_dir = Path(PROJECT_ROOT) / "morf_lerf" / "cnn14" / "msos" / "road"
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
#     out_dir = Path(PROJECT_ROOT) / "morf_lerf" / "cnn14" / "msos" / args.mask_type
#     out_dir.mkdir(parents=True, exist_ok=True)

#     with open(out_dir / f"{args.expl_method}_morf.pkl", "wb") as f:
#         pickle.dump(morf, f)
#     with open(out_dir / f"{args.expl_method}_lerf.pkl", "wb") as f:
#         pickle.dump(lerf, f)

#     print("Saved to:", out_dir)


# if __name__ == "__main__":
#     main()












# experiment/test_msos_spectrogram.py

import os
import sys
import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ------------------------------------------------------------
# Path
# ------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)

from experiment_utils.model.alexnet import AlexNet_Audio
from experiment_utils.model.cnn14 import Cnn14
from experiment_utils.expl_gen_msos_plus import MSOSDatasetForExpl, CNN14LogmelWrapper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------
# Eval
# ------------------------------------------------------------
@torch.no_grad()
def eval_epoch(model, loader):
    model.eval()
    correct, total = 0, 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)

    return correct / total


# ------------------------------------------------------------
# PGD
# ------------------------------------------------------------
def pgd_attack(model, loader, eps=0.45, steps=20):
    """
    Important:
      - alexnet loader yields (B,1,F,T)
      - cnn14  loader MUST yield (B,1,mel,time)  (logmel space)
    """
    model.eval()
    adv_list = []
    step = eps / steps
    loss_fn = nn.CrossEntropyLoss()

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        x0 = x.clone().detach()
        x = x.clone().detach().requires_grad_(True)

        for _ in range(steps):
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()

            x = x + step * x.grad.sign()
            x = torch.max(torch.min(x, x0 + eps), x0 - eps)
            x = x.detach().requires_grad_(True)

        adv_list.append(x.detach().cpu())

    return torch.cat(adv_list, dim=0).numpy()


# ------------------------------------------------------------
# ROAD
# ------------------------------------------------------------
@torch.no_grad()
def road_input(X_np, noise_std=0.1):
    X = torch.from_numpy(X_np)
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

    return (interp + noise).numpy()


# ------------------------------------------------------------
# MoRF / LeRF
# ------------------------------------------------------------
def run_morf_lerf(model, X, y, saliency, mode, n_steps, batch_size, X_adv=None):

    N, C, H, W = X.shape
    HW = H * W

    x = X.reshape(N, C, HW).copy()
    s = saliency.reshape(N, HW)

    order = np.argsort(-s, 1) if mode == "morf" else np.argsort(s, 1)
    step = max(1, HW // n_steps)
    zero = float(X.min())
    print(f"\nzero_value={zero:.4f}", f"x_mean={X.mean():.4f}", f"x_std={X.std():.4f}")

    acc_curve = []

    for k in range(n_steps + 1):

        if k > 0:
            for i in range(N):
                idx = order[i, (k - 1)*step : k*step]
                if X_adv is None:
                    x[i, :, idx] = zero
                else:
                    xadv = X_adv.reshape(N, C, HW)
                    x[i, :, idx] = xadv[i, :, idx]

        loader = DataLoader(
            list(zip(
                torch.tensor(x.reshape(N, C, H, W), dtype=torch.float32),
                torch.tensor(y, dtype=torch.long)
            )),
            batch_size=batch_size,
            shuffle=False
        )

        acc = eval_epoch(model, loader)
        acc_curve.append(acc)
        print(f"[{mode.upper()}] step {k}/{n_steps} | acc={acc:.4f}")

    return np.array(acc_curve)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["alexnet", "cnn14"], default="alexnet")
    parser.add_argument("--data_dir", default="../experiment_utils/data/MSoS/preprocessed_data")
    parser.add_argument("--split_txt", default="../experiment_utils/data/MSoS/preprocessed_data/MSOS_test.txt")
    parser.add_argument("--mask_type", choices=["zero", "pgd", "road"], default="zero")
    parser.add_argument("--expl_method", default="random")
    parser.add_argument("--eps", type=float, default=0.015)
    parser.add_argument("--pgd_steps", type=int, default=20)
    parser.add_argument("--n_steps", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--road_noise", type=float, default=0.1)
    args = parser.parse_args()

    NUM_CLASSES = 5

    # --------------------------------------------------------
    # Loader (matches expl_gen_msos_plus)
    # --------------------------------------------------------
    test_dataset = MSOSDatasetForExpl(args.split_txt, args.data_dir, args.model)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    if args.model == "alexnet":

        model = AlexNet_Audio(num_classes=NUM_CLASSES).to(device)
        ckpt_path = Path(PROJECT_ROOT) / "experiment_utils" / "checkpoints" / "msos" / "alexnet" / "best_model.pth"

        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)
        model.eval()

        logmel_loader = None  # not used
        input_loader = test_loader

    else:  # cnn14

        base = Cnn14(num_classes=NUM_CLASSES).to(device)
        ckpt_path = Path(PROJECT_ROOT) / "experiment_utils" / "checkpoints" / "msos" / "cnn14" / "best_model.pth"
        ckpt = torch.load(ckpt_path, map_location="cpu")

        base.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)
        base.eval()

        model = CNN14LogmelWrapper(base).to(device)
        model.eval()

        # IMPORTANT: build logmel loader (model input space)
        logmel_list, label_list = [], []
        for wav, yb in test_loader:  # wav: (B,1,T)
            wav = wav.to(device)
            with torch.no_grad():
                logmel = model.helper.waveform_to_logmel(wav)  # -> 4D logmel
            logmel_list.append(logmel.cpu())
            label_list.append(yb)

        logmel_all = torch.cat(logmel_list, dim=0)
        y_all = torch.cat(label_list, dim=0)

        logmel_loader = DataLoader(
            list(zip(logmel_all, y_all)),
            batch_size=args.batch_size,
            shuffle=False
        )

        # use logmel_loader for EVERYTHING downstream (sanity / PGD / X collection)
        input_loader = logmel_loader

    print("Loaded ckpt best_acc:", ckpt.get("best_acc"))

    # --------------------------------------------------------
    # Sanity check (STEP 0) — ALWAYS in model input space
    # --------------------------------------------------------
    sanity_acc = eval_epoch(model, input_loader)
    print("Sanity step0 accuracy:", sanity_acc)

    # --------------------------------------------------------
    # Collect full input (X) in model input space
    # --------------------------------------------------------
    X_list, y_list = [], []
    for x, yb in input_loader:
        X_list.append(x)
        y_list.append(yb)

    X = torch.cat(X_list, dim=0).numpy()
    y = torch.cat(y_list, dim=0).numpy()

    # --------------------------------------------------------
    # Load saliency
    # --------------------------------------------------------
    sal_path = Path(PROJECT_ROOT) / "expl_msos" / args.model / f"{args.expl_method}.npy"
    saliency = np.load(sal_path)

    # CNN14 transpose correction (keep this if your saved expl is (N,1,time,mel))
    if args.model == "cnn14":
        saliency = saliency.transpose(0, 1, 3, 2)  # -> (N,1,mel,time) to match X

    # --------------------------------------------------------
    # PGD / ROAD (cache)
    # --------------------------------------------------------
    X_adv = None

    if args.mask_type == "pgd":
        pgd_dir = Path(PROJECT_ROOT) / "morf_lerf" / args.model / "msos" / "pgd_cache"
        pgd_dir.mkdir(parents=True, exist_ok=True)
        pgd_path = pgd_dir / f"pgd_eps{args.eps}_steps{args.pgd_steps}.npy"

        if pgd_path.exists():
            print(f"[PGD] Loading cached PGD from: {pgd_path}")
            X_adv = np.load(pgd_path)
        else:
            print("[PGD] Cache not found, running PGD once...")
            X_adv = pgd_attack(
                model,
                input_loader,   # IMPORTANT: cnn14 uses logmel_loader
                eps=args.eps,
                steps=args.pgd_steps
            )
            np.save(pgd_path, X_adv)
            print(f"[PGD] Saved PGD to: {pgd_path}")

    elif args.mask_type == "road":
        road_dir = Path(PROJECT_ROOT) / "morf_lerf" / args.model / "msos" / "road_cache"
        road_dir.mkdir(parents=True, exist_ok=True)
        road_path = road_dir / f"road_noise{args.road_noise}.npy"

        if road_path.exists():
            print(f"[ROAD] Loading cached ROAD from: {road_path}")
            X_adv = np.load(road_path)
        else:
            print("[ROAD] Cache not found, running ROAD once...")
            torch.manual_seed(0)
            np.random.seed(0)
            X_adv = road_input(X, noise_std=args.road_noise)
            np.save(road_path, X_adv)
            print(f"[ROAD] Saved ROAD to: {road_path}")

    # --------------------------------------------------------
    # MoRF / LeRF
    # --------------------------------------------------------
    morf = run_morf_lerf(model, X, y, saliency, "morf", args.n_steps, args.batch_size, X_adv)
    lerf = run_morf_lerf(model, X, y, saliency, "lerf", args.n_steps, args.batch_size, X_adv)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------
    out_dir = Path(PROJECT_ROOT) / "morf_lerf" / args.model / "msos" / args.mask_type
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / f"{args.expl_method}_morf.pkl", "wb") as f:
        pickle.dump(morf, f)
    with open(out_dir / f"{args.expl_method}_lerf.pkl", "wb") as f:
        pickle.dump(lerf, f)

    print("Saved to:", out_dir)


if __name__ == "__main__":
    main()
