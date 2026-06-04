# ============================================================
# test_imagenet.py
#
# ImageNet MoRF / LeRF evaluation
# - Fixed ImageNet val 500 subset
# - Pretrained models (ResNet50 / EfficientNet-B0 / RepVGG-B0)
# - Zero or PGD masking (PGD cached once per model)
# ============================================================

import os
import argparse
import pickle
import numpy as np

import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset, Subset
from tqdm import tqdm

import timm
from experiment_utils.utils import pgd_attack, road
pgd = pgd_attack


# ============================================================
# Utilities
# ============================================================
def build_loader_from_np(X_np, y_np, batch_size):
    x = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.long)
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


# ============================================================
# MoRF / LeRF (Image)
# ============================================================
def run_morf_lerf_image(
    model,
    device,
    X_np,           # (N,3,H,W)
    y_np,           # (N,)
    saliency_np,    # (N,3,H,W)
    mask_np=None,   # (N,3,H,W) or None
    n_steps=20,
    batch_size=64,
    mask_type="zero",   # "zero" | "pgd" | "road"
    mode="morf",        # "morf" | "lerf"
):
    model.eval()

    N, C, H, W = X_np.shape
    HW = H * W

    X_flat = X_np.reshape(N, C, HW).copy()
    mask_flat = mask_np.reshape(N, C, HW) if mask_np is not None else None

    # ---------------- saliency ranking ----------------
    sal = np.abs(saliency_np).sum(axis=1).reshape(N, HW)

    if mode == "morf":
        order = np.argsort(-sal, axis=1)
    elif mode == "lerf":
        order = np.argsort(+sal, axis=1)
    else:
        raise ValueError(mode)

    pixels_per_step = max(1, HW // n_steps)

    # ---------------- eval helper ----------------
    def eval_step(X_flat_step):
        X = X_flat_step.reshape(N, C, H, W)
        loader = DataLoader(
            TensorDataset(
                torch.tensor(X, dtype=torch.float32),
                torch.tensor(y_np, dtype=torch.long),
            ),
            batch_size=batch_size,
            shuffle=False,
        )

        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb).argmax(dim=1)
                correct += (pred == yb).sum().item()
                total += yb.numel()

        return correct / max(total, 1)

    # ---------------- MoRF / LeRF ----------------
    hist = np.zeros(n_steps + 1, dtype=np.float32)

    hist[0] = eval_step(X_flat.copy())
    print(f"[{mode.upper()}] Step 0 acc = {hist[0]:.4f}")

    X_work = X_flat.copy()

    for k in range(1, n_steps + 1):
        start = (k - 1) * pixels_per_step
        end   = min(k * pixels_per_step, HW)

        for i in range(N):
            idxs = order[i, start:end]

            if mask_type == "zero":
                X_work[i, :, idxs] = 0.0

            elif mask_type in ["pgd", "road"]:
                if mask_flat is None:
                    raise RuntimeError(f"mask_np required for {mask_type}")
                X_work[i, :, idxs] = mask_flat[i, :, idxs]

            else:
                raise ValueError(mask_type)

        hist[k] = eval_step(X_work)
        print(f"[{mode.upper()}] Step {k}/{n_steps} acc = {hist[k]:.4f}")

    return hist



# ============================================================
# Main
# ============================================================
if __name__ == "__main__":

    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, required=True,
                        choices=["resnet_50", "efficientnet_b0", "repvgg_b0"])
    parser.add_argument("--expl_method", type=str, default="smoothgrad")

    parser.add_argument("--imagenet_root", type=str,
                        default=os.path.join(PROJECT_ROOT, "data", "ImageNet"))

    parser.add_argument("--subset_idx", type=str,
                        default=os.path.join(
                            PROJECT_ROOT,
                            "data", "ImageNet", "imagenet_val_500.npy"
                        ))

    parser.add_argument("--n_steps", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--mask_type", type=str,
                        default="pgd", choices=["zero", "pgd", "road"])
    parser.add_argument("--mode", type=str,
                        default="morf", choices=["morf", "lerf"])

    # PGD params
    parser.add_argument("--epsilon", type=float, default=8/255)
    parser.add_argument("--pgd_steps", type=int, default=10)

    args = parser.parse_args()

    # ============================================================
    # Device
    # ============================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ============================================================
    # Dataset (ImageNet val + fixed 500)
    # ============================================================
    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225],
        )
    ])

    val_set = datasets.ImageFolder(
        os.path.join(args.imagenet_root, "val"),
        transform=tf
    )

    fixed_idx = np.load(args.subset_idx).astype(int)
    val_subset = Subset(val_set, fixed_idx)

    loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    )

    Xs, ys = [], []
    for x, y in loader:
        Xs.append(x.numpy())
        ys.append(y.numpy())

    X_test = np.concatenate(Xs, axis=0)
    y_test = np.concatenate(ys, axis=0)

    print(f"[INFO] Using fixed ImageNet subset: {len(X_test)} samples")

    # ============================================================
    # Model (pretrained)
    # ============================================================
    if args.model == "resnet_50":
        model = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        )
    elif args.model == "efficientnet_b0":
        model = torchvision.models.efficientnet_b0(
            weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )
    elif args.model == "repvgg_b0":
        model = timm.create_model("repvgg_b0", pretrained=True, num_classes=1000)
    else:
        raise ValueError(args.model)

    model = model.to(device).eval()

    # ============================================================
    # PGD and ROAD cache (run once per model)
    # ============================================================
    if args.mask_type == "pgd":

        pgd_cache_dir = os.path.join(
            PROJECT_ROOT, "experiment", "adv_cache", "imagenet", args.model
        )
        os.makedirs(pgd_cache_dir, exist_ok=True)

        pgd_cache_path = os.path.join(
            pgd_cache_dir,
            f"imagenet_val_500_eps{args.epsilon}_steps{args.pgd_steps}.npy"
        )

        if os.path.exists(pgd_cache_path):
            print("[PGD] Loading cached adversarial examples")
            mask_np = np.load(pgd_cache_path)
        else:
            print("[PGD] Running PGD once...")
            loader_pgd = build_loader_from_np(X_test, y_test, args.batch_size)
            adv = pgd(model, device, loader_pgd, nn.CrossEntropyLoss(),
                    epsilon=args.epsilon, steps=args.pgd_steps)
            mask_np = adv.cpu().numpy()
            np.save(pgd_cache_path, mask_np)

    elif args.mask_type == "road":

        road_cache_dir = os.path.join(
            PROJECT_ROOT, "experiment", "road_cache", "imagenet"
        )
        os.makedirs(road_cache_dir, exist_ok=True)

        road_cache_path = os.path.join(
            road_cache_dir, "road_noise0.2.npy"
        )

        if os.path.exists(road_cache_path):
            print("[ROAD] Loading cached ROAD images")
            mask_np = np.load(road_cache_path)
        else:
            print("[ROAD] Generating ROAD images once...")
            mask_np = road(loader, noise_std=0.2)
            np.save(road_cache_path, mask_np)

    else:
        # zero masking
        mask_np = None



    # ============================================================
    # Load saliency
    # ============================================================
    expl_path = os.path.join(
        PROJECT_ROOT,
        "expl_image_imagenet",
        args.model,
        f"{args.expl_method}.npy"
    )
    saliency = np.load(expl_path)
    saliency = saliency[:len(X_test)]

    print("[INFO] Saliency shape:", saliency.shape)

    # ============================================================
    # Run MoRF / LeRF
    # ============================================================
    hist = run_morf_lerf_image(
        model=model,
        device=device,
        X_np=X_test,
        y_np=y_test,
        saliency_np=saliency,
        mask_np=mask_np,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        mask_type=args.mask_type,
        mode=args.mode,
    )

    # ============================================================
    # Save
    # ============================================================
    out_dir = os.path.join(
        PROJECT_ROOT,
        "morf_lerf_image",
        args.model,
        "imagenet",
        args.mask_type
    )
    os.makedirs(out_dir, exist_ok=True)

    save_path = os.path.join(
        out_dir,
        f"{args.expl_method}_{args.mode}.pkl"
    )

    with open(save_path, "wb") as f:
        pickle.dump(hist, f)

    print("[DONE] Saved results to:", save_path)
