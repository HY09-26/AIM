# ============================================================
# test_oxfordpet.py
#
# Oxford-IIIT Pet MoRF / LeRF evaluation
# - Fixed test subset (fixed_test_indices_500.npy)
# - Trained models (ResNet50 / EfficientNet-B0 / RepVGG-B0)
# - Masking: zero / PGD / ROAD
#   - PGD cached once per model + eps + steps + subset tag
#   - ROAD cached once per dataset/subset + noise_std (model-agnostic)
# ============================================================

import os
import sys
import argparse
import pickle

import numpy as np
import torch
import torch.nn as nn
import torchvision
import timm

from torch.utils.data import DataLoader, TensorDataset, Subset
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

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
# MoRF / LeRF (Image)  -- unified mask_np
# ============================================================
def run_morf_lerf_image(
    model,
    device,
    X_np,           # (N,3,H,W)
    y_np,           # (N,)
    saliency_np,    # (N,3,H,W) or (N,1,H,W)
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

    # saliency → ranking
    sal = np.abs(saliency_np).sum(axis=1).reshape(N, HW)

    if mode == "morf":
        order = np.argsort(-sal, axis=1)
    elif mode == "lerf":
        order = np.argsort(+sal, axis=1)
    else:
        raise ValueError(mode)

    pixels_per_step = max(1, HW // n_steps)

    @torch.no_grad()
    def eval_step(X_flat_step):
        X_step = X_flat_step.reshape(N, C, H, W)
        loader = DataLoader(
            TensorDataset(
                torch.tensor(X_step, dtype=torch.float32),
                torch.tensor(y_np, dtype=torch.long)
            ),
            batch_size=batch_size,
            shuffle=False
        )

        correct, total = 0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.numel()

        return correct / max(total, 1)

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
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt", type=str,
                        default=os.path.join(PROJECT_ROOT, "experiment_utils", "checkpoints", "repvgg_pets_best.pth"))
    parser.add_argument("--expl_dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "expl_image_oxfordpet", "repvgg_b0"))
    parser.add_argument("--expl_method", type=str, default="gradients",
                        choices=["gradcam", "gradcampp", "gradients", "smoothgrad", "random"])

    parser.add_argument("--fixed_idx_path", type=str,
                        default=os.path.join(PROJECT_ROOT, "experiment_utils", "checkpoints", "fixed_test_indices_500.npy"))

    parser.add_argument("--model", type=str, default="repvgg_b0",
                        choices=["resnet_50", "efficientnet_b0", "repvgg_b0"])

    parser.add_argument("--n_steps", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--mask_type", type=str, default="pgd",
                        choices=["zero", "pgd", "road"])
    parser.add_argument("--mode", type=str, default="lerf",
                        choices=["morf", "lerf"])

    # PGD params
    parser.add_argument("--epsilon", type=float, default=2/255)
    parser.add_argument("--pgd_steps", type=int, default=10)

    # ROAD params
    parser.add_argument("--road_noise", type=float, default=0.2)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ============================================================
    # Dataset (ONLY fixed_test_indices_500)
    # ============================================================
    from experiment_utils.pets_dataset import OxfordPetsDataset
    from torchvision import transforms

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    full_test = OxfordPetsDataset(
        root=os.path.join(PROJECT_ROOT, "data", "Oxford_Pet"),
        split="test",
        transform=tf
    )

    fixed_idx = np.load(args.fixed_idx_path).astype(int)
    test_set = Subset(full_test, fixed_idx)

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )

    Xs, ys = [], []
    for x, y in tqdm(test_loader, desc="Loading fixed test samples"):
        Xs.append(x.numpy())
        ys.append(y.numpy())

    X_test = np.concatenate(Xs, axis=0)
    y_test = np.concatenate(ys, axis=0)

    print("#Fixed test samples:", len(X_test))

    # ============================================================
    # Model
    # ============================================================
    num_classes = 37

    if args.model == "resnet_50":
        model = torchvision.models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif args.model == "efficientnet_b0":
        model = torchvision.models.efficientnet_b0(weights=None)
        in_dim = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_dim, num_classes)

    elif args.model == "repvgg_b0":
        model = timm.create_model("repvgg_b0", pretrained=False, num_classes=num_classes)

    else:
        raise ValueError(args.model)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model = model.to(device).eval()

    # ============================================================
    # Load saliency
    # ============================================================
    saliency = np.load(os.path.join(args.expl_dir, f"{args.expl_method}.npy"))
    saliency = saliency[:len(X_test)]
    print("[INFO] Saliency shape:", saliency.shape)

    # ============================================================
    # Build mask_np (zero / pgd / road)
    # ============================================================
    mask_np = None

    subset_tag = os.path.splitext(os.path.basename(args.fixed_idx_path))[0]

    if args.mask_type == "pgd":

        adv_cache_dir = os.path.join(
            PROJECT_ROOT, "experiment", "adv_cache", "oxford_pet", args.model
        )
        os.makedirs(adv_cache_dir, exist_ok=True)

        eps_tag = f"{args.epsilon:.6f}".replace(".", "p")
        adv_cache_path = os.path.join(
            adv_cache_dir,
            f"{subset_tag}_pgd_eps{eps_tag}_steps{args.pgd_steps}.npy"
        )

        if os.path.exists(adv_cache_path):
            print("[PGD] Loading cached adversarial examples:")
            print(" ", adv_cache_path)
            mask_np = np.load(adv_cache_path)
        else:
            print("[PGD] Cache not found, running PGD once...")
            loader_pgd = build_loader_from_np(X_test, y_test, batch_size=args.batch_size)

            adv = pgd(
                model,
                device,
                loader_pgd,
                nn.CrossEntropyLoss(),
                epsilon=args.epsilon,
                steps=args.pgd_steps,
            )

            mask_np = adv.detach().cpu().numpy().astype(np.float32)
            np.save(adv_cache_path, mask_np)
            print("[PGD] Saved PGD cache to:")
            print(" ", adv_cache_path)

            del adv
            torch.cuda.empty_cache()

        assert mask_np.shape == X_test.shape, f"PGD mask shape mismatch: {mask_np.shape} vs {X_test.shape}"

    elif args.mask_type == "road":

        road_cache_dir = os.path.join(
            PROJECT_ROOT, "experiment", "road_cache", "oxford_pet"
        )
        os.makedirs(road_cache_dir, exist_ok=True)

        noise_tag = f"{args.road_noise:.3f}".replace(".", "p")
        road_cache_path = os.path.join(
            road_cache_dir,
            f"{subset_tag}_road_noise{noise_tag}.npy"
        )

        if os.path.exists(road_cache_path):
            print("[ROAD] Loading cached ROAD replacement images:")
            print(" ", road_cache_path)
            mask_np = np.load(road_cache_path)
        else:
            print("[ROAD] Generating ROAD replacement images (first time only)...")
            # IMPORTANT: use the SAME order as X_test (fixed subset loader)
            mask_np = road(test_loader, noise_std=args.road_noise)
            np.save(road_cache_path, mask_np)
            print("[ROAD] Saved ROAD cache to:")
            print(" ", road_cache_path)

        assert mask_np.shape == X_test.shape, f"ROAD mask shape mismatch: {mask_np.shape} vs {X_test.shape}"

    else:
        # zero masking: mask_np is None
        mask_np = None

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
        "oxford_pet",
        args.mask_type,
    )
    os.makedirs(out_dir, exist_ok=True)

    save_path = os.path.join(out_dir, f"{args.expl_method}_{args.mode}.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(hist, f)

    print("[DONE] Saved results to:", save_path)
