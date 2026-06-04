import os
import argparse
import pickle
import numpy as np

import timm
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from tqdm import tqdm
from experiment_utils.utils import pgd_attack_brainmri, road
pgd = pgd_attack_brainmri
from experiment_utils.model import get_efficientnet_b0, get_resnet50, get_repvgg_b0
# ============================================================
# Utilities
# ============================================================
def build_loader_from_np(X_np, y_np, batch_size):
    x = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.long)
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def saliency_to_rank(sal):
    """
    sal: (N,3,H,W) or (N,1,H,W)
    return: (N, H*W) ranking
    """
    if sal.ndim == 4:
        sal = np.abs(sal).sum(axis=1)
    N, H, W = sal.shape
    sal = sal.reshape(N, -1)
    return np.argsort(-sal, axis=1)


# ============================================================
# MoRF / LeRF (Image)
# ============================================================
def run_morf_lerf_image(
    model,
    device,
    X_np,           # (N,3,H,W)
    y_np,           # (N,)
    saliency_np,    # (N,3,H,W) or (N,1,H,W)
    mask_np=None,    # (N,3,H,W) or None
    n_steps=20,
    batch_size=64,
    mask_type="zero",    # "pgd" or "zero"
    mode="morf",         # "morf" or "lerf"
):
    model.eval()

    N, C, H, W = X_np.shape
    HW = H * W

    X_flat = X_np.reshape(N, C, HW).copy()
    mask_flat = mask_np.reshape(N, C, HW) if mask_np is not None else None


    sal = np.abs(saliency_np).sum(axis=1).reshape(N, HW)

    if mode == "morf":
        order = np.argsort(-sal, axis=1)
    elif mode == "lerf":
        order = np.argsort(+sal, axis=1)
    else:
        raise ValueError(mode)

    pixels_per_step = max(1, HW // n_steps)

    def eval_step(X_flat_step):
        X_step = X_flat_step.reshape(N, C, H, W)
        loader = DataLoader(
            TensorDataset(
                torch.tensor(X_step, dtype=torch.float32),
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

    hist = np.zeros(n_steps + 1, dtype=np.float32)

    hist[0] = eval_step(X_flat.copy())
    print(f"[{mode.upper()}] Step 0 acc = {hist[0]:.4f}")

    X_work = X_flat.copy()
    zero_value = 0.0

    for k in range(1, n_steps + 1):
        start = (k - 1) * pixels_per_step
        end   = min(k * pixels_per_step, HW)

        for i in range(N):
            idxs = order[i, start:end]

            if mask_type == "pgd":
                if mask_flat is None:
                    raise RuntimeError("mask_np required for PGD masking")
                X_work[i, :, idxs] = mask_flat[i, :, idxs]

            elif mask_type == "zero":
                X_work[i, :, idxs] = zero_value

            elif mask_type == "road":
                if mask_flat is None:
                    raise RuntimeError("mask_np required for ROAD masking")
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
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    parser.add_argument("--model", choices=["resnet_50", "efficientnet_b0","repvgg_b0"], required=True)
    parser.add_argument("--ckpt", type=str, default=os.path.join(PROJECT_ROOT,"experiment_utils","checkpoints","resnet50_brain_mri_best.pth"))
    parser.add_argument("--expl_dir", type=str, default=os.path.join(PROJECT_ROOT,"expl_image_brainmri","resnet_50"))

    parser.add_argument("--expl_method", type=str, default="smoothgrad")

    parser.add_argument("--mask_type", choices=["zero", "pgd", "road"], default="zero")
    parser.add_argument("--mode", choices=["morf", "lerf"], default="morf")

    parser.add_argument("--n_steps", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--epsilon", type=float, default=2/255)
    parser.add_argument("--pgd_steps", type=int, default=10)
    args = parser.parse_args()

    DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "Brain_MRI_Tumor")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # --------------------------------------------------
    # Load Brain MRI test set (ImageFolder order)
    # --------------------------------------------------
    test_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])

    test_set = datasets.ImageFolder(
        root=os.path.join(DATA_ROOT, "Testing"),
        transform=test_tf
    )

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )

    Xs, ys = [], []
    for x, y in tqdm(test_loader, desc="Loading test data"):
        Xs.append(x.numpy())
        ys.append(y.numpy())

    X_test = np.concatenate(Xs, axis=0)
    y_test = np.concatenate(ys, axis=0)

    print(f"#Test samples = {len(X_test)}")

    # --------------------------------------------------
    # Load model
    # --------------------------------------------------
    num_classes = len(test_set.classes)

    if args.model == "resnet_50":
        model = torchvision.models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif args.model == "efficientnet_b0":
        model = torchvision.models.efficientnet_b0(weights=None)
        in_dim = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_dim, num_classes)
    elif args.model == "repvgg_b0":
        model = get_repvgg_b0(num_classes=num_classes, pretrained=False)

        in_dim = model.head.fc.in_features
        model.head.fc = nn.Linear(in_dim, num_classes)

    print("Loading checkpoint:", args.ckpt)
    state_dict = torch.load(args.ckpt, map_location="cpu")

    model.load_state_dict(torch.load(args.ckpt, map_location=device))


    model = model.to(device).eval()

    # --------------------------------------------------
    # Load saliency
    # --------------------------------------------------
    saliency = np.load(os.path.join(args.expl_dir, f"{args.expl_method}.npy"))
    saliency = saliency[:len(X_test)]
    print("Saliency shape:", saliency.shape)

    mask_np = None
    # --------------------------------------------------
    # PGD adversarial masking (Brain MRI)
    # --------------------------------------------------
    adv_np = None

    if args.mask_type == "pgd":

        adv_cache_dir = os.path.join(
            PROJECT_ROOT,
            "experiment",
            "adv_cache",
            "brain_mri",
            args.model
        )
        os.makedirs(adv_cache_dir, exist_ok=True)

        adv_cache_path = os.path.join(
            adv_cache_dir,
            f"pgd_eps{args.epsilon}_steps{args.pgd_steps}.npy"
        )

        if os.path.exists(adv_cache_path):
            print("[PGD] Loading cached adversarial examples:")
            print(" ", adv_cache_path)
            adv_np = np.load(adv_cache_path)

        else:
            print("[PGD] Running PGD (first time only)...")

            loader_pgd = build_loader_from_np(
                X_test, y_test, batch_size=args.batch_size
            )

            adv = pgd(
                model,
                device,
                loader_pgd,
                nn.CrossEntropyLoss(),
                epsilon=args.epsilon,
                steps=args.pgd_steps,
            )

            adv_np = adv.detach().cpu().numpy()
            np.save(adv_cache_path, adv_np)

            print("[PGD] Saved adversarial examples to:")
            print(" ", adv_cache_path)

            del adv
            torch.cuda.empty_cache()
        
        mask_np = adv_np

    else:
        mask_np = None
    

    # --------------------------------------------------
    # ROAD
    # --------------------------------------------------

    if args.mask_type == "road":
        road_cache_dir = os.path.join(PROJECT_ROOT, "experiment", "road_cache", "brain_mri")
        os.makedirs(road_cache_dir, exist_ok=True)

        road_cache_path = os.path.join(
            road_cache_dir,
            f"road_noise0.2.npy"
        )

        if os.path.exists(road_cache_path):
            print("[ROAD] Loading cached ROAD replacement images:")
            print(" ", road_cache_path)
            mask_np = np.load(road_cache_path)
        else:
            print("[ROAD] Generating ROAD replacement images (first time only)...")
            loader_road = build_loader_from_np(X_test, y_test, batch_size=args.batch_size)

            mask_np = road(
                loader_road, noise_std=0.1
            )

            np.save(road_cache_path, mask_np)
            print("[ROAD] Saved ROAD replacement images to:")
            print(" ", road_cache_path)
        print("ROAD mask shape:", mask_np.shape)

    # --------------------------------------------------
    # Run MoRF / LeRF
    # --------------------------------------------------
    hist = run_morf_lerf_image(
        model, device,
        X_np=X_test,
        y_np=y_test,
        saliency_np=saliency,
        mask_np=mask_np,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        mask_type=args.mask_type,
        mode=args.mode,
    )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------
    out_dir = os.path.join(
        PROJECT_ROOT,
        "morf_lerf_image",
        args.model,
        "brain_mri",
        args.mask_type,
    )
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, f"{args.expl_method}_{args.mode}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(hist, f)

    print("Saved results to:", out_path)
