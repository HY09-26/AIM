"""
test_image.py - MoRF/LeRF faithfulness evaluation unified for image datasets.

Dataset-specific details
------------------------
BrainMRI (4 classes):
    Fine-tuned models loaded from checkpoint.
    Transform: Grayscale → 3-channel, Resize(224), ImageNet normalisation.
    PGD runs in raw [0,1] pixel space and returns normalised adversarials,
    avoiding accumulated normalisation error.

ImageNet (1000 classes):
    Pretrained weights from torchvision / timm.  No checkpoint needed.
    Transform: Resize(256) → CenterCrop(224), ImageNet normalisation.
    Fixed 500-sample validation subset (imagenet_val_500.npy).
    PGD runs directly in normalised feature space.

OxfordPet (37 classes):
    Fine-tuned models loaded from checkpoint.
    Transform: Resize(224), ImageNet normalisation.
    Fixed 500-sample test subset (fixed_test_indices_500.npy).
    PGD runs directly in normalised feature space.

Usage
-----
python test_image.py --dataset brain_mri --model resnet_50 \\
    --expl_method smoothgrad --mask_type pgd
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import timm
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms
from tqdm import tqdm

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)

from experiment_utils.utils import pgd_attack, pgd_attack_brainmri, road

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Constants
# ============================================================

DATASET_NUM_CLASSES: dict[str, int] = {
    "brain_mri":  4,
    "imagenet":   1000,
    "oxford_pet": 37,
}

# Default PGD ε.  ImageNet uses a larger budget because pretrained models are
# more robust; BrainMRI uses a smaller one because images are grayscale MRI.
PGD_EPS_DEFAULT: dict[str, float] = {
    "brain_mri":  2 / 255,
    "imagenet":   8 / 255,
    "oxford_pet": 2 / 255,
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ============================================================
# Dataset-specific data loaders
# ============================================================

def _load_brain_mri(
    data_root: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the Brain MRI Tumor test set from an ImageFolder directory.

    Applies Grayscale→3-channel conversion, resizes to 224×224, and applies
    ImageNet mean/std normalisation.

    Args:
        data_root:  Root directory containing a "Testing/" subdirectory.
        batch_size: DataLoader batch size.

    Returns:
        (X_test, y_test): float32 (N,3,224,224) and int64 (N,) arrays.
    """
    tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    test_set = datasets.ImageFolder(
        root=os.path.join(data_root, "Testing"), transform=tf
    )
    loader = DataLoader(test_set, batch_size=batch_size,
                        shuffle=False, num_workers=4)
    Xs, ys = [], []
    for x, y in tqdm(loader, desc="Loading BrainMRI test set"):
        Xs.append(x.numpy()); ys.append(y.numpy())
    return np.concatenate(Xs), np.concatenate(ys)


def _load_imagenet(
    data_root: str,
    subset_idx_path: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a fixed subset of the ImageNet validation set.

    Args:
        data_root:       Root directory containing a "val/" subdirectory.
        subset_idx_path: Path to .npy file with fixed sample indices.
        batch_size:      DataLoader batch size.

    Returns:
        (X_test, y_test): float32 (N,3,224,224) and int64 (N,) arrays.
    """
    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_set    = datasets.ImageFolder(os.path.join(data_root, "val"), transform=tf)
    fixed_idx  = np.load(subset_idx_path).astype(int)
    val_subset = Subset(val_set, fixed_idx)
    loader     = DataLoader(val_subset, batch_size=batch_size,
                            shuffle=False, num_workers=4)
    Xs, ys = [], []
    for x, y in tqdm(loader, desc="Loading ImageNet subset"):
        Xs.append(x.numpy()); ys.append(y.numpy())
    return np.concatenate(Xs), np.concatenate(ys)


def _load_oxford_pet(
    data_root: str,
    fixed_idx_path: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a fixed subset of the Oxford-IIIT Pet test set.

    Uses the custom ``OxfordPetsDataset`` class to handle the non-standard
    directory layout.

    Args:
        data_root:      Root directory of the Oxford-IIIT Pet dataset.
        fixed_idx_path: Path to .npy file with fixed 500-sample indices.
        batch_size:     DataLoader batch size.

    Returns:
        (X_test, y_test): float32 (N,3,224,224) and int64 (N,) arrays.
    """
    from experiment_utils.pets_dataset import OxfordPetsDataset

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    full_test  = OxfordPetsDataset(root=data_root, split="test", transform=tf)
    fixed_idx  = np.load(fixed_idx_path).astype(int)
    test_subset = Subset(full_test, fixed_idx)
    loader      = DataLoader(test_subset, batch_size=batch_size,
                             shuffle=False, num_workers=4)
    Xs, ys = [], []
    for x, y in tqdm(loader, desc="Loading OxfordPet test subset"):
        Xs.append(x.numpy()); ys.append(y.numpy())
    return np.concatenate(Xs), np.concatenate(ys)


def load_test_data(
    dataset: str,
    data_root: str,
    batch_size: int,
    subset_idx_path: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Dispatcher: load (X_test, y_test) for the given dataset.

    Args:
        dataset:         One of "brain_mri", "imagenet", "oxford_pet".
        data_root:       Root directory of the dataset.
        batch_size:      DataLoader batch size.
        subset_idx_path: Path to fixed-subset .npy (required for imagenet and oxford_pet).

    Returns:
        X_test: float32 array, shape (N, 3, 224, 224).
        y_test: int64 array, shape (N,).
    """
    if dataset == "brain_mri":
        return _load_brain_mri(data_root, batch_size)
    if dataset == "imagenet":
        if subset_idx_path is None:
            subset_idx_path = str(
                Path(data_root) / "imagenet_val_500.npy"
            )
        return _load_imagenet(data_root, subset_idx_path, batch_size)
    if dataset == "oxford_pet":
        if subset_idx_path is None:
            subset_idx_path = str(
                Path(PROJECT_ROOT) / "experiment_utils" / "checkpoints"
                / "fixed_test_indices_500.npy"
            )
        return _load_oxford_pet(data_root, subset_idx_path, batch_size)
    raise ValueError(f"Unknown dataset: {dataset!r}")


# ============================================================
# Model builder
# ============================================================

def build_model(
    dataset: str,
    model_name: str,
    num_classes: int,
    ckpt_path: Optional[str],
) -> nn.Module:
    """
    Build and load a classification model.

    ImageNet models use pretrained weights from torchvision / timm; no
    checkpoint is needed.  BrainMRI and OxfordPet models are fine-tuned and
    must be loaded from a checkpoint.

    Args:
        dataset:     Dataset identifier (determines pretrained vs checkpoint).
        model_name:  One of "resnet_50", "efficientnet_b0", "repvgg_b0".
        num_classes: Number of output classes.
        ckpt_path:   Path to the checkpoint .pth file (ignored for ImageNet).

    Returns:
        Initialised model in eval mode, moved to ``device``.
    """
    pretrained = dataset == "imagenet"

    if model_name == "resnet_50":
        if pretrained:
            model = torchvision.models.resnet50(
                weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2
            )
        else:
            model = torchvision.models.resnet50(weights=None)
            model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "efficientnet_b0":
        if pretrained:
            model = torchvision.models.efficientnet_b0(
                weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
            )
        else:
            model = torchvision.models.efficientnet_b0(weights=None)
            model.classifier[1] = nn.Linear(
                model.classifier[1].in_features, num_classes
            )

    elif model_name == "repvgg_b0":
        if pretrained:
            model = timm.create_model("repvgg_b0", pretrained=True,
                                      num_classes=num_classes)
        else:
            model = timm.create_model("repvgg_b0", pretrained=False,
                                      num_classes=num_classes)

    else:
        raise ValueError(f"Unknown model: {model_name!r}")

    if not pretrained:
        if ckpt_path is None:
            raise ValueError(f"--ckpt is required for dataset={dataset!r}")
        logger.info("Loading checkpoint: %s", ckpt_path)
        ckpt  = torch.load(ckpt_path, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        model.load_state_dict(state)
    else:
        logger.info("Using pretrained weights for ImageNet.")

    return model.to(device).eval()


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def eval_epoch(model: nn.Module, loader: DataLoader) -> float:
    """Compute classification accuracy over a DataLoader."""
    model.eval()
    correct = total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        correct += (model(xb).argmax(1) == yb).sum().item()
        total   += yb.size(0)
    return correct / max(1, total)


# ============================================================
# MoRF / LeRF runner
# ============================================================

def run_morf_lerf(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    saliency: np.ndarray,
    n_steps: int,
    batch_size: int,
    mode: str,
    mask_np: Optional[np.ndarray] = None,
    mask_type: str = "zero",
) -> np.ndarray:
    """
    Rank-based MoRF or LeRF faithfulness evaluation for image inputs.

    Saliency scores are summed over colour channels (L1) to obtain a single
    spatial importance map, which is then used to rank pixels.

    Args:
        model:      Classifier accepting (B, 3, H, W) inputs.
        X:          Input images, shape (N, 3, H, W), float32.
        y:          Labels, shape (N,), int64.
        saliency:   Attribution maps, shape (N, C, H, W) or (N, H, W).
        n_steps:    Number of masking steps.
        batch_size: Inference batch size.
        mode:       "morf" (mask highest-saliency pixels first) or "lerf".
        mask_np:    Replacement pixel values for "pgd" / "road" masking,
                    shape (N, 3, H, W).  Must be provided if mask_type != "zero".
        mask_type:  One of "zero", "pgd", "road".

    Returns:
        acc_curve: Accuracy at each step, shape (n_steps + 1,).
    """
    model.eval()
    N, C, H, W = X.shape
    HW = H * W

    # Aggregate saliency to a single spatial map via channel-sum of |s|
    if saliency.ndim == 4:
        sal_map = np.abs(saliency).sum(axis=1)       # (N, H, W)
    else:
        sal_map = np.abs(saliency)                   # already (N, H, W)
    sal_flat = sal_map.reshape(N, HW)

    order = (np.argsort(-sal_flat, axis=1) if mode == "morf"
             else np.argsort(sal_flat, axis=1))

    step_sz    = max(1, HW // n_steps)
    X_flat     = X.reshape(N, C, HW).copy()
    mask_flat  = mask_np.reshape(N, C, HW) if mask_np is not None else None

    acc_curve = []
    for k in range(n_steps + 1):
        if k > 0:
            start = (k - 1) * step_sz
            end   = min(k * step_sz, HW)
            for i in range(N):
                idxs = order[i, start:end]
                if mask_type == "zero":
                    X_flat[i, :, idxs] = 0.0
                else:
                    if mask_flat is None:
                        raise RuntimeError(
                            f"mask_np required for mask_type={mask_type!r}"
                        )
                    X_flat[i, :, idxs] = mask_flat[i, :, idxs]

        loader = DataLoader(
            TensorDataset(
                torch.tensor(X_flat.reshape(N, C, H, W), dtype=torch.float32),
                torch.tensor(y, dtype=torch.long),
            ),
            batch_size=batch_size,
            shuffle=False,
        )
        acc = eval_epoch(model, loader)
        acc_curve.append(acc)
        logger.info(
            "[%s] step %02d/%d | masked=%5.1f%% | acc=%.4f",
            mode.upper(), k, n_steps, 100 * k / n_steps, acc,
        )

    return np.array(acc_curve, dtype=np.float32)


# ============================================================
# Default path helpers
# ============================================================

def _default_data_root(dataset: str) -> str:
    base = Path(PROJECT_ROOT) / "data"
    return str({
        "brain_mri":  base / "Brain_MRI_Tumor",
        "imagenet":   base / "ImageNet",
        "oxford_pet": base / "Oxford_Pet",
    }[dataset])


def _default_ckpt(dataset: str, model_name: str) -> Optional[str]:
    """Return None for ImageNet (uses pretrained weights)."""
    if dataset == "imagenet":
        return None
    ckpt_names = {
        ("brain_mri",  "resnet_50"):       "resnet50_brain_mri_best.pth",
        ("brain_mri",  "efficientnet_b0"): "efficientnet_b0_brain_mri_best.pth",
        ("brain_mri",  "repvgg_b0"):       "repvgg_b0_brain_mri_best.pth",
        ("oxford_pet", "resnet_50"):       "resnet50_pets_best.pth",
        ("oxford_pet", "efficientnet_b0"): "efficientnet_b0_pets_best.pth",
        ("oxford_pet", "repvgg_b0"):       "repvgg_pets_best.pth",
    }
    name = ckpt_names.get((dataset, model_name), f"{model_name}_{dataset}_best.pth")
    return str(Path(PROJECT_ROOT) / "experiment_utils" / "checkpoints" / name)


def _saliency_path(dataset: str, model_name: str, method: str) -> Path:
    expl_dirs = {
        "brain_mri":  Path(PROJECT_ROOT) / "expl_image_brainmri",
        "imagenet":   Path(PROJECT_ROOT) / "expl_image_imagenet",
        "oxford_pet": Path(PROJECT_ROOT) / "expl_image_oxfordpet",
    }
    return expl_dirs[dataset] / model_name / f"{method}.npy"


def _pgd_cache_path(
    dataset: str, model_name: str, eps: float, steps: int
) -> Path:
    dataset_tag = dataset.replace("_", "")   # "brain_mri" → "brainmri" for readability
    cache_dir = (
        Path(PROJECT_ROOT) / "experiment" / "adv_cache"
        / dataset / model_name
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"pgd_eps{eps:.6f}_steps{steps}.npy"


def _road_cache_path(dataset: str, noise_std: float) -> Path:
    cache_dir = Path(PROJECT_ROOT) / "experiment" / "road_cache" / dataset
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"road_noise{noise_std:.3f}.npy"


def _output_dir(dataset: str, model_name: str, mask_type: str) -> Path:
    out = Path(PROJECT_ROOT) / "morf_lerf_image" / model_name / dataset / mask_type
    out.mkdir(parents=True, exist_ok=True)
    return out


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MoRF/LeRF faithfulness evaluation for image datasets."
    )
    parser.add_argument("--dataset",
                        choices=["brain_mri", "imagenet", "oxford_pet"],
                        required=True)
    parser.add_argument("--model",
                        choices=["resnet_50", "efficientnet_b0", "repvgg_b0"],
                        required=True)
    parser.add_argument("--expl_method",  default="smoothgrad")
    parser.add_argument("--mask_type",    choices=["zero", "pgd", "road"],
                        default="zero")
    parser.add_argument("--data_root",    default=None,
                        help="Dataset root directory. Auto-detected if omitted.")
    parser.add_argument("--subset_idx",   default=None,
                        help="Path to fixed-subset .npy file "
                             "(imagenet / oxford_pet).")
    parser.add_argument("--ckpt",         default=None,
                        help="Checkpoint path (required for brain_mri / oxford_pet).")
    parser.add_argument("--n_steps",      type=int, default=20)
    parser.add_argument("--batch_size",   type=int, default=32)
    parser.add_argument("--epsilon",      type=float, default=None,
                        help="PGD ε. Auto-selected per dataset if omitted.")
    parser.add_argument("--pgd_steps",    type=int, default=10)
    parser.add_argument("--road_noise",   type=float, default=0.2)
    args = parser.parse_args()

    # ── Fill defaults ────────────────────────────────────────────────────────
    if args.data_root is None:
        args.data_root = _default_data_root(args.dataset)
    if args.ckpt is None:
        args.ckpt = _default_ckpt(args.dataset, args.model)
    if args.epsilon is None:
        args.epsilon = PGD_EPS_DEFAULT[args.dataset]

    num_classes = DATASET_NUM_CLASSES[args.dataset]
    logger.info("Device: %s | Dataset: %s | Model: %s | Mask: %s",
                device, args.dataset, args.model, args.mask_type)

    # ── Build model ──────────────────────────────────────────────────────────
    model = build_model(args.dataset, args.model, num_classes, args.ckpt)

    # ── Load test data ───────────────────────────────────────────────────────
    logger.info("Loading test data from: %s", args.data_root)
    X_test, y_test = load_test_data(
        dataset=args.dataset,
        data_root=args.data_root,
        batch_size=args.batch_size,
        subset_idx_path=args.subset_idx,
    )
    logger.info("Test set shape: X=%s  y=%s", X_test.shape, y_test.shape)

    # ── Sanity check ─────────────────────────────────────────────────────────
    sanity_loader = DataLoader(
        TensorDataset(torch.tensor(X_test), torch.tensor(y_test, dtype=torch.long)),
        batch_size=args.batch_size, shuffle=False,
    )
    logger.info("Baseline accuracy (unmasked): %.4f",
                eval_epoch(model, sanity_loader))

    # ── Load saliency ────────────────────────────────────────────────────────
    sal_path = _saliency_path(args.dataset, args.model, args.expl_method)
    logger.info("Loading saliency: %s", sal_path)
    saliency = np.load(sal_path)[:len(X_test)]
    logger.info("Saliency shape: %s", saliency.shape)

    # ── Build mask (PGD / ROAD / zero) ───────────────────────────────────────
    mask_np = None

    if args.mask_type == "pgd":
        pgd_path = _pgd_cache_path(args.dataset, args.model,
                                   args.epsilon, args.pgd_steps)
        if pgd_path.exists():
            logger.info("Loading cached PGD from %s", pgd_path)
            mask_np = np.load(pgd_path)
        else:
            logger.info("Running PGD (eps=%.6f, steps=%d)...",
                        args.epsilon, args.pgd_steps)
            pgd_loader = DataLoader(
                TensorDataset(torch.tensor(X_test), torch.tensor(y_test, dtype=torch.long)),
                batch_size=args.batch_size, shuffle=False,
            )
            loss_fn = nn.CrossEntropyLoss()

            if args.dataset == "brain_mri":
                # PGD in raw [0,1] pixel space to avoid normalisation issues
                adv = pgd_attack_brainmri(
                    model, device, pgd_loader, loss_fn,
                    epsilon=args.epsilon,
                    alpha=args.epsilon / args.pgd_steps,
                    steps=args.pgd_steps,
                )
            else:
                # PGD directly in normalised feature space
                adv = pgd_attack(
                    model, device, pgd_loader, loss_fn,
                    epsilon=args.epsilon,
                    steps=args.pgd_steps,
                )

            mask_np = adv.detach().cpu().numpy().astype(np.float32)
            np.save(pgd_path, mask_np)
            logger.info("Saved PGD cache to %s", pgd_path)

    elif args.mask_type == "road":
        road_path = _road_cache_path(args.dataset, args.road_noise)
        if road_path.exists():
            logger.info("Loading cached ROAD from %s", road_path)
            mask_np = np.load(road_path)
        else:
            logger.info("Running ROAD (noise_std=%.3f)...", args.road_noise)
            road_loader = DataLoader(
                TensorDataset(torch.tensor(X_test), torch.tensor(y_test, dtype=torch.long)),
                batch_size=args.batch_size, shuffle=False,
            )
            mask_np = road(road_loader, noise_std=args.road_noise)
            np.save(road_path, mask_np)
            logger.info("Saved ROAD cache to %s", road_path)

    # ── MoRF and LeRF ────────────────────────────────────────────────────────
    shared = dict(
        model=model, X=X_test, y=y_test, saliency=saliency,
        n_steps=args.n_steps, batch_size=args.batch_size,
        mask_np=mask_np, mask_type=args.mask_type,
    )
    morf = run_morf_lerf(**shared, mode="morf")
    lerf = run_morf_lerf(**shared, mode="lerf")

    # ── Save results ─────────────────────────────────────────────────────────
    out_dir = _output_dir(args.dataset, args.model, args.mask_type)
    with open(out_dir / f"{args.expl_method}_morf.pkl", "wb") as f:
        pickle.dump(morf, f)
    with open(out_dir / f"{args.expl_method}_lerf.pkl", "wb") as f:
        pickle.dump(lerf, f)
    logger.info("Results saved to: %s", out_dir)


if __name__ == "__main__":
    main()
