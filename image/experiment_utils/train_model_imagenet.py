# experiment_utils/train_model_imagenet.py
# ===========================================================
# Using pretrained models to evaluate on ImageNet validation set
# No training is performed in this script
# - Model: ResNet-50 / EfficientNet-B0 / RepVGG-B0
# - Dataset: ImageNet val (fixed 500 samples)
# ===========================================================
import argparse
import random
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import (
    resnet50, ResNet50_Weights,
    efficientnet_b0, EfficientNet_B0_Weights,
)
import timm


# --------------------------------------------------
# Args
# --------------------------------------------------
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, required=True,
        choices=["resnet50", "efficientnet_b0", "repvgg_b0"]
    )
    parser.add_argument(
        "--imagenet_root", type=str, default="data/ImageNet",
        help="Path to ImageNet root, containing val/"
    )
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# --------------------------------------------------
# Model
# --------------------------------------------------
def get_model(name, device):
    if name == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V2
        model = resnet50(weights=weights)

    elif name == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        model = efficientnet_b0(weights=weights)

    elif name == "repvgg_b0":
        model = timm.create_model(
            "repvgg_b0",
            pretrained=True,
            num_classes=1000
        )
    else:
        raise ValueError(name)

    model.to(device)
    model.eval()   # IMPORTANT: eval mode only
    return model


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    args = get_args()

    # reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f">>> device = {device}")

    # ImageNet validation transform
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    ])

    # Dataset
    val_dir = os.path.join(args.imagenet_root, "val")
    val_dataset = datasets.ImageFolder(val_dir, transform=val_tf)
    print(f">>> ImageNet val size = {len(val_dataset)}")

    # --------------------------------------------------
    # Fixed ImageNet validation subset (500 images)
    # --------------------------------------------------
    IDX_PATH = os.path.join(
        args.imagenet_root,
        f"imagenet_val_{args.num_samples}.npy"
    )

    if os.path.exists(IDX_PATH):
        print(f">>> Loading fixed ImageNet val subset from {IDX_PATH}")
        indices = np.load(IDX_PATH).tolist()
    else:
        print(">>> Sampling and saving fixed ImageNet val subset")
        indices = list(range(len(val_dataset)))
        random.shuffle(indices)
        indices = indices[:args.num_samples]

        np.save(IDX_PATH, np.array(indices))
        print(f">>> Saved subset indices to {IDX_PATH}")

    val_subset = Subset(val_dataset, indices)

    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    # Model
    model = get_model(args.model, device)

    # Evaluation (Top-1 Accuracy)
    correct = 0
    total = 0

    torch.set_grad_enabled(False)
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            preds = logits.argmax(dim=1)

            correct += (preds == y).sum().item()
            total += y.size(0)

    acc = correct / total * 100.0
    print(f">>> Model: {args.model}")
    print(f">>> Samples: {total}")
    print(f">>> Top-1 Accuracy: {acc:.2f}%")
    print(">>> Done (no training performed)")


if __name__ == "__main__":
    main()
