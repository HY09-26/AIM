# experiment_utils/expl_gen_imagenet.py
# ============================================================
# ImageNet explanation generator (pretrained, no checkpoints)
#
# Models:
#   - resnet50
#   - efficientnet_b0
#   - repvgg_b0
#
# Dataset:
#   - ImageNet validation set
#   - Fixed 500-image subset (imagenet_val_500.npy)
#
# Methods:
#   - Grad-CAM
#   - Grad-CAM++
#   - Gradients (Vanilla)
#   - SmoothGrad
#   - Random baseline
#
# Output:
#   expl_image/imagenet/<model_name>/<method>.npy
# ============================================================

import os
import argparse
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import (
    resnet50, ResNet50_Weights,
    efficientnet_b0, EfficientNet_B0_Weights,
)
import timm

from captum.attr import (
    Saliency,
    NoiseTunnel,
    IntegratedGradients,
)

try:
    from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
except Exception as e:
    GradCAM = GradCAMPlusPlus = ScoreCAM = None
    ClassifierOutputTarget = None
    _CAM_IMPORT_ERROR = e


# ============================================================
# Args
# ============================================================
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["resnet50", "efficientnet_b0", "repvgg_b0"]
    )
    parser.add_argument(
        "--imagenet_root",
        type=str,
        default="data/ImageNet",
        help="Path to ImageNet root (contains val/)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for explanation generation"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )
    return parser.parse_args()


# ============================================================
# Model
# ============================================================
def get_model(name: str, device):
    if name == "resnet50":
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

    elif name == "efficientnet_b0":
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

    elif name == "repvgg_b0":
        model = timm.create_model(
            "repvgg_b0",
            pretrained=True,
            num_classes=1000
        )
    else:
        raise ValueError(name)

    model.to(device)
    model.eval()
    return model


# ============================================================
# CAM helpers
# ============================================================
def _ensure_cam_available():
    if GradCAM is None:
        raise ImportError(
            "pytorch-grad-cam is required.\n"
            "Install with: pip install grad-cam\n"
            f"Original error: {_CAM_IMPORT_ERROR}"
        )


def _default_cam_target_layer(model, model_name: str):
    """
    Pick a reasonable last conv layer for CAM.
    """
    if model_name.startswith("resnet"):
        return model.layer4[-1]

    if model_name.startswith("efficientnet"):
        return model.features[-1]

    if model_name.startswith("repvgg"):
        # timm RepVGG is a ByobNet
        # last conv block is in model.stages[-1][-1]
        return model.stages[-1][-1]

    raise ValueError(f"Unknown model_name for CAM target layer: {model_name}")


# ============================================================
# Gradient-based explanations
# ============================================================
def get_gradients(model, loader, abs_val=False):
    sal = Saliency(model)
    expls = []

    for x, y in tqdm(loader, desc="Gradients"):
        x = x.cuda()
        y = y.cuda()
        x.requires_grad_(True)

        e = sal.attribute(x, target=y, abs=abs_val)
        expls.append(e.detach().cpu().numpy())

    return np.concatenate(expls, axis=0)


def get_smoothgrad(model, loader, abs_val=False,
                   nt_samples=8, nt_batch_size=8, stdevs=0.01):
    base = Saliency(model)
    nt = NoiseTunnel(base)
    expls = []

    for x, y in tqdm(loader, desc="SmoothGrad"):
        x = x.cuda()
        y = y.cuda()
        x.requires_grad_(True)

        e = nt.attribute(
            x,
            target=y,
            nt_type="smoothgrad",
            nt_samples=nt_samples,
            nt_samples_batch_size=nt_batch_size,
            stdevs=stdevs,
            abs=abs_val,
        )
        expls.append(e.detach().cpu().numpy())

    return np.concatenate(expls, axis=0)


def get_integrated_gradients(model, loader, abs_val=False, steps=8):
    ig = IntegratedGradients(model)
    expls = []

    for x, y in tqdm(loader, desc="IntegratedGradients"):
        x = x.cuda()
        y = y.cuda()
        x.requires_grad_(True)

        e = ig.attribute(
            x,
            target=y,
            n_steps=steps,
            method="gausslegendre",
        )
        if abs_val:
            e = e.abs()

        expls.append(e.detach().cpu().numpy())

    return np.concatenate(expls, axis=0)


def get_blur_integrated_gradients(model, loader, abs_val=False,
                                 steps=50, max_sigma=50.0):
    big = BlurIntegratedGradients(model)
    expls = []

    for x, y in tqdm(loader, desc="BlurIG"):
        x = x.cuda()
        y = y.cuda()
        x.requires_grad_(True)

        e = big.attribute(
            x,
            target=y,
            n_steps=steps,
            max_sigma=max_sigma,
        )
        if abs_val:
            e = e.abs()

        expls.append(e.detach().cpu().numpy())

    return np.concatenate(expls, axis=0)


# ============================================================
# CAM-based explanations
# ============================================================
def get_cam_maps(model, loader, model_name, cam_type="gradcam"):
    _ensure_cam_available()
    target_layer = _default_cam_target_layer(model, model_name)
    #print("CAM target layer:", target_layer)


    if cam_type == "gradcam":
        cam = GradCAM(model, [target_layer])
    elif cam_type == "gradcampp":
        cam = GradCAMPlusPlus(model, [target_layer])
    elif cam_type == "scorecam":
        cam = ScoreCAM(model, [target_layer])
    else:
        raise ValueError(cam_type)

    all_maps = []

    for x, y in tqdm(loader, desc=cam_type.upper()):
        x = x.cuda()
        targets = [ClassifierOutputTarget(int(cls)) for cls in y]
        maps = cam(input_tensor=x, targets=targets)
        maps = maps[:, None, :, :]
        all_maps.append(maps.astype(np.float32))

    return np.concatenate(all_maps, axis=0)


# ============================================================
# Random baseline
# ============================================================
def make_random_expl_like(ref):
    rand = np.zeros_like(ref)
    for i in range(ref.shape[0]):
        mu, sigma = ref[i].mean(), ref[i].std()
        sigma = max(sigma, 1e-8)
        rand[i] = np.random.normal(mu, sigma, size=ref[i].shape)
    return rand


# ============================================================
# Main
# ============================================================
def main():
    args = get_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(">>> device =", device)

    # --------------------------------------------------
    # Dataset + fixed subset
    # --------------------------------------------------
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    ])

    val_dir = os.path.join(args.imagenet_root, "val")
    val_dataset = datasets.ImageFolder(val_dir, transform=val_tf)

    idx_path = os.path.join(args.imagenet_root, "imagenet_val_500.npy")
    assert os.path.exists(idx_path), f"Missing fixed subset: {idx_path}"
    indices = np.load(idx_path)

    val_subset = Subset(val_dataset, indices)
    loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f">>> Using fixed ImageNet subset: {len(val_subset)} samples")

    # --------------------------------------------------
    # Model
    # --------------------------------------------------
    model = get_model(args.model, device)

    # --------------------------------------------------
    # Output directory
    # --------------------------------------------------
    out_dir = os.path.join(
        "expl_image_imagenet",
        args.model
    )
    os.makedirs(out_dir, exist_ok=True)

    # --------------------------------------------------
    # Methods
    # --------------------------------------------------
    methods = [
        "gradcam",
        "gradcampp",
        "gradients",
        "smoothgrad",
        "random",
    ]

    grad_ref = None

    for m in methods:
        print(f"\n===== Method: {m} =====")
        save_path = os.path.join(out_dir, f"{m}.npy")

        if m == "gradcam":
            expl = get_cam_maps(model, loader, args.model, "gradcam")

        elif m == "gradcampp":
            expl = get_cam_maps(model, loader, args.model, "gradcampp")

        elif m == "gradients":
            expl = get_gradients(model, loader, abs_val=False)
            grad_ref = expl

        elif m == "smoothgrad":
            expl = get_smoothgrad(model, loader, abs_val=False)

        elif m == "integrad":
            expl = get_integrated_gradients(model, loader, abs_val=False)

        elif m == "blur_integrad":
            expl = get_blur_integrated_gradients(model, loader, abs_val=False)

        elif m == "random":
            assert grad_ref is not None, "Run gradients first for random baseline"
            expl = make_random_expl_like(grad_ref)

        else:
            raise ValueError(m)

        np.save(save_path, expl)
        print(f"[SAVED] {save_path}, shape={expl.shape}")

    print("\nAll ImageNet explanations generated successfully!")


if __name__ == "__main__":
    main()
