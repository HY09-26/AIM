# expl_gen_oxfordpet.py
# ------------------------------------------------------------
# Image-version expl generator
# Methods:
#   - Grad-CAM
#   - Grad-CAM++
#   - Gradients (Vanilla Saliency)
#   - SmoothGrad
#   - Random baseline
#
# Output:
#   Saves each method as .npy under EXPL_DIR/<model_name>/<method>.npy
#
# Notes:
#   1) Dataloader must yield (x, y)
#   2) x shape: (B,3,H,W), y shape: (B,)
#   3) x must be normalized correctly
# ------------------------------------------------------------

import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision
import timm

from torch.utils.data import Subset, DataLoader
from captum.attr import Saliency, NoiseTunnel, IntegratedGradients

try:
    from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
except Exception as e:
    GradCAM = GradCAMPlusPlus = ScoreCAM = None
    ClassifierOutputTarget = None
    _CAM_IMPORT_ERROR = e


# ============================================================
# 0. Basic setup
# ============================================================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SAVE_DIR = os.path.join(PROJECT_ROOT, "experiment_utils", "checkpoints")
EXPL_DIR = os.path.join(PROJECT_ROOT, "expl_image_oxfordpet")


# ============================================================
# 1) Simple eval
# ============================================================
@torch.no_grad()
def evaluate_top1(model, loader):
    model.eval()
    correct, total = 0, 0
    for x, y in tqdm(loader, desc="Eval", leave=False):
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


# ============================================================
# 2) Gradient-based explanations
# ============================================================
def get_gradients(model, loader, abs_val=False):
    model.eval()
    sal = Saliency(model)
    outs = []

    for x, y in tqdm(loader, desc="Gradients"):
        x, y = x.to(device), y.to(device)
        x.requires_grad_(True)
        model.zero_grad(set_to_none=True)
        e = sal.attribute(x, target=y, abs=abs_val)
        outs.append(e.detach().cpu().numpy())

    return np.concatenate(outs, axis=0)


def get_smoothgrad(model, loader, nt_samples=8, nt_batch_size=4, stdevs=0.01, abs_val=False):
    model.eval()
    base = Saliency(model)
    nt = NoiseTunnel(base)
    outs = []

    for x, y in tqdm(loader, desc="SmoothGrad"):
        x, y = x.to(device), y.to(device)
        x.requires_grad_(True)
        model.zero_grad(set_to_none=True)

        e = nt.attribute(
            x,
            target=y,
            nt_type="smoothgrad",
            nt_samples=nt_samples,
            nt_samples_batch_size=nt_batch_size,
            stdevs=stdevs,
            abs=abs_val,
        )
        outs.append(e.detach().cpu().numpy())

    return np.concatenate(outs, axis=0)


def get_integrated_gradients(model, loader, steps=8, abs_val=False):
    model.eval()
    ig = IntegratedGradients(model)
    outs = []

    for x, y in tqdm(loader, desc="IntegratedGradients"):
        x, y = x.to(device), y.to(device)
        x.requires_grad_(True)
        model.zero_grad(set_to_none=True)

        e = ig.attribute(x, target=y, n_steps=steps)
        if abs_val:
            e = e.abs()

        outs.append(e.detach().cpu().numpy())

    return np.concatenate(outs, axis=0)


# ============================================================
# 3) CAM-based
# ============================================================
def _ensure_cam_available():
    if GradCAM is None:
        raise ImportError(f"grad-cam not installed: {_CAM_IMPORT_ERROR}")


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


def get_cam_maps(model, loader, cam_type, target_layer):
    _ensure_cam_available()
    model.eval()

    if cam_type == "gradcam":
        cam = GradCAM(model, [target_layer])
    elif cam_type == "gradcampp":
        cam = GradCAMPlusPlus(model, [target_layer])
    elif cam_type == "scorecam":
        cam = ScoreCAM(model, [target_layer])
    else:
        raise ValueError(cam_type)

    outs = []
    for x, y in tqdm(loader, desc=cam_type.upper()):
        x, y = x.to(device), y.to(device)
        targets = [ClassifierOutputTarget(int(i)) for i in y]
        m = cam(input_tensor=x, targets=targets)
        outs.append(m[:, None, :, :].astype(np.float32))

    return np.concatenate(outs, axis=0)


# ============================================================
# 4) Random baseline
# ============================================================
def make_random_expl_like(ref):
    rand = np.zeros_like(ref)
    for i in range(ref.shape[0]):
        mu, sigma = ref[i].mean(), max(ref[i].std(), 1e-8)
        rand[i] = np.random.normal(mu, sigma, size=ref[i].shape)
    return rand


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    # =========================
    # Choose model
    # =========================
    model_name = "repvgg_b0"
    num_classes = 37

    # =========================
    # Dataset
    # =========================
    from torchvision import transforms
    from .pets_dataset import OxfordPetsDataset

    test_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        ),
    ])

    fixed_idx = np.load(os.path.join(SAVE_DIR, "fixed_test_indices_500.npy"))

    full_test = OxfordPetsDataset(
        root=os.path.join(PROJECT_ROOT, "data", "Oxford_Pet"),
        split="test",
        transform=test_tf
    )

    test_set = Subset(full_test, fixed_idx)
    test_loader = DataLoader(test_set, batch_size=8, shuffle=False, num_workers=4)

    # =========================
    # Model
    # =========================
    model = timm.create_model(
        "repvgg_b0",
        pretrained=False,
        num_classes=num_classes
    )

    ckpt_path = os.path.join(SAVE_DIR, "repvgg_pets_best.pth")
    print("Loading:", ckpt_path)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)

    model = model.to(device).eval()
    print("Num classes:", model.head.fc.out_features)

    acc = evaluate_top1(model, test_loader)
    print(f"Test Acc: {acc*100:.2f}%")

    # =========================
    # Explanation
    # =========================
    model_expl_dir = os.path.join(EXPL_DIR, model_name)
    os.makedirs(model_expl_dir, exist_ok=True)

    target_layer = _default_cam_target_layer(model, model_name)

    grad_ref = None
    methods = ["gradcam", "gradcampp", "gradients", "smoothgrad", "random"]

    for m in methods:
        print(f"\n===== {m} =====")
        save_path = os.path.join(model_expl_dir, f"{m}.npy")

        if m in ["gradcam", "gradcampp"]:
            expl = get_cam_maps(model, test_loader, m, target_layer)
        elif m == "gradients":
            expl = get_gradients(model, test_loader)
            grad_ref = expl
        elif m == "smoothgrad":
            expl = get_smoothgrad(model, test_loader)
        elif m == "random":
            expl = make_random_expl_like(grad_ref)
        else:
            continue

        np.save(save_path, expl)
        print("Saved:", save_path)

    print("\nAll explanations done.")
