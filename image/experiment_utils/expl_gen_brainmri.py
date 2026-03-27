# ------------------------------------------------------------
# Brain MRI Image Explanation Generator
# Methods:
#   - Grad-CAM
#   - Grad-CAM++
#   - Gradients (Vanilla Saliency)
#   - SmoothGrad
#   - Random (baseline)
# ------------------------------------------------------------

import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from .model import get_efficientnet_b0, get_resnet50, get_repvgg_b0

from captum.attr import Saliency, NoiseTunnel

try:
    from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
except Exception as e:
    GradCAM = GradCAMPlusPlus = None
    ClassifierOutputTarget = None
    _CAM_IMPORT_ERROR = e


# ============================================================
# 0. Basic setup
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "Brain_MRI_Tumor")
CKPT_DIR  = os.path.join(PROJECT_ROOT, "experiment_utils", "checkpoints")
EXPL_DIR  = os.path.join(PROJECT_ROOT, "expl_image_brainmri")

os.makedirs(EXPL_DIR, exist_ok=True)


# ============================================================
# 1. Gradient-based methods
# ============================================================
def get_gradients(model, loader):
    model.eval()
    expls = []
    sal = Saliency(model)

    for x, y in tqdm(loader, desc="Gradients"):
        x = x.to(device).requires_grad_(True)
        y = y.to(device)
        model.zero_grad(set_to_none=True)

        e = sal.attribute(x, target=y)
        expls.append(e.detach().cpu().numpy())

    return np.concatenate(expls, axis=0)


def get_smoothgrad(model, loader, nt_samples=8, stdevs=0.01):
    model.eval()
    expls = []
    nt = NoiseTunnel(Saliency(model))

    for x, y in tqdm(loader, desc="SmoothGrad"):
        x = x.to(device).requires_grad_(True)
        y = y.to(device)
        model.zero_grad(set_to_none=True)

        e = nt.attribute(
            x,
            target=y,
            nt_type="smoothgrad",
            nt_samples=nt_samples,
            stdevs=stdevs,
        )
        expls.append(e.detach().cpu().numpy())

    return np.concatenate(expls, axis=0)


# ============================================================
# 2. CAM-based methods
# ============================================================
def _ensure_cam_available():
    if GradCAM is None or GradCAMPlusPlus is None:
        raise ImportError(
            "pytorch-grad-cam is required.\n"
            f"Original error: {_CAM_IMPORT_ERROR}"
        )


def _find_last_conv2d(model: nn.Module) -> nn.Module:
    # Grad-CAM target layer must output (B, C, H, W)
    for m in reversed(list(model.modules())):
        if isinstance(m, nn.Conv2d):
            return m
    raise RuntimeError("No nn.Conv2d layer found for CAM target layer.")


def _default_cam_target_layer(model, model_name):
    if model_name == "resnet50":
        return model.layer4[-1]
    if model_name == "efficientnet_b0":
        return model.features[-1]
    if model_name == "repvgg_b0":
        return _find_last_conv2d(model)
    raise ValueError(f"Unknown model_name: {model_name}")



def get_cam_maps(model, loader, cam_type, target_layer):
    _ensure_cam_available()
    model.eval()

    cam_cls = GradCAM if cam_type == "gradcam" else GradCAMPlusPlus
    cam = cam_cls(model=model, target_layers=[target_layer])

    all_maps = []
    for x, y in tqdm(loader, desc=cam_type.upper()):
        x = x.to(device)
        targets = [ClassifierOutputTarget(int(t)) for t in y]
        maps = cam(input_tensor=x, targets=targets)  # (B,H,W)
        all_maps.append(maps[:, None, :, :])          # (B,1,H,W)

    return np.concatenate(all_maps, axis=0)


# ============================================================
# 3. Random baseline
# ============================================================
def make_random_expl_like(ref):
    rand = np.zeros_like(ref)
    for i in range(ref.shape[0]):
        mu = ref[i].mean()
        sigma = ref[i].std() + 1e-8
        rand[i] = np.random.normal(mu, sigma, size=ref[i].shape)
    return rand


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":

    # =========================
    # Config
    # =========================
    model_name = "repvgg_b0"     # or "resnet50"
    ckpt_name  = "repvgg_b0_brain_mri_best.pth"

    batch_size = 8

    methods = [
        "gradcam",
        "gradcampp",
        "gradients",
        "smoothgrad",
        "random",
    ]

    # =========================
    # Dataset
    # =========================
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
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    num_classes = len(test_set.classes)
    print(f"#Test samples = {len(test_set)}, #classes = {num_classes}")

    # =========================
    # Model
    # =========================
    if model_name == "resnet50":
        model = torch.hub.load("pytorch/vision", "resnet50", weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "efficientnet_b0":
        model = torch.hub.load("pytorch/vision", "efficientnet_b0", weights=None)
        in_dim = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_dim, num_classes)

    elif model_name == "repvgg_b0":
        model = get_repvgg_b0(num_classes=num_classes, pretrained=False)

        in_dim = model.head.fc.in_features
        model.head.fc = nn.Linear(in_dim, num_classes)



    ckpt_path = os.path.join(CKPT_DIR, ckpt_name)
    print("Loading checkpoint:", ckpt_path)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model = model.to(device).eval()

    # =========================
    # Output dir
    # =========================
    out_dir = os.path.join(EXPL_DIR, model_name)
    os.makedirs(out_dir, exist_ok=True)

    # =========================
    # Run explanations
    # =========================
    grad_ref = None
    target_layer = _default_cam_target_layer(model, model_name)

    for m in methods:
        print(f"\n===== {m} =====")
        save_path = os.path.join(out_dir, f"{m}.npy")

        if m == "gradcam":
            expl = get_cam_maps(model, test_loader, "gradcam", target_layer)

        elif m == "gradcampp":
            expl = get_cam_maps(model, test_loader, "gradcampp", target_layer)

        elif m == "gradients":
            expl = get_gradients(model, test_loader)
            grad_ref = expl

        elif m == "smoothgrad":
            expl = get_smoothgrad(model, test_loader)

        elif m == "random":
            if grad_ref is None:
                raise RuntimeError("Random requires gradients first.")
            expl = make_random_expl_like(grad_ref)

        else:
            raise ValueError(m)

        np.save(save_path, expl)
        print(f"Saved {m}: {expl.shape}")

    print("\nAll Brain MRI explanations saved!")
    print("Saved to:", out_dir)
