import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from torchvision import datasets, transforms
from torch.utils.data import Subset
from PIL import Image
import random


# ============================================================
# Basic paths
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IMAGENET_ROOT = os.path.join(PROJECT_ROOT, "data", "ImageNet")
EXPL_ROOT     = os.path.join(PROJECT_ROOT, "expl_image_imagenet")
SAVE_DIR      = os.path.join(PROJECT_ROOT, "expl_image_imagenet")

os.makedirs(SAVE_DIR, exist_ok=True)


# ============================================================
# Heatmap utils 
# ============================================================
def normalize_heatmap(hm, eps=1e-8):
    hm = hm - hm.min()
    hm = hm / (hm.max() + eps)
    return hm


def reduce_attr_to_heatmap(attr):
    """
    attr: (H,W) or (C,H,W) or (1,H,W)
    """
    if attr.ndim == 2:
        hm = np.abs(attr)
    elif attr.ndim == 3:
        hm = np.abs(attr).max(axis=0)
    else:
        raise ValueError(attr.shape)

    return normalize_heatmap(hm)


def denormalize(img):
    """
    img: tensor-style np (3,H,W) normalized by ImageNet stats
    """
    mean = np.array([0.485, 0.456, 0.406])[:, None, None]
    std  = np.array([0.229, 0.224, 0.225])[:, None, None]
    img = img * std + mean
    img = np.clip(img, 0, 1)
    return np.transpose(img, (1, 2, 0))


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":

    # -----------------------------
    # Config
    # -----------------------------
    models = ["resnet50", "efficientnet_b0", "repvgg_b0"]
    methods = ["gradcam", "gradcampp", "gradients", "smoothgrad", "random"]

    alpha = 0.5
    cmap = plt.get_cmap("jet")

    # -----------------------------
    # Dataset (ImageNet val subset)
    # -----------------------------
    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225],
        )
    ])

    full_val = datasets.ImageFolder(
        os.path.join(IMAGENET_ROOT, "val"),
        transform=tf
    )

    fixed_idx = np.load(os.path.join(IMAGENET_ROOT, "imagenet_val_500.npy"))
    subset = Subset(full_val, fixed_idx)

    # -----------------------------
    # Pick ONE sample (random every run)
    # -----------------------------
    sample_idx = random.choice(range(len(subset)))
    x, y = subset[sample_idx]
    img = denormalize(x.numpy())

    print(f"[INFO] Visualizing sample idx = {sample_idx}")

    # ============================================================
    # Figure layout
    # ============================================================
    n_rows = len(models)
    n_cols = 1 + len(methods)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.2 * n_cols, 3.2 * n_rows)
    )

    # ============================================================
    # Plot
    # ============================================================
    for r, model in enumerate(models):
        # --- Original ---
        ax = axes[r, 0]
        ax.imshow(img)
        ax.set_title(f"{model}\nOriginal", fontsize=10)
        ax.axis("off")

        # --- Methods ---
        for c, m in enumerate(methods):
            ax = axes[r, c + 1]

            expl_path = os.path.join(EXPL_ROOT, model, f"{m}.npy")
            expl = np.load(expl_path)

            hm = reduce_attr_to_heatmap(expl[sample_idx])

            ax.imshow(img)
            ax.imshow(hm, cmap=cmap, vmin=0, vmax=1, alpha=alpha)
            ax.set_title(m, fontsize=9)
            ax.axis("off")



    save_path = os.path.join(
        SAVE_DIR,
        "plots",
        f"sample_{sample_idx:04d}_3models.png"
    )
    plt.savefig(save_path, dpi=200)
    plt.show()

    print(f"[DONE] Saved to {save_path}")
