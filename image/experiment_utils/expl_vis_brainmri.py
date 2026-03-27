import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# --------------------------------------------------
# Utilities
# --------------------------------------------------
def normalize_heatmap(hm, eps=1e-8):
    hm = hm - hm.min()
    hm = hm / (hm.max() + eps)
    return hm


def reduce_attr_to_heatmap(attr):
    """
    attr: (H,W) or (C,H,W) or (1,H,W)
    return: (H,W) normalized heatmap
    """
    if attr.ndim == 2:
        hm = attr
    elif attr.ndim == 3:
        hm = np.abs(attr).max(axis=0)
    else:
        raise ValueError(f"Unexpected attr shape: {attr.shape}")

    return normalize_heatmap(hm)


def load_image(img_path):
    """
    Brain MRI image → RGB (for visualization consistency)
    """
    return Image.open(img_path).convert("RGB")


def preprocess_for_display(img):
    """
    PIL Image -> np.ndarray (H,W,3) in [0,1]
    """
    img = img.resize((224, 224))
    img = np.asarray(img).astype(np.float32) / 255.0
    return img


# --------------------------------------------------
# Visualization
# --------------------------------------------------
def visualize_methods(
    img_path,
    expl_dir,
    methods,
    sample_idx,
    save_path=None,
    alpha=0.5
):
    """
    Original image + multiple explanation overlays
    """

    img = load_image(img_path)
    img_np = preprocess_for_display(img)

    n_panels = 1 + len(methods)
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    axes = axes.flatten()

    # --- Original image ---
    axes[0].imshow(img_np)
    axes[0].set_title("Original MRI")
    axes[0].axis("off")

    # --- Explanation methods ---
    for i, m in enumerate(methods):
        ax = axes[i + 1]

        expl_path = os.path.join(expl_dir, f"{m}.npy")
        assert os.path.exists(expl_path), f"Missing: {expl_path}"

        expl = np.load(expl_path)  # (N,C,H,W) or (N,1,H,W)
        attr = expl[sample_idx]
        hm = reduce_attr_to_heatmap(attr)

        ax.imshow(img_np)
        ax.imshow(hm, cmap="jet", alpha=alpha)
        ax.set_title(m)
        ax.axis("off")

    # --- Turn off unused panels ---
    for j in range(n_panels, 9):
        axes[j].axis("off")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200)
        print(f"[INFO] Saved figure to: {save_path}")

    plt.show()


# --------------------------------------------------
# Main (Brain MRI)
# --------------------------------------------------
if __name__ == "__main__":

    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # --------------------------------------------------
    # CONFIG
    # --------------------------------------------------
    model_name = "repvgg_b0"   # or "resnet50"
    class_name = "no_tumor"      # folder name under Testing/

    SAMPLE_IDX = 19              # index in .npy (and in ImageFolder order)
    alpha = 0.5

    methods = [
        "gradcam",
        "gradcampp",
        "gradients",
        "smoothgrad",
        "random",
    ]

    # --------------------------------------------------
    # Paths
    # --------------------------------------------------
    IMG_DIR = os.path.join(
        PROJECT_ROOT,
        "data",
        "Brain_MRI_Tumor",
        "Testing",
        class_name
    )

    EXPL_DIR = os.path.join(
        PROJECT_ROOT,
        "expl_image_brainmri",
        model_name
    )

    # --------------------------------------------------
    # Image list (ImageFolder ordering)
    # --------------------------------------------------
    img_list = sorted([
        f for f in os.listdir(IMG_DIR)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ])

    assert SAMPLE_IDX < len(img_list), "SAMPLE_IDX out of range"

    IMG_NAME = img_list[SAMPLE_IDX]
    IMG_PATH = os.path.join(IMG_DIR, IMG_NAME)

    print(f"[INFO] Model      : {model_name}")
    print(f"[INFO] Class      : {class_name}")
    print(f"[INFO] SAMPLE_IDX : {SAMPLE_IDX}")
    print(f"[INFO] Image file : {IMG_NAME}")

    SAVE_FIG = os.path.join(
        EXPL_DIR,
        f"saliency_{class_name}_{SAMPLE_IDX:03d}.png"
    )

    # --------------------------------------------------
    # Visualize
    # --------------------------------------------------
    visualize_methods(
        img_path=IMG_PATH,
        expl_dir=EXPL_DIR,
        methods=methods,
        sample_idx=SAMPLE_IDX,
        save_path=SAVE_FIG,
        alpha=alpha
    )
