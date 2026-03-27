# experiment_utils/train_model_chestxray.py
# ============================================================
# Train ResNet-50 / EfficientNet-B0 on NIH Chest X-ray (LT)
# - Train: CSV + PNG (lazy loading)
# - Test : fixed 500 samples (preprocessed npy)
# - Baseline training ONLY (no finetune)
# ============================================================

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset

from .model import get_efficientnet_b0, get_repvgg_b0, get_resnet50
from .utils import fit_classifier
from .chestxray_dataset import ChestXrayTrainDataset


# ------------------------------------------------------------
# Local helper (avoid touching global utils)
# ------------------------------------------------------------
def build_loader_from_np(X_np, y_np, batch_size):
    x = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.long)
    ds = TensorDataset(x, y)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )


def main():
    # --------------------------------------------------------
    # Args
    # --------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        choices=["resnet50", "efficientnet_b0", "repvgg_b0"],
        required=True,
        help="Model backbone"
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(">>> main() started")
    print(f">>> model = {args.model}")

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------
    BASE_DIR = os.path.join(os.environ["HOME"], "hsinyuan")
    DATA_ROOT = os.path.join(BASE_DIR, "AIM_image", "data", "chestxray")
    SAVE_DIR = os.path.join(
        BASE_DIR, "AIM_image", "experiment_utils", "checkpoints"
    )
    os.makedirs(SAVE_DIR, exist_ok=True)

    #TRAIN_CSV = os.path.join(DATA_ROOT, "nih-cxr-lt_single-label_train.csv")
    IMG_DIR = os.path.join(DATA_ROOT, "images", "images")

    PREP_DIR = os.path.join(DATA_ROOT, "preprocessed")
    TEST_X = os.path.join(PREP_DIR, "test_images_500.npy")
    TEST_Y = os.path.join(PREP_DIR, "test_labels_500.npy")

    # --------------------------------------------------------
    # Dataset / Loader (Train)
    # --------------------------------------------------------
    train_dataset = ChestXrayTrainDataset(
        label_txt_path="data/chestxray/train_label_nf10k.txt",
        img_dir=IMG_DIR
        )


    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=16,
        pin_memory=False
    )

    num_classes = train_dataset.num_classes
    print(f">>> num_classes = {num_classes}")

    # --------------------------------------------------------
    # Test loader (Fixed 500)
    # --------------------------------------------------------
    X_test = np.load(TEST_X)
    y_test = np.load(TEST_Y)

    test_loader = build_loader_from_np(
        X_test, y_test, batch_size=32
    )

    # --------------------------------------------------------
    # Model init (Baseline)
    # --------------------------------------------------------
    if args.model == "resnet50":
        model = get_resnet50(
            num_classes=num_classes,
            pretrained=False
        ).to(device)
        ckpt_best = "resnet50_chestxray_best.pth"
        ckpt_last = "resnet50_chestxray_last.pth"

    elif args.model == "efficientnet_b0":
        model = get_efficientnet_b0(
            num_classes=num_classes,
            pretrained=False
        ).to(device)
        ckpt_best = "efficientnet_chestxray_best.pth"
        ckpt_last = "efficientnet_chestxray_last.pth"

    elif args.model == "repvgg_b0":
        model = get_repvgg_b0(
            num_classes=num_classes,
            pretrained=False
        ).to(device)
        ckpt_best = "repvgg_b0_chestxray_best.pth"
        ckpt_last = "repvgg_b0_chestxray_last.pth"

    # --------------------------------------------------------
    # Optimizer / Training setup (GUARANTEED)
    # --------------------------------------------------------
    optimizer = optim.Adam(model.parameters(), lr=8e-4)
    criterion = nn.CrossEntropyLoss()
    EPOCHS = 10

    print(f">>> optimizer ready, epochs = {EPOCHS}")

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------
    fit_classifier(
        model=model,
        device=device,
        train_loader=train_loader,
        eval_loader=test_loader,
        loss_fn=criterion,
        optimizer=optimizer,
        epochs=EPOCHS,
        ckpt_dir=SAVE_DIR,
        ckpt_best=ckpt_best,
        ckpt_last=ckpt_last,
        task="multiclass",
    )


if __name__ == "__main__":
    main()
