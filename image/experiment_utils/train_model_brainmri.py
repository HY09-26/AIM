import os
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from .model import get_efficientnet_b0, get_resnet50, get_repvgg_b0
from .utils import fit_classifier



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
        help="Model backbone to use"
    )
    parser.add_argument(
        "--finetune",
        action="store_true",
        help="Fine-tune on top of baseline checkpoint"
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(">>> main() started")
    print(f">>> model    = {args.model}")
    print(f">>> finetune = {args.finetune}")

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------
    BASE_DIR = os.path.join(os.environ["HOME"], "hsinyuan")
    DATA_ROOT = os.path.join(
        BASE_DIR, "AIM_image", "data", "Brain_MRI_Tumor"
    )
    SAVE_DIR = os.path.join(
        BASE_DIR, "AIM_image", "experiment_utils", "checkpoints"
    )
    os.makedirs(SAVE_DIR, exist_ok=True)

    # --------------------------------------------------------
    # Transforms
    # --------------------------------------------------------
    train_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(30),
        transforms.RandomAffine(degrees=0, translate=(0.05,0.05), scale=(0.9,1.1)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])

    test_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])

    # --------------------------------------------------------
    # Dataset / Loader
    # --------------------------------------------------------
    train_set = ImageFolder(
        root=os.path.join(DATA_ROOT, "Training"),
        transform=train_tf
    )
    test_set = ImageFolder(
        root=os.path.join(DATA_ROOT, "Testing"),
        transform=test_tf
    )

    train_loader = DataLoader(
        train_set, batch_size=32, shuffle=True, num_workers=4
    )
    test_loader = DataLoader(
        test_set, batch_size=32, shuffle=False, num_workers=4
    )

    num_classes = len(train_set.classes)
    print(f">>> num_classes = {num_classes}")

    # --------------------------------------------------------
    # Model init
    # --------------------------------------------------------
    if args.model == "resnet50":
        model = get_resnet50(
            num_classes=num_classes,
            pretrained=True
        ).to(device)

        baseline_ckpt = "resnet50_brain_mri_best.pth"
        finetune_ckpt_best = "resnet50_brain_mri_finetune_best.pth"
        finetune_ckpt_last = "resnet50_brain_mri_finetune_last.pth"

    elif args.model == "efficientnet_b0":
        model = get_efficientnet_b0(
            num_classes=num_classes,
            pretrained=True
        ).to(device)

        baseline_ckpt = "efficientnet_brain_mri_best.pth"
        finetune_ckpt_best = "efficientnet_brain_mri_finetune_best.pth"
        finetune_ckpt_last = "efficientnet_brain_mri_finetune_last.pth"

    elif args.model == "repvgg_b0":
        model = get_repvgg_b0(
            num_classes=num_classes,
            pretrained=True
        ).to(device)

        baseline_ckpt = "repvgg_b0_brain_mri_best.pth"
        finetune_ckpt_best = "repvgg_b0_brain_mri_finetune_best.pth"
        finetune_ckpt_last = "repvgg_b0_brain_mri_finetune_last.pth"

    # --------------------------------------------------------
    # Fine-tune or baseline
    # --------------------------------------------------------
    if args.finetune:

        # ---- freeze all ----
        for p in model.parameters():
            p.requires_grad = False

        # ---- architecture-specific unfreeze ----
        if args.model == "resnet50":
            for p in model.layer3.parameters():
                p.requires_grad = True
            for p in model.layer4.parameters():
                p.requires_grad = True
            for p in model.fc.parameters():
                p.requires_grad = True

        elif args.model == "efficientnet_b0":
            for p in model.features[-1].parameters():
                p.requires_grad = True
            for p in model.classifier.parameters():
                p.requires_grad = True

        trainable_params = filter(
            lambda p: p.requires_grad, model.parameters()
        )

        optimizer = optim.Adam(trainable_params, lr=1e-4)
        EPOCHS = 20

        ckpt_best = finetune_ckpt_best
        ckpt_last = finetune_ckpt_last

    else:
        # ---- baseline training ----
        optimizer = optim.Adam(model.parameters(), lr=8e-5)
        EPOCHS = 15

        ckpt_best = baseline_ckpt
        ckpt_last = baseline_ckpt.replace("_best", "_last")

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------
    criterion = nn.CrossEntropyLoss()

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
