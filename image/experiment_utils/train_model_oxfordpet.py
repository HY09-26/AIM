
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Subset

from torch.utils.data import DataLoader
from torchvision import transforms

from .model import get_resnet50, get_efficientnet_b0, get_repvgg_b0     
from .pets_dataset import OxfordPetsDataset
from .utils import fit_classifier       
 


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(">>> main() started")

    # Paths
    BASE_DIR = os.path.join(os.environ["HOME"], "hsinyuan")
    DATA_ROOT = os.path.join(BASE_DIR, "AIM_image", "data", "Oxford_Pet")
    SAVE_DIR = os.path.join(BASE_DIR, "AIM_image", "experiment_utils", "checkpoints")

    # Dataset / Loader
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    test_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    train_set = OxfordPetsDataset(DATA_ROOT, split="train", transform=train_tf)

    # Fixed random 500-sample test subset
    test_set  = OxfordPetsDataset(DATA_ROOT, split="test",  transform=test_tf)
    NUM_TEST_SAMPLES = 500
    RANDOM_SEED = 42   

    rng = np.random.RandomState(RANDOM_SEED)
    all_indices = np.arange(len(test_set))
    subset_indices = rng.choice(all_indices, size=NUM_TEST_SAMPLES, replace=False)
    test_set = Subset(test_set, subset_indices)
    np.save(os.path.join(SAVE_DIR, "fixed_test_indices_500.npy"), subset_indices)



    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4)
    test_loader  = DataLoader(test_set,  batch_size=32, shuffle=False, num_workers=4)

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    # model = get_resnet50(num_classes=37, pretrained=True).to(device)
    # model = get_efficientnet_b0(num_classes=37, pretrained=True).to(device)
    model = get_repvgg_b0(num_classes=37, pretrained=True).to(device)
  


    # --------------------------------------------------------
    # Training setup
    # --------------------------------------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    EPOCHS = 15

    # --------------------------------------------------------
    # Train (SAVE BEST)
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
        ckpt_best="repvgg_pets_best.pth",
        ckpt_last="repvgg_pets_last.pth",
        task="multiclass",
    )


if __name__ == "__main__":
    
    main()


