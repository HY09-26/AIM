#train_esc50.py
import os
import random
import argparse
from pathlib import Path

import numpy as np
import h5py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from model.audionet import AudioNet
from model.alexnet import AlexNet_Audio
from model.cnn14 import Cnn14, load_cnn14_pretrained
from model.res1dnet31 import Res1dNet31, load_res1dnet31_pretrained


# ============================================================
# Reproducibility
# ============================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Dataset utils
# ============================================================
def is_waveform(path):
    return "waveform" in path.lower()

def is_spectrogram(path):
    return "spectrogram" in path.lower()


class ESC50WaveDataset(Dataset):
    def __init__(self, split_txt, data_dir):
        self.paths = []
        with open(split_txt) as f:
            for line in f:
                p = line.strip()
                if p and is_waveform(p):
                    self.paths.append(os.path.join(data_dir, p))
        assert len(self.paths) > 0, f"Empty split: {split_txt}"

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        with h5py.File(self.paths[idx], "r") as f:
            x = f["data"][0, 0, 0]   # (T,)
            y = int(f["label"][0][0])

        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)   # (1, T)
        y = torch.tensor(y, dtype=torch.long)
        return x, y


class ESC50SpecDataset(Dataset):
    def __init__(self, split_txt, data_dir):
        self.paths = []
        with open(split_txt) as f:
            for line in f:
                p = line.strip()
                if p and is_spectrogram(p):
                    self.paths.append(os.path.join(data_dir, p))
        assert len(self.paths) > 0, f"Empty split: {split_txt}"

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        with h5py.File(self.paths[idx], "r") as f:
            x = f["data"][0, 0]      # (F, T)
            y = int(f["label"][0][0])

        x = torch.from_numpy(x).float()
        mean = x.mean()
        std  = x.std()
        x = (x - mean) / (std + 1e-6)
        x = x.unsqueeze(0) 
        y = torch.tensor(y, dtype=torch.long)
        return x, y


# ============================================================
# Loader
# ============================================================
def get_loader(split_txt, data_dir, model_name, batch_size, shuffle, num_workers):
    if model_name == "alexnet":
        ds = ESC50SpecDataset(split_txt, data_dir)
    elif model_name in ["audionet", "res1dnet31", "cnn14"]:
        ds = ESC50WaveDataset(split_txt, data_dir)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


# ============================================================
# Train / Eval
# ============================================================
def run_epoch(model, loader, loss_fn, optimizer=None, device="cuda"):
    train = optimizer is not None
    model.train() if train else model.eval()

    total, correct, loss_sum = 0, 0, 0.0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad()

        logits = model(x)
        loss = loss_fn(logits, y)

        if train:
            loss.backward()
            optimizer.step()

        loss_sum += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)

    return loss_sum / total, correct / total


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["alexnet", "audionet", "cnn14", "res1dnet31"], default="alexnet")
    parser.add_argument("--data_dir", default="data/ESC50/preprocessed_data")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    set_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    base = Path(args.data_dir)
    train_txt = base / f"ESC50_fold{args.fold}_train.txt"
    test_txt  = base / f"ESC50_fold{args.fold}_test.txt"

    train_loader = get_loader(train_txt, args.data_dir, args.model, args.batch_size, True, 4)
    test_loader  = get_loader(test_txt,  args.data_dir, args.model, args.batch_size, False, 4)

    # ----------------------------
    # Model
    # ----------------------------
    if args.model == "alexnet":
        model = AlexNet_Audio(num_classes=50)

    elif args.model == "audionet":
        model = AudioNet(num_classes=50)

    elif args.model == "cnn14":
        model = Cnn14(num_classes=50)
        load_cnn14_pretrained(model, "model/pretrained/Cnn14_mAP=0.431.pth")

    elif args.model == "res1dnet31":
        model = Res1dNet31(num_classes=50)
        load_res1dnet31_pretrained(model, ckpt_path="model/pretrained/Res1dNet31_mAP=0.365.pth")

    model = model.to(device)

    # ----------------------------
    # Optim
    # ----------------------------
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    #optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4) #for alexnet

    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    best_acc = -1.0
    best_epoch = -1

    # ----------------------------
    # Training loop
    # ----------------------------
    for ep in range(args.epochs):
        tr_loss, tr_acc = run_epoch(model, train_loader, loss_fn, optimizer, device=device)
        te_loss, te_acc = run_epoch(model, test_loader, loss_fn, optimizer=None, device=device)

        print(
            f"Epoch [{ep+1:03d}] "
            f"Train : loss={tr_loss:.4f}, acc={tr_acc:.4f} | "
            f"Test : loss={te_loss:.4f}, acc={te_acc:.4f}"
        )

        if te_acc > best_acc:
            best_acc = te_acc
            best_epoch = ep

            save_dir = Path("checkpoints/esc50") / args.model / f"fold_{args.fold}"
            save_dir.mkdir(parents=True, exist_ok=True)

            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "best_acc": float(best_acc),
                    "epoch": int(best_epoch),
                    "model": args.model,
                    "fold": args.fold,
                    "seed": 0,
                    "num_classes": 50,
                },
                save_dir / "best_model.pth",
                _use_new_zipfile_serialization=False,
            )

        
        scheduler.step()

    print(f"Best test acc: {best_acc:.4f} at epoch {best_epoch+1}")


if __name__ == "__main__":
    main()
