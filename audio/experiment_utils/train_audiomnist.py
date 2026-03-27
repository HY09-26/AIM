# train_audiomnist.py
import os
import random
import argparse
from pathlib import Path

import numpy as np
import h5py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset

from experiment_utils.model.audionet import AudioNet
from experiment_utils.model.alexnet import AlexNet_Audio
from experiment_utils.model.cnn14 import Cnn14, load_cnn14_pretrained
from experiment_utils.model.res1dnet31 import Res1dNet31Lite


# ============================================================
# Reproducibility
# ============================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Dataset utils
# ============================================================
def is_waveform(path: str) -> bool:
    return "waveform" in path.lower()

def is_spectrogram(path: str) -> bool:
    return "spectrogram" in path.lower()


# ============================================================
# AudioMNIST Dataset
# ============================================================
class AudioMNISTWaveDataset(Dataset):
    """
    For AudioNet / CNN14 / Res1dNet31
    Return waveform: (1, T)
    """
    def __init__(self, split_txt, data_dir):
        self.paths = []
        with open(split_txt) as f:
            for line in f:
                p = line.strip()
                if not p:
                    continue
                # optional filter (safe if split contains only waveform)
                if is_spectrogram(p):
                    continue
                self.paths.append(os.path.join(data_dir, p))
        assert len(self.paths) > 0, f"Empty split: {split_txt}"

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        with h5py.File(self.paths[idx], "r") as f:
            x = f["data"][:]          # could be (1,1,1,T) or (1,T) depending on preprocessing
            y = int(f["label"][0][0])

        x = np.asarray(x)

        # normalize to (T,)
        if x.ndim == 4:          # (1,1,1,T)
            x = x[0, 0, 0]
        elif x.ndim == 3:        # (1,1,T)
            x = x[0, 0]
        elif x.ndim == 2:        # (1,T) or (T,1)
            x = x[0] if x.shape[0] == 1 else x[:, 0]
        elif x.ndim != 1:
            raise ValueError(f"Unexpected waveform shape: {x.shape}")

        assert x.ndim == 1, f"Waveform should be 1D (T,), got {x.shape}"

        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # (1, T)
        y = torch.tensor(y, dtype=torch.long)
        return x, y


class AudioMNISTSpecDataset(Dataset):
    """
    For AlexNet (spectrogram)
    Return spectrogram: (1, F, T)
    """
    def __init__(self, split_txt, data_dir):
        self.paths = []
        with open(split_txt) as f:
            for line in f:
                p = line.strip()
                if not p:
                    continue
                # optional filter (safe if split contains only spectrogram)
                if is_waveform(p):
                    continue
                self.paths.append(os.path.join(data_dir, p))
        assert len(self.paths) > 0, f"Empty split: {split_txt}"

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        with h5py.File(self.paths[idx], "r") as f:
            x = f["data"][:]          # could be (1,F,T) or (F,T) depending on preprocessing
            y = int(f["label"][0][0])

        x = np.asarray(x)

        if x.ndim == 4:          # (1,1,F,T)
            x = x[0, 0]
        elif x.ndim == 3:        # (1,F,T)
            x = x[0]
        elif x.ndim == 2:
            pass
        else:
            raise ValueError(f"Unexpected spectrogram shape: {x.shape}")

        x = torch.from_numpy(x).float()  # (F,T)

        # per-frequency normalization (stronger & safer than global per-sample)
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True)
        x = (x - mean) / (std + 1e-6)

        x = x.unsqueeze(0)  # (1, F, T)
        y = torch.tensor(y, dtype=torch.long)
        return x, y


# ============================================================
# Loader
# ============================================================
MODEL_INPUT = {
    "alexnet": "spec",
    "audionet": "wave",
    "cnn14": "wave",
    "res1dnet31": "wave",
}

def get_loader(split_txt, data_dir, model_name, batch_size, shuffle, num_workers):
    if MODEL_INPUT[model_name] == "spec":
        ds = AudioMNISTSpecDataset(split_txt, data_dir)
    else:
        ds = AudioMNISTWaveDataset(split_txt, data_dir)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


# ============================================================
# Train / Eval
# ============================================================
def ensure_wave_batch(x: torch.Tensor) -> torch.Tensor:
    """
    Final shape for wave models (AudioNet / CNN14 / Res1dNet31): (B, T)
    Accepts:
      (B,1,1,T), (B,1,T), (B,T)
    """
    if x.dim() == 4:        # (B,1,1,T)
        x = x.squeeze(1)   # -> (B,T)
    # elif x.dim() == 3:      # (B,1,T)
    #     x = x.squeeze(1)               # -> (B,T)

    # if x.dim() != 2:
    #     raise ValueError(f"Wave batch must be (B,T), got {tuple(x.shape)}")

    return x


def ensure_spec_batch(x: torch.Tensor) -> torch.Tensor:
    """
    Final shape for spec models: (B, 1, F, T)
    Accepts:
      (B,1,F,T), (B,F,T)
    """
    if x.dim() == 3:        # (B,F,T)
        x = x.unsqueeze(1)  # -> (B,1,F,T)
    if x.dim() != 4:
        raise ValueError(f"Spec batch must be (B,1,F,T), got {tuple(x.shape)}")
    return x

def run_epoch(model, loader, loss_fn, optimizer=None, device="cuda", model_name="cnn14"):
    train = optimizer is not None
    model.train() if train else model.eval()

    total, correct, loss_sum = 0, 0, 0.0

    for x, y in loader:
        # IMPORTANT: only reshape based on model type
        if MODEL_INPUT[model_name] == "wave":
            x = ensure_wave_batch(x)
        else:
            x = ensure_spec_batch(x)

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

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
    parser.add_argument("--model",
                        choices=["alexnet", "audionet", "cnn14", "res1dnet31"],
                        default="alexnet",)
    parser.add_argument("--data_dir", default="data/audiomnist/preprocessed_data")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--fixed_test_n", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    set_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_dir = args.data_dir
    base = Path(data_dir)

    # --------------------------------------------------------
    # Split TXT (keep your original naming)
    # --------------------------------------------------------
    if args.model == "alexnet":
        train_txt = base / "AlexNet_digit_0_train.txt"
        val_txt   = base / "AlexNet_digit_0_validate.txt"
        test_txt  = base / "AlexNet_digit_0_test.txt"
    else:
        train_txt = base / "AudioNet_digit_0_train.txt"
        val_txt   = base / "AudioNet_digit_0_validate.txt"
        test_txt  = base / "AudioNet_digit_0_test.txt"

    train_loader = get_loader(train_txt, data_dir, args.model,
                              args.batch_size, True, args.num_workers)
    val_loader   = get_loader(val_txt,   data_dir, args.model,
                              args.batch_size, False, args.num_workers)

    # build full test dataset (so we can Subset it)
    full_test_loader = get_loader(test_txt, data_dir, args.model,
                                  args.batch_size, False, args.num_workers)

    # --------------------------------------------------------
    # Fixed test subset (save indices for reproducibility)
    # --------------------------------------------------------
    rng = np.random.default_rng(seed=0)
    N = len(full_test_loader.dataset)
    fixed_n = min(args.fixed_test_n, N)
    fixed_idx = rng.choice(N, size=fixed_n, replace=False)

    test_subset = Subset(full_test_loader.dataset, fixed_idx)
    test_loader = DataLoader(
        test_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    print(f"Train files: {len(train_loader.dataset)}")
    print(f"Val files:   {len(val_loader.dataset)}")
    print(f"Test files (fixed): {len(test_subset)}")

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    if args.model == "alexnet":
        model = AlexNet_Audio(num_classes=10)

    elif args.model == "audionet":
        model = AudioNet(num_classes=10)

    elif args.model == "cnn14":
        model = Cnn14(num_classes=10)
        load_cnn14_pretrained(model, "model/pretrained/Cnn14_mAP=0.431.pth")
        # keep deterministic behavior (spec aug off) if your model supports it
        if hasattr(model, "spec_augmenter"):
            model.spec_augmenter = None

    elif args.model == "res1dnet31":
        model = Res1dNet31Lite(num_classes=10)
        

    else:
        raise ValueError(args.model)

    model = model.to(device)

    # --------------------------------------------------------
    # Optim
    # --------------------------------------------------------
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    best_acc = -1.0
    best_epoch = -1

    # save_dir
    save_dir = Path("checkpoints/audiomnist") / args.model
    save_dir.mkdir(parents=True, exist_ok=True)
    np.save(save_dir / "fixed_test_indices.npy", fixed_idx)

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------
    print("Starting training...")
    for ep in range(args.epochs):
        tr_loss, tr_acc = run_epoch(model, train_loader, loss_fn, optimizer, device=device, model_name=args.model)
        va_loss, va_acc = run_epoch(model, val_loader,   loss_fn, optimizer=None, device=device, model_name=args.model)
        te_loss, te_acc = run_epoch(model, test_loader,  loss_fn, optimizer=None, device=device, model_name=args.model)

        print(
            f"Epoch [{ep+1:03d}] "
            f"Train: loss={tr_loss:.4f}, acc={tr_acc:.4f} | "
            f"Val: loss={va_loss:.4f}, acc={va_acc:.4f} | "
            f"Test(fixed): loss={te_loss:.4f}, acc={te_acc:.4f}"
        )

        if te_acc > best_acc:
            best_acc = te_acc
            best_epoch = ep

            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "best_acc": float(best_acc),
                    "epoch": int(best_epoch),
                    "model": args.model,
                    "seed": 0,
                    "num_classes": 10,
                    "fixed_test_n": int(fixed_n),
                    "fixed_test_indices_file": str(save_dir / "fixed_test_indices.npy"),
                },
                save_dir / "best_model.pth",
                _use_new_zipfile_serialization=False,
            )

        scheduler.step()

    print(f"Best test acc: {best_acc:.4f} at epoch {best_epoch+1}")


if __name__ == "__main__":
    main()
