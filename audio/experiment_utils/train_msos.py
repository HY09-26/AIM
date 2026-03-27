# train_msos.py
import os
import argparse
from pathlib import Path

import numpy as np
import h5py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchaudio

from model.cnn14 import Cnn14, load_cnn14_pretrained
from model.res1dnet31 import Res1dNet31, load_res1dnet31_pretrained
from model.alexnet import AlexNet_Audio
from model.audionet import AudioNet


# ============================================================
# Reproducibility
# ============================================================
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Dataset
# ============================================================
import os
import h5py
import torch
import torchaudio
from torch.utils.data import Dataset


class MSOSDataset(Dataset):
    """
    MSOS Dataset
    - alexnet   -> linear STFT spectrogram (1, F, T)
    - audionet  -> waveform (1, T)
    - res1dnet31 / cnn14 -> waveform (1, T)
    """

    def __init__(self, split_txt, data_dir, model_type, sr=44100):
        self.paths = []
        with open(split_txt) as f:
            for line in f:
                p = line.strip()
                if p:
                    self.paths.append(os.path.join(data_dir, p))

        assert len(self.paths) > 0, f"Empty split file: {split_txt}"

        self.model_type = model_type
        self.sr = sr

        # ---------- spectrogram for AlexNet  ----------
        if self.model_type == "alexnet":
            self.spec = torchaudio.transforms.Spectrogram(
                n_fft=1024,
                hop_length=512,
                power=1
            )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        with h5py.File(self.paths[idx], "r") as f:
            x = f["data"][0, 0, 0]       # (T,)
            y = int(f["label"][0][0])

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)

        # ----------------------------------------------------
        # Model-specific preprocessing
        # ----------------------------------------------------
        if self.model_type == "alexnet":
            # waveform -> linear STFT spectrogram
            x = self.spec(x)             # (F, T)
            x = (x - x.mean()) / (x.std() + 1e-8)
            x = x.unsqueeze(0)           # (1, F, T)

        else:
            # audionet / res1dnet31 / cnn14
            x = x.unsqueeze(0)           # (1, T)

        return x, y


# ============================================================
# Loader
# ============================================================
def get_loader(split_txt, data_dir, batch_size, shuffle, num_workers, model_type):
    ds = MSOSDataset(split_txt, data_dir, model_type)
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
def run_epoch(model, loader, loss_fn, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()

    total, correct, loss_sum = 0, 0, 0.0

    for x, y in loader:
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)

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
    parser.add_argument(
        "--model",
        choices=["cnn14", "res1dnet31", "audionet", "alexnet"],
        default="alexnet",
    )
    parser.add_argument("--data_dir", default="data/MSoS/preprocessed_data")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)

    device = "cuda"

    data_dir = Path(args.data_dir)
    train_txt = data_dir / "MSOS_train.txt"
    test_txt  = data_dir / "MSOS_test.txt"

    train_loader = get_loader(
        train_txt, data_dir,
        args.batch_size, True, 4, args.model
    )
    test_loader = get_loader(
        test_txt, data_dir,
        args.batch_size, False, 4, args.model
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    NUM_CLASSES = 5

    if args.model == "cnn14":
        model = Cnn14(num_classes=NUM_CLASSES)
        load_cnn14_pretrained(
            model,
            "model/pretrained/Cnn14_mAP=0.431.pth",
        )

    elif args.model == "res1dnet31":
        model = Res1dNet31(num_classes=NUM_CLASSES)
        load_res1dnet31_pretrained(
            model,
            ckpt_path="model/pretrained/Res1dNet31_mAP=0.365.pth",
        )

    elif args.model == "audionet":
        model = AudioNet(num_classes=NUM_CLASSES)

    elif args.model == "alexnet":
        model = AlexNet_Audio(num_classes=NUM_CLASSES)

    model = model.to(device)

    # --------------------------------------------------------
    # Optim
    # --------------------------------------------------------
    loss_fn = nn.CrossEntropyLoss()
    #optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    best_acc = -1.0
    best_epoch = -1

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------
    for ep in range(args.epochs):
        tr_loss, tr_acc = run_epoch(
            model, train_loader, loss_fn, optimizer
        )
        te_loss, te_acc = run_epoch(
            model, test_loader, loss_fn
        )

        print(
            f"Epoch [{ep+1:03d}] "
            f"Train : loss={tr_loss:.4f}, acc={tr_acc:.4f} | "
            f"Test : loss={te_loss:.4f}, acc={te_acc:.4f}"
        )

        if te_acc > best_acc:
            best_acc = te_acc
            best_epoch = ep

            save_dir = Path("checkpoints/msos") / args.model
            save_dir.mkdir(parents=True, exist_ok=True)

            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "best_acc": best_acc,
                    "epoch": best_epoch,
                    "model": args.model,
                    "seed": args.seed,
                    "num_classes": NUM_CLASSES,
                },
                save_dir / "best_model.pth",
                _use_new_zipfile_serialization=False,
            )

        scheduler.step()

    print(f"Best test acc: {best_acc:.4f} at epoch {best_epoch+1}")


if __name__ == "__main__":
    main()
 