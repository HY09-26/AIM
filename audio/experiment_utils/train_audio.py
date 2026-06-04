"""Unified audio model training — AudioMNIST, ESC-50, MSoS.

Run from the audio/ directory:
    python experiment_utils/train_audio.py --dataset audiomnist --model audionet
    python experiment_utils/train_audio.py --dataset esc50      --model res1dnet31 --fold 2
    python experiment_utils/train_audio.py --dataset msos       --model alexnet
"""
import os
import random
import argparse
from pathlib import Path

import numpy as np
import h5py
import torchaudio

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset

from experiment_utils.model.audionet import AudioNet
from experiment_utils.model.alexnet import AlexNet_Audio
from experiment_utils.model.cnn14 import Cnn14, load_cnn14_pretrained
from experiment_utils.model.res1dnet31 import Res1dNet31, Res1dNet31Lite, load_res1dnet31_pretrained


# ── Per-dataset defaults ──────────────────────────────────────────────────────
_DATASET_INFO = {
    'audiomnist': dict(num_classes=10, epochs=20,  lr=1e-3, batch=32),
    'esc50':      dict(num_classes=50, epochs=150, lr=1e-3, batch=16),
    'msos':       dict(num_classes=5,  epochs=50,  lr=1e-4, batch=16),
}


# ── Reproducibility ───────────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Dataset ───────────────────────────────────────────────────────────────────
class AudioDataset(Dataset):
    """HDF5 audio dataset for AudioMNIST, ESC-50, and MSoS.

    Returns:
      wave models : (1, T)
      spec models : (1, F, T)   — AlexNet; MSoS computes STFT on the fly
    """

    def __init__(self, split_txt: str, data_dir: str, dataset: str, model_name: str):
        self.data_dir = data_dir
        self.dataset  = dataset
        self.use_spec = (model_name == 'alexnet')
        # MSoS has only waveform files; AlexNet applies STFT inline
        self.msos_spec = (dataset == 'msos' and self.use_spec)

        # AudioMNIST and ESC-50 have separate waveform/spectrogram files
        need_spec_files = self.use_spec and not self.msos_spec

        self.paths: list[str] = []
        with open(split_txt) as fh:
            for line in fh:
                p = line.strip()
                if not p:
                    continue
                if dataset in ('audiomnist', 'esc50'):
                    if need_spec_files and 'waveform' in p.lower():
                        continue
                    if not need_spec_files and 'spectrogram' in p.lower():
                        continue
                self.paths.append(os.path.join(data_dir, p))

        assert self.paths, f"No files found in split: {split_txt}"

        if self.msos_spec:
            self._stft = torchaudio.transforms.Spectrogram(
                n_fft=1024, hop_length=512, power=1
            )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        with h5py.File(self.paths[idx], 'r') as f:
            raw = np.asarray(f['data'][:])
            y   = int(f['label'][0][0])

        y = torch.tensor(y, dtype=torch.long)

        if self.use_spec and not self.msos_spec:
            # Spectrogram file: squeeze leading size-1 dims to (F, T)
            x = raw
            while x.ndim > 2 and x.shape[0] == 1:
                x = x[0]
            x = torch.from_numpy(x.copy()).float()
            if self.dataset == 'audiomnist':
                mean = x.mean(dim=1, keepdim=True)   # per-frequency
                std  = x.std(dim=1, keepdim=True)
            else:
                mean, std = x.mean(), x.std()
            x = (x - mean) / (std + 1e-6)
            x = x.unsqueeze(0)  # (1, F, T)

        else:
            # Waveform file: squeeze to 1-D
            x = raw
            while x.ndim > 1 and x.shape[0] == 1:
                x = x[0]
            x = torch.tensor(x.flatten().copy(), dtype=torch.float32)

            if self.msos_spec:
                x = self._stft(x)                    # (F, T)
                x = (x - x.mean()) / (x.std() + 1e-8)
                x = x.unsqueeze(0)                   # (1, F, T)
            else:
                x = x.unsqueeze(0)                   # (1, T)

        return x, y


# ── Training loop ─────────────────────────────────────────────────────────────
def run_epoch(model, loader, loss_fn, optimizer=None, device='cuda'):
    training = optimizer is not None
    model.train() if training else model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss   = loss_fn(logits, y)
        if training:
            loss.backward()
            optimizer.step()
        loss_sum += loss.item() * x.size(0)
        correct  += (logits.argmax(1) == y).sum().item()
        total    += x.size(0)
    return loss_sum / total, correct / total


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Train audio classification model')
    parser.add_argument('--dataset',      required=True,
                        choices=['audiomnist', 'esc50', 'msos'])
    parser.add_argument('--model',        required=True,
                        choices=['audionet', 'res1dnet31', 'alexnet', 'cnn14'])
    parser.add_argument('--data_dir',     default=None,
                        help='Defaults to data/<DATASET>/preprocessed_data')
    parser.add_argument('--fold',         type=int, default=1,
                        help='ESC-50 fold (1–5); ignored for other datasets')
    parser.add_argument('--epochs',       type=int, default=None)
    parser.add_argument('--batch_size',   type=int, default=None)
    parser.add_argument('--lr',           type=float, default=None)
    parser.add_argument('--num_workers',  type=int, default=4)
    parser.add_argument('--fixed_test_n', type=int, default=500,
                        help='AudioMNIST: fixed test-subset size (reproducibility)')
    args = parser.parse_args()

    info        = _DATASET_INFO[args.dataset]
    num_classes = info['num_classes']
    epochs      = args.epochs     if args.epochs     is not None else info['epochs']
    batch_size  = args.batch_size if args.batch_size is not None else info['batch']
    lr          = args.lr         if args.lr         is not None else info['lr']

    _dir_map = {'audiomnist': 'audiomnist', 'esc50': 'ESC50', 'msos': 'MSoS'}
    if args.data_dir is None:
        args.data_dir = os.path.join('data', _dir_map[args.dataset], 'preprocessed_data')

    set_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base   = Path(args.data_dir)

    # ── Split file paths ──────────────────────────────────────────────────────
    if args.dataset == 'audiomnist':
        prefix    = 'AlexNet' if args.model == 'alexnet' else 'AudioNet'
        train_txt = base / f'{prefix}_digit_0_train.txt'
        val_txt   = base / f'{prefix}_digit_0_validate.txt'
        test_txt  = base / f'{prefix}_digit_0_test.txt'
    elif args.dataset == 'esc50':
        train_txt = base / f'ESC50_fold{args.fold}_train.txt'
        test_txt  = base / f'ESC50_fold{args.fold}_test.txt'
        val_txt   = None
    else:  # msos
        train_txt = base / 'MSOS_train.txt'
        test_txt  = base / 'MSOS_test.txt'
        val_txt   = None

    # ── Data loaders ──────────────────────────────────────────────────────────
    def make_loader(txt, shuffle):
        ds = AudioDataset(str(txt), args.data_dir, args.dataset, args.model)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=args.num_workers, pin_memory=True, drop_last=False)

    train_loader = make_loader(train_txt, shuffle=True)
    val_loader   = make_loader(val_txt, shuffle=False) if val_txt else None

    # AudioMNIST: fixed random test subset (saved for reproducibility)
    fixed_idx = None
    if args.dataset == 'audiomnist':
        full_test_ds = AudioDataset(str(test_txt), args.data_dir, args.dataset, args.model)
        N         = len(full_test_ds)
        fixed_n   = min(args.fixed_test_n, N)
        rng       = np.random.default_rng(seed=0)
        fixed_idx = rng.choice(N, size=fixed_n, replace=False)
        test_loader = DataLoader(
            Subset(full_test_ds, fixed_idx),
            batch_size=batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True, drop_last=False,
        )
    else:
        test_loader = make_loader(test_txt, shuffle=False)

    # ── Model ─────────────────────────────────────────────────────────────────
    PRETRAINED_DIR = os.path.join('experiment_utils', 'model', 'pretrained')

    if args.model == 'alexnet':
        model = AlexNet_Audio(num_classes=num_classes)
    elif args.model == 'audionet':
        model = AudioNet(num_classes=num_classes)
    elif args.model == 'cnn14':
        model = Cnn14(num_classes=num_classes)
        load_cnn14_pretrained(model, os.path.join(PRETRAINED_DIR, 'Cnn14_mAP=0.431.pth'))
    elif args.model == 'res1dnet31':
        if args.dataset == 'audiomnist':
            model = Res1dNet31Lite(num_classes=num_classes)
        else:
            model = Res1dNet31(num_classes=num_classes)
            load_res1dnet31_pretrained(
                model, ckpt_path=os.path.join(PRETRAINED_DIR, 'Res1dNet31_mAP=0.365.pth')
            )

    model = model.to(device)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    loss_fn   = nn.CrossEntropyLoss()
    weight_decay = 1e-4 if args.dataset == 'msos' else 0.0
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    # ── Save dir ──────────────────────────────────────────────────────────────
    if args.dataset == 'esc50':
        save_dir = Path('checkpoints') / 'esc50' / args.model / f'fold_{args.fold}'
    else:
        save_dir = Path('checkpoints') / args.dataset / args.model
    save_dir.mkdir(parents=True, exist_ok=True)

    if fixed_idx is not None:
        np.save(save_dir / 'fixed_test_indices.npy', fixed_idx)

    print(f'Dataset={args.dataset} | Model={args.model} | Classes={num_classes}')
    print(f'Train={len(train_loader.dataset)} | Test={len(test_loader.dataset)}'
          + (f' | Val={len(val_loader.dataset)}' if val_loader else ''))

    best_acc, best_epoch = -1.0, -1

    for ep in range(epochs):
        tr_loss, tr_acc = run_epoch(model, train_loader, loss_fn, optimizer, device)
        if val_loader:
            va_loss, va_acc = run_epoch(model, val_loader, loss_fn, None, device)
        te_loss, te_acc = run_epoch(model, test_loader, loss_fn, None, device)

        parts = [f'Epoch [{ep+1:03d}] Train: {tr_loss:.4f}/{tr_acc:.4f}']
        if val_loader:
            parts.append(f'Val: {va_loss:.4f}/{va_acc:.4f}')
        parts.append(f'Test: {te_loss:.4f}/{te_acc:.4f}')
        print(' | '.join(parts))

        if te_acc > best_acc:
            best_acc, best_epoch = te_acc, ep
            ckpt: dict = {
                'state_dict': model.state_dict(),
                'best_acc':   float(best_acc),
                'epoch':      int(best_epoch),
                'model':      args.model,
                'dataset':    args.dataset,
                'seed':       0,
                'num_classes': num_classes,
            }
            if args.dataset == 'esc50':
                ckpt['fold'] = args.fold
            if fixed_idx is not None:
                ckpt['fixed_test_n'] = int(min(args.fixed_test_n, N))
                ckpt['fixed_test_indices_file'] = str(save_dir / 'fixed_test_indices.npy')
            torch.save(ckpt, save_dir / 'best_model.pth', _use_new_zipfile_serialization=False)

        scheduler.step()

    print(f'Best test acc: {best_acc:.4f} at epoch {best_epoch + 1}')


if __name__ == '__main__':
    main()
