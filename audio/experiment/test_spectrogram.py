"""
test_spectrogram.py - MoRF/LeRF faithfulness evaluation on 2-D spectrogram
(or log-mel) inputs, unified for AudioMNIST, ESC-50, and MSoS.

Dataset-specific details
------------------------
AudioMNIST (10 classes):
    AlexNet  → AudioMNISTSpecDataset   → (N, 1, F, T) spectrogram
    CNN14    → AudioMNISTWaveDataset   → waveform → log-mel (N, 1, mel, time)

ESC-50 (50 classes, fold 1–5):
    AlexNet  → H5 spectrogram files   → (N, 1, F, T), z-normalised per sample
    CNN14    → H5 waveform files      → waveform → log-mel (N, 1, mel, time)

MSoS (5 classes):
    AlexNet  → MSOSDatasetForExpl     → (N, 1, F, T) spectrogram
    CNN14    → MSOSDatasetForExpl     → waveform → log-mel (N, 1, mel, time)
    Note: MSoS saliency maps are saved in (N, 1, time, mel) and are transposed
          here to match the (N, 1, mel, time) model input space.
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)

from experiment_utils.model.alexnet import AlexNet_Audio
from experiment_utils.model.cnn14 import Cnn14

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Constants
# ============================================================

DATASET_NUM_CLASSES: dict[str, int] = {
    "audiomnist": 10,
    "esc50":      50,
    "msos":       5,
}

# Default PGD ε per (model, dataset)
PGD_EPS_DEFAULT: dict[tuple[str, str], float] = {
    ("alexnet", "audiomnist"): 1.0,
    ("cnn14",   "audiomnist"): 1.0,
    ("alexnet", "esc50"):      0.75,
    ("cnn14",   "esc50"):      0.75,
    ("alexnet", "msos"):       0.015,
    ("cnn14",   "msos"):       0.015,
}


# ============================================================
# CNN14 log-mel wrapper  (shared across all datasets)
# ============================================================

def _call_conv_block(block, x: torch.Tensor, pool_size) -> torch.Tensor:
    """Call a Cnn14 conv-block, handling both old and new pool_type API."""
    try:
        return block(x, pool_size=pool_size, pool_type="avg")
    except TypeError:
        return block(x, pool_size=pool_size)


class CNN14SpectrogramWrapper(nn.Module):
    """
    Wraps a pretrained Cnn14 model to accept log-mel spectrograms
    (shape B×1×mel×time) instead of raw waveforms.

    The ``waveform_to_logmel`` method converts raw waveforms to the log-mel
    space used in the MoRF/LeRF masking loop, decoupling feature extraction
    from the masking experiment.
    """

    def __init__(self, cnn14: Cnn14) -> None:
        super().__init__()
        self.m = cnn14

    @torch.no_grad()
    def waveform_to_logmel(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Convert waveform tensor to log-mel spectrogram.

        Args:
            wav: Shape (B, T) or (B, 1, T).

        Returns:
            Log-mel tensor of shape (B, 1, mel, time).
        """
        if wav.dim() == 3:        # (B, 1, T) → (B, T)
            wav = wav.squeeze(1)
        elif wav.dim() == 4:      # (B, 1, 1, T) → (B, T)
            wav = wav.squeeze(1).squeeze(1)
        spec   = self.m.spectrogram_extractor(wav)
        logmel = self.m.logmel_extractor(spec)
        return logmel.permute(0, 1, 3, 2)     # → (B, 1, mel, time)

    def forward_from_logmel(self, logmel: torch.Tensor) -> torch.Tensor:
        """Run CNN14 classifier head from a log-mel tensor (B, 1, mel, time)."""
        x = logmel.permute(0, 1, 3, 2)        # → (B, 1, time, mel)
        x = x.transpose(1, 3)
        x = self.m.bn0(x)
        x = x.transpose(1, 3)
        x = _call_conv_block(self.m.conv_block1, x, (2, 2))
        x = _call_conv_block(self.m.conv_block2, x, (2, 2))
        x = _call_conv_block(self.m.conv_block3, x, (2, 2))
        x = _call_conv_block(self.m.conv_block4, x, (2, 2))
        x = _call_conv_block(self.m.conv_block5, x, (2, 2))
        x = _call_conv_block(self.m.conv_block6, x, (1, 1))
        x  = torch.mean(x, dim=3)
        x1, _ = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x  = x1 + x2
        x  = torch.relu_(self.m.fc1(x))
        return self.m.fc_out(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_from_logmel(x)


# ============================================================
# Dataset-specific input loaders
# ============================================================

def _load_audiomnist_inputs(
    model_name: str,
    split_txt: str,
    data_dir: str,
    batch_size: int,
    cnn14_model: Optional[CNN14SpectrogramWrapper],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load AudioMNIST test inputs in the correct feature space.

    AlexNet: returns (N, 1, F, T) spectrograms.
    CNN14  : returns (N, 1, mel, time) log-mel converted from waveforms.
    """
    from experiment_utils.train_audiomnist import (
        AudioMNISTSpecDataset,
        AudioMNISTWaveDataset,
        MODEL_INPUT,
        ensure_spec_batch,
        ensure_wave_batch,
    )

    if MODEL_INPUT[model_name] == "spec":
        ds     = AudioMNISTSpecDataset(split_txt, data_dir)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
        X_all, y_all = [], []
        for x, y in loader:
            X_all.append(ensure_spec_batch(x))
            y_all.append(y)
        return torch.cat(X_all).numpy(), torch.cat(y_all).numpy()

    # CNN14: waveform → log-mel
    ds     = AudioMNISTWaveDataset(split_txt, data_dir)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)
    X_all, y_all = [], []
    for x, y in loader:
        x = ensure_wave_batch(x).to(device)
        with torch.no_grad():
            logmel = cnn14_model.waveform_to_logmel(x)
        X_all.append(logmel.cpu())
        y_all.append(y)
    return torch.cat(X_all).numpy(), torch.cat(y_all).numpy()


def _load_esc50_inputs(
    model_name: str,
    split_txt: str,
    data_dir: str,
    batch_size: int,
    cnn14_model: Optional[CNN14SpectrogramWrapper],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load ESC-50 test inputs in the correct feature space.

    Split-txt lines contain either "waveform" or "spectrogram" in the path;
    the correct variant is selected per model.

    AlexNet: reads H5 spectrogram files, z-normalises per sample →
             returns (N, 1, F, T).
    CNN14  : reads H5 waveform files, converts to log-mel →
             returns (N, 1, mel, time).
    """

    def _load_h5_wave(path: str) -> tuple[np.ndarray, int]:
        with h5py.File(path, "r") as f:
            x = np.asarray(f["data"][:], np.float32).reshape(-1)   # (T,)
            y = int(f["label"][0][0])
        return x, y

    def _load_h5_spec(path: str) -> tuple[np.ndarray, int]:
        with h5py.File(path, "r") as f:
            x = np.asarray(f["data"][0, 0], np.float32)             # (F, T)
            y = int(f["label"][0][0])
        mean, std = x.mean(), x.std()
        x = (x - mean) / (std + 1e-6)
        return x, y

    with open(split_txt, "r") as f:
        lines = [l.strip() for l in f if l.strip()]

    if model_name == "alexnet":
        # Filter to spectrogram files
        xs, ys = [], []
        for rel in lines:
            if "spectrogram" not in rel:
                continue
            x, y = _load_h5_spec(os.path.join(data_dir, rel))
            xs.append(x)
            ys.append(y)
        X = np.stack(xs, axis=0)[:, None, :, :]       # (N, 1, F, T)
        return X.astype(np.float32), np.array(ys, np.int64)

    # CNN14: waveform files → log-mel
    xs_wave, ys = [], []
    for rel in lines:
        if "waveform" not in rel:
            continue
        x, y = _load_h5_wave(os.path.join(data_dir, rel))
        xs_wave.append(x)
        ys.append(y)

    wav_tensor = torch.tensor(np.stack(xs_wave, 0), dtype=torch.float32)
    X_all = []
    for i in range(0, len(wav_tensor), batch_size):
        wav_b = wav_tensor[i:i + batch_size].to(device)
        with torch.no_grad():
            logmel = cnn14_model.waveform_to_logmel(wav_b)
        X_all.append(logmel.cpu())
    return torch.cat(X_all).numpy(), np.array(ys, np.int64)


def _load_msos_inputs(
    model_name: str,
    split_txt: str,
    data_dir: str,
    batch_size: int,
    cnn14_model: Optional[CNN14SpectrogramWrapper],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load MSoS test inputs in the correct feature space.

    Uses ``MSOSDatasetForExpl`` for both models.

    AlexNet: dataset yields spectrogram directly →
             returns (N, 1, F, T).
    CNN14  : dataset yields waveform → convert to log-mel →
             returns (N, 1, mel, time).
    """
    from experiment_utils.expl_gen_msos_plus import MSOSDatasetForExpl

    ds     = MSOSDatasetForExpl(split_txt, data_dir, model_name)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)

    if model_name == "alexnet":
        X_all, y_all = [], []
        for x, y in loader:
            # Ensure shape (B, 1, F, T)
            if x.dim() == 3:
                x = x.unsqueeze(1)
            X_all.append(x)
            y_all.append(y)
        return torch.cat(X_all).numpy(), torch.cat(y_all).numpy()

    # CNN14: waveform → log-mel
    X_all, y_all = [], []
    for wav, y in loader:
        wav = wav.to(device)
        if wav.dim() == 2:      # (B, T)
            pass
        elif wav.dim() == 3:    # (B, 1, T)
            wav = wav.squeeze(1)
        with torch.no_grad():
            logmel = cnn14_model.waveform_to_logmel(wav)
        X_all.append(logmel.cpu())
        y_all.append(y)
    return torch.cat(X_all).numpy(), torch.cat(y_all).numpy()


def load_inputs(
    dataset: str,
    model_name: str,
    split_txt: str,
    data_dir: str,
    batch_size: int,
    cnn14_wrapper: Optional[CNN14SpectrogramWrapper],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Dispatcher: load (X_2d, y) in model input space for the given dataset.

    Args:
        dataset:      One of "audiomnist", "esc50", "msos".
        model_name:   "alexnet" or "cnn14".
        split_txt:    Path to the test split text file.
        data_dir:     Root directory of preprocessed data.
        batch_size:   DataLoader batch size.
        cnn14_wrapper: CNN14SpectrogramWrapper instance (needed only for cnn14).

    Returns:
        X_2d: (N, 1, H, W) float32 array in model input space.
        y:    (N,)  int64 label array.
    """
    if dataset == "audiomnist":
        return _load_audiomnist_inputs(model_name, split_txt, data_dir,
                                       batch_size, cnn14_wrapper)
    if dataset == "esc50":
        return _load_esc50_inputs(model_name, split_txt, data_dir,
                                  batch_size, cnn14_wrapper)
    if dataset == "msos":
        return _load_msos_inputs(model_name, split_txt, data_dir,
                                 batch_size, cnn14_wrapper)
    raise ValueError(f"Unknown dataset: {dataset!r}")


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def eval_epoch(model: nn.Module, loader: DataLoader) -> float:
    """Compute classification accuracy over a DataLoader."""
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total   += y.size(0)
    return correct / max(1, total)


# ============================================================
# PGD attack (2-D feature space)
# ============================================================

def pgd_attack_2d(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    eps: float,
    steps: int = 20,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Run PGD adversarial attack in the 2-D feature space (spectrogram or log-mel).

    Args:
        model:      Classifier accepting (B, 1, H, W) inputs.
        X:          Input array, shape (N, 1, H, W), float32.
        y:          Ground-truth labels, shape (N,).
        eps:        L-inf perturbation budget.
        steps:      Number of PGD iterations.
        batch_size: Batch size for the attack loop.

    Returns:
        Adversarial array of shape (N, 1, H, W).
    """
    model.eval()
    X_adv = X.copy().astype(np.float32)
    step_size = eps / max(1, steps)
    N = X.shape[0]

    for i in range(0, N, batch_size):
        xb0 = torch.tensor(X[i:i + batch_size], device=device)
        xb  = torch.tensor(X_adv[i:i + batch_size], device=device,
                           requires_grad=True)
        yb  = torch.tensor(y[i:i + batch_size], device=device)

        for _ in range(steps):
            loss = nn.CrossEntropyLoss()(model(xb), yb)
            loss.backward()
            xb = xb + step_size * xb.grad.sign()
            xb = torch.clamp(xb, xb0 - eps, xb0 + eps).detach().requires_grad_(True)

        X_adv[i:i + batch_size] = xb.detach().cpu().numpy()

    return X_adv


# ============================================================
# ROAD masking (2-D feature space)
# ============================================================

@torch.no_grad()
def road_2d(X: np.ndarray, noise_std: float = 0.1) -> np.ndarray:
    """
    Generate in-distribution ROAD replacements for 2-D feature inputs.

    Each pixel is replaced by a weighted average of its 8 neighbours
    (wd=1/6 for cardinal, wi=1/12 for diagonal), plus Gaussian noise.

    Args:
        X:          Input array, shape (N, 1, H, W), float32.
        noise_std:  Standard deviation of additive Gaussian noise.

    Returns:
        ROAD-imputed array of shape (N, 1, H, W).
    """
    X_t = torch.from_numpy(X.astype(np.float32))
    _, _, H, W = X_t.shape
    wd, wi = 1 / 6, 1 / 12
    Xp = F.pad(X_t, (1, 1, 1, 1), mode="reflect")

    up    = Xp[:, :, 0:H,     1:W + 1]
    down  = Xp[:, :, 2:H + 2, 1:W + 1]
    left  = Xp[:, :, 1:H + 1, 0:W]
    right = Xp[:, :, 1:H + 1, 2:W + 2]
    ul    = Xp[:, :, 0:H,     0:W]
    ur    = Xp[:, :, 0:H,     2:W + 2]
    dl    = Xp[:, :, 2:H + 2, 0:W]
    dr    = Xp[:, :, 2:H + 2, 2:W + 2]

    interp = wd * (up + down + left + right) + wi * (ul + ur + dl + dr)
    noise  = noise_std * torch.randn_like(interp)
    return (interp + noise).numpy().astype(np.float32)


# ============================================================
# MoRF / LeRF runner
# ============================================================

def run_morf_lerf(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    saliency: np.ndarray,
    n_steps: int,
    batch_size: int,
    mode: str,
    X_adv: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Rank-based MoRF or LeRF faithfulness evaluation on 2-D inputs.

    Args:
        model:      Classifier accepting (B, 1, H, W) tensors.
        X:          Input array, shape (N, 1, H, W).
        y:          Labels, shape (N,).
        saliency:   Attribution scores, shape (N, 1, H, W) or (N, H, W).
        n_steps:    Number of masking steps.
        batch_size: Inference batch size.
        mode:       "morf" (mask highest first) or "lerf" (mask lowest first).
        X_adv:      Optional adversarial replacement, shape (N, 1, H, W).

    Returns:
        acc_curve: Accuracy at each step, shape (n_steps + 1,).
    """
    if saliency.ndim == 3:
        saliency = saliency[:, None, :, :]

    N, _, H, W = X.shape
    HW       = H * W
    x_flat   = X.reshape(N, 1, HW).copy()
    s_flat   = saliency.reshape(N, HW)
    order    = (np.argsort(-s_flat, axis=1) if mode == "morf"
                else np.argsort(s_flat, axis=1))
    step_sz  = max(1, HW // n_steps)
    zero_val = float(X.min())
    adv_flat = X_adv.reshape(N, 1, HW) if X_adv is not None else None

    acc_curve = []
    for k in range(n_steps + 1):
        if k > 0:
            for i in range(N):
                idx = order[i, (k - 1) * step_sz : k * step_sz]
                if adv_flat is None:
                    x_flat[i, :, idx] = zero_val
                else:
                    x_flat[i, :, idx] = adv_flat[i, :, idx]

        loader = DataLoader(
            TensorDataset(
                torch.tensor(x_flat.reshape(N, 1, H, W), dtype=torch.float32),
                torch.tensor(y, dtype=torch.long),
            ),
            batch_size=batch_size,
            shuffle=False,
        )
        acc = eval_epoch(model, loader)
        acc_curve.append(acc)
        logger.info(
            "[%s-2D] step %02d/%d | masked=%5.1f%% | acc=%.4f",
            mode.upper(), k, n_steps, 100 * k / n_steps, acc,
        )

    return np.array(acc_curve, dtype=np.float32)


# ============================================================
# Default path helpers
# ============================================================

def _default_data_dir(dataset: str) -> str:
    base = Path(PROJECT_ROOT) / "experiment_utils" / "data"
    return str({
        "audiomnist": base / "audiomnist" / "preprocessed_data",
        "esc50":      base / "ESC50"      / "preprocessed_data",
        "msos":       base / "MSoS"       / "preprocessed_data",
    }[dataset])


def _default_split_txt(dataset: str, data_dir: str, fold: int) -> str:
    p = Path(data_dir)
    return str({
        "audiomnist": p / "AudioNet_digit_0_test.txt",
        "esc50":      p / f"ESC50_fold{fold}_test.txt",
        "msos":       p / "MSOS_test.txt",
    }[dataset])


def _default_ckpt(dataset: str, model_name: str, fold: int) -> str:
    base = Path(PROJECT_ROOT) / "experiment_utils" / "checkpoints"
    ckpt_map = {
        "audiomnist": base / "audiomnist" / model_name / "best_model.pth",
        "esc50":      base / "esc50"      / model_name / f"fold_{fold}" / "best_model.pth",
        "msos":       base / "msos"       / model_name / "best_model.pth",
    }
    return str(ckpt_map[dataset])


def _saliency_path(dataset: str, model_name: str, fold: int, method: str) -> Path:
    expl_dir = {
        "audiomnist": Path(PROJECT_ROOT) / "expl_audiomnist" / model_name,
        "esc50":      Path(PROJECT_ROOT) / "expl_esc50"      / model_name / f"fold_{fold}",
        "msos":       Path(PROJECT_ROOT) / "expl_msos"        / model_name,
    }[dataset]
    return expl_dir / f"{method}.npy"


def _cache_dir(out_root: str, dataset: str, model_name: str,
               fold: int, mask_type: str) -> Path:
    base = Path(out_root) / "_cache" / dataset / model_name
    if dataset == "esc50":
        base = base / f"fold_{fold}"
    return base / mask_type


def _output_dir(out_root: str, dataset: str, model_name: str,
                fold: int, mask_type: str) -> Path:
    base = Path(out_root) / dataset / model_name
    if dataset == "esc50":
        base = base / f"fold_{fold}"
    return base / mask_type


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MoRF/LeRF faithfulness evaluation on 2-D spectrogram inputs."
    )
    parser.add_argument("--dataset",     choices=["audiomnist", "esc50", "msos"],
                        required=True)
    parser.add_argument("--model",       choices=["alexnet", "cnn14"], default="cnn14")
    parser.add_argument("--fold",        type=int, default=1,
                        help="ESC-50 fold index (1–5). Ignored for other datasets.")
    parser.add_argument("--data_dir",    default=None)
    parser.add_argument("--split_txt",   default=None)
    parser.add_argument("--ckpt",        default=None)
    parser.add_argument("--expl_method", default="gradient_abs")
    parser.add_argument("--mask_type",   choices=["zero", "pgd", "road"], default="zero")
    parser.add_argument("--eps",         type=float, default=None,
                        help="PGD ε. Defaults to dataset/model-specific preset.")
    parser.add_argument("--pgd_steps",   type=int, default=20)
    parser.add_argument("--n_steps",     type=int, default=20)
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--road_noise",  type=float, default=0.1)
    parser.add_argument("--out_root",    default=None,
                        help="Root directory for saving results. "
                             "Defaults to <project_root>/morf_lerf_spec.")
    args = parser.parse_args()

    # ── Fill defaults ────────────────────────────────────────────────────────
    if args.data_dir is None:
        args.data_dir = _default_data_dir(args.dataset)
    if args.split_txt is None:
        args.split_txt = _default_split_txt(args.dataset, args.data_dir, args.fold)
    if args.ckpt is None:
        args.ckpt = _default_ckpt(args.dataset, args.model, args.fold)
    if args.eps is None:
        args.eps = PGD_EPS_DEFAULT.get((args.model, args.dataset), 0.5)
    if args.out_root is None:
        args.out_root = str(Path(PROJECT_ROOT) / "morf_lerf_spec")

    num_classes = DATASET_NUM_CLASSES[args.dataset]
    logger.info("Device: %s | Dataset: %s | Model: %s | Fold: %s",
                device, args.dataset, args.model,
                args.fold if args.dataset == "esc50" else "N/A")

    # ── Build model ──────────────────────────────────────────────────────────
    if args.model == "cnn14":
        base_model = Cnn14(num_classes=num_classes)
        logger.info("Loading checkpoint: %s", args.ckpt)
        ckpt  = torch.load(args.ckpt, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        base_model.load_state_dict(state)
        base_model = base_model.to(device).eval()
        model = CNN14SpectrogramWrapper(base_model).to(device)
    else:
        model = AlexNet_Audio(num_classes=num_classes)
        logger.info("Loading checkpoint: %s", args.ckpt)
        ckpt  = torch.load(args.ckpt, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        model.load_state_dict(state)
        model = model.to(device).eval()

    # ── Load inputs (X_2d, y) ───────────────────────────────────────────────
    logger.info("Loading test inputs from: %s", args.data_dir)
    X_2d, y = load_inputs(
        dataset=args.dataset,
        model_name=args.model,
        split_txt=args.split_txt,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        cnn14_wrapper=model if args.model == "cnn14" else None,
    )
    logger.info("Input shape: %s  Labels: %s", X_2d.shape, y.shape)

    # ── Sanity check accuracy ────────────────────────────────────────────────
    sanity_loader = DataLoader(
        TensorDataset(torch.tensor(X_2d), torch.tensor(y, dtype=torch.long)),
        batch_size=args.batch_size, shuffle=False,
    )
    logger.info("Baseline accuracy (unmasked): %.4f", eval_epoch(model, sanity_loader))

    # ── Load saliency ────────────────────────────────────────────────────────
    sal_path = _saliency_path(args.dataset, args.model, args.fold, args.expl_method)
    logger.info("Loading saliency: %s", sal_path)
    saliency = np.load(sal_path)

    # MSoS CNN14 saliency is saved in (N, 1, time, mel); transpose to (N, 1, mel, time)
    if args.dataset == "msos" and args.model == "cnn14":
        saliency = saliency.transpose(0, 1, 3, 2)
        logger.info("MSoS+CNN14: transposed saliency to %s", saliency.shape)
    elif saliency.ndim == 3:
        saliency = saliency[:, None, :, :]

    logger.info("Saliency shape: %s", saliency.shape)

    # ── Masking replacement (cached) ─────────────────────────────────────────
    X_adv = None
    cache_dir = _cache_dir(args.out_root, args.dataset, args.model,
                           args.fold, args.mask_type)

    if args.mask_type == "pgd":
        cache_dir.mkdir(parents=True, exist_ok=True)
        pgd_path = cache_dir / f"pgd_eps{args.eps}_steps{args.pgd_steps}.npy"

        if pgd_path.exists():
            logger.info("Loading cached PGD from %s", pgd_path)
            X_adv = np.load(pgd_path)
        else:
            logger.info("Running PGD (eps=%.4f, steps=%d)...", args.eps, args.pgd_steps)
            X_adv = pgd_attack_2d(model, X_2d, y,
                                  eps=args.eps, steps=args.pgd_steps,
                                  batch_size=args.batch_size)
            np.save(pgd_path, X_adv)
            logger.info("Saved PGD cache to %s", pgd_path)

    elif args.mask_type == "road":
        cache_dir.mkdir(parents=True, exist_ok=True)
        road_path = cache_dir / f"road_noise{args.road_noise:.3f}.npy"

        if road_path.exists():
            logger.info("Loading cached ROAD from %s", road_path)
            X_adv = np.load(road_path)
        else:
            logger.info("Running ROAD (noise_std=%.3f)...", args.road_noise)
            X_adv = road_2d(X_2d, noise_std=args.road_noise)
            np.save(road_path, X_adv)
            logger.info("Saved ROAD cache to %s", road_path)

    # ── MoRF and LeRF ────────────────────────────────────────────────────────
    shared = dict(model=model, X=X_2d, y=y, saliency=saliency,
                  n_steps=args.n_steps, batch_size=args.batch_size, X_adv=X_adv)
    morf = run_morf_lerf(**shared, mode="morf")
    lerf = run_morf_lerf(**shared, mode="lerf")

    # ── Save results ─────────────────────────────────────────────────────────
    out_dir = _output_dir(args.out_root, args.dataset, args.model,
                          args.fold, args.mask_type)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / f"{args.expl_method}_morf.pkl", "wb") as f:
        pickle.dump(morf, f)
    with open(out_dir / f"{args.expl_method}_lerf.pkl", "wb") as f:
        pickle.dump(lerf, f)
    logger.info("Results saved to: %s", out_dir)


if __name__ == "__main__":
    main()
