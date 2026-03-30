"""utils.py - Shared utilities for AIM audio experiments.

This module consolidates all shared code used by the experiment scripts:
  - test_waveform.py (rank-based MoRF/LeRF on 1-D waveforms)
  - test_waveform_interval.py (interval-based MoRF/LeRF on 1-D waveforms)
  - test_spectrogram.py (MoRF/LeRF on 2-D spectrogram / log-mel inputs)

Contents
--------
Constants
  DATASET_NUM_CLASSES    : number of output classes per audio dataset
  PGD_EPS_DEFAULT        : default L-inf epsilon per (model, dataset) pair
  device                 : torch.device (CUDA when available)

Waveform loaders
  load_test_set()        : dispatcher → (X_wave, y) arrays

Model helpers
  adapt_wave_input()     : reshape waveform for AudioNet vs Res1dNet31
  eval_epoch()           : classification accuracy over a DataLoader
                           (model_name=None → no reshape, used for 2-D inputs)

CNN14 spectrogram wrapper
  CNN14SpectrogramWrapper: wraps Cnn14 to accept log-mel (B,1,mel,time) inputs

PGD adversarial attack
  pgd_attack_waveform()  : L-inf PGD on 1-D waveform inputs
  pgd_attack_2d()        : L-inf PGD on 2-D spectrogram / log-mel inputs

ROAD in-distribution masking – 1-D waveforms
  road_generate_segment(): build a smooth replacement segment
  road_apply_points_only(): replace individual points (rank-based protocol)

ROAD in-distribution masking – 2-D feature maps
  road_2d()              : neighbourhood-average + noise replacement

Default path resolvers
  default_data_dir()     : preprocessed data directory
  default_split_txt()    : test split text file
  default_ckpt()         : model checkpoint path
  default_expl_dir()     : saliency/explanation output directory
  saliency_path()        : full path to a specific attribution .npy file
  cache_dir()            : per-(dataset, model, fold, mask_type) cache dir
  output_dir()           : per-(dataset, model, fold, mask_type) result dir

Helper / internal
  _linear_fill_three_points(): piecewise-linear interpolation (3 anchors)
  _mfbb_noise_fast()     : smooth low-amplitude noise via double cumsum
  get_default_eps()      : look up PGD_EPS_DEFAULT
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)

# ============================================================
# Project root (parent of this package's directory)
# ============================================================

_UTILS_DIR   = os.path.dirname(os.path.abspath(__file__))   # .../experiment_utils/
PROJECT_ROOT = os.path.dirname(_UTILS_DIR)                   # .../audio/


# ============================================================
# Constants
# ============================================================

#: Number of output classes for each supported audio dataset.
DATASET_NUM_CLASSES: dict[str, int] = {
    "audiomnist": 10,
    "esc50":      50,
    "msos":       5,
}

#: Default PGD L-inf epsilon per (model, dataset).
#: Covers both waveform models (audionet, res1dnet31) and
#: spectrogram models (alexnet, cnn14).
PGD_EPS_DEFAULT: dict[tuple[str, str], float] = {
    # 1-D waveform models
    ("audionet",   "audiomnist"): 0.045,
    ("audionet",   "esc50"):      0.07,
    ("audionet",   "msos"):       0.0025,
    ("res1dnet31", "audiomnist"): 0.065,
    ("res1dnet31", "esc50"):      0.02,
    ("res1dnet31", "msos"):       0.0015,
    # 2-D spectrogram models
    ("alexnet",    "audiomnist"): 1.0,
    ("cnn14",      "audiomnist"): 1.0,
    ("alexnet",    "esc50"):      0.75,
    ("cnn14",      "esc50"):      0.75,
    ("alexnet",    "msos"):       0.015,
    ("cnn14",      "msos"):       0.015,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Waveform loaders
# ============================================================

def _load_h5_waveform_audiomnist(path: str) -> tuple[np.ndarray, int]:
    """Load a single AudioMNIST waveform sample from an HDF5 file."""
    with h5py.File(path, "r") as f:
        x = None
        for k in ("data", "waveform", "x"):
            if k in f:
                x = f[k][:]
                break
        if x is None:
            raise KeyError(f"No waveform key in {path}. Keys: {list(f.keys())}")

        lab = None
        for k in ("label", "y"):
            if k in f:
                lab = np.asarray(f[k][:])
                break
        if lab is None:
            raise KeyError(f"No label key in {path}. Keys: {list(f.keys())}")

    y = int(lab[0][0]) if lab.ndim >= 2 else int(lab[0])
    return np.asarray(x, np.float32).reshape(-1), y


def _load_h5_waveform_generic(path: str) -> tuple[np.ndarray, int]:
    """Load a single waveform sample from an HDF5 file (ESC-50 / MSoS format)."""
    with h5py.File(path, "r") as f:
        x = np.asarray(f["data"][:], np.float32).reshape(-1)
        y = int(f["label"][0][0])
    return x, y


def load_test_set_audiomnist(
    split_txt: str,
    data_dir: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Load AudioMNIST waveform test set from a split file."""
    xs, ys = [], []
    with open(split_txt) as f:
        for line in f:
            p = line.strip()
            if not p or p.startswith("#"):
                continue
            full = os.path.join(data_dir, p)
            if not os.path.exists(full):
                logger.warning("Missing file: %s", full)
                continue
            x, y = _load_h5_waveform_audiomnist(full)
            xs.append(x)
            ys.append(y)
    if not xs:
        raise ValueError(f"No files loaded from split_txt={split_txt}")
    return np.stack(xs).astype(np.float32), np.array(ys, np.int64)


def load_test_set_esc50(
    split_txt: str,
    data_dir: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Load ESC-50 waveform test set from a split file."""
    xs, ys = [], []
    with open(split_txt) as f:
        for line in f:
            p = line.strip()
            if not p or "waveform" not in p:
                continue
            x, y = _load_h5_waveform_generic(os.path.join(data_dir, p))
            xs.append(x)
            ys.append(y)
    return np.stack(xs).astype(np.float32), np.array(ys, np.int64)


def load_test_set_msos(
    split_txt: str,
    data_dir: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Load MSoS waveform test set from a split file."""
    xs, ys = [], []
    with open(split_txt) as f:
        for line in f:
            p = line.strip()
            if not p or "waveform" not in p:
                continue
            x, y = _load_h5_waveform_generic(os.path.join(data_dir, p))
            xs.append(x)
            ys.append(y)
    return np.stack(xs).astype(np.float32), np.array(ys, np.int64)


def load_test_set(
    dataset: str,
    split_txt: str,
    data_dir: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Dispatcher: load (X_wave, y) arrays for the given dataset."""
    loaders = {
        "audiomnist": load_test_set_audiomnist,
        "esc50":      load_test_set_esc50,
        "msos":       load_test_set_msos,
    }
    if dataset not in loaders:
        raise ValueError(
            f"Unknown dataset: {dataset!r}. Expected one of {list(loaders)}"
        )
    return loaders[dataset](split_txt, data_dir)


# ============================================================
# Model helpers
# ============================================================

def adapt_wave_input(x: torch.Tensor, model_name: str) -> torch.Tensor:
    """
    Reshape waveform tensor to match model input format.

    AudioNet  expects (B, 1, T).
    Res1dNet31 expects (B, T).

    Args:
        x:          Waveform tensor of shape (B, T) or (B, 1, T).
        model_name: One of "audionet", "res1dnet31*".

    Returns:
        Reshaped tensor.
    """
    if model_name == "audionet" and x.ndim == 2:
        x = x.unsqueeze(1)
    return x


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    model_name: Optional[str] = None,
) -> float:
    """
    Compute classification accuracy of model over a DataLoader.

    Args:
        model:      Trained classifier.
        loader:     DataLoader of (x, y) batches.
        model_name: When set, waveform inputs are reshaped via
                    ``adapt_wave_input`` before being fed to the model.
                    Pass ``None`` for 2-D (spectrogram / image) inputs.

    Returns:
        Top-1 accuracy in [0, 1].
    """
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if model_name is not None:
            x = adapt_wave_input(x, model_name)
        correct += (model(x).argmax(1) == y).sum().item()
        total   += y.size(0)
    return correct / max(1, total)


# ============================================================
# CNN14 log-mel wrapper
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

    Usage::

        base = Cnn14(num_classes=50)
        base.load_state_dict(torch.load("checkpoint.pth")["state_dict"])
        model = CNN14SpectrogramWrapper(base).to(device)

        # Convert waveform batch to log-mel
        logmel = model.waveform_to_logmel(wav_batch)   # (B, 1, mel, time)

        # Run classifier
        logits = model(logmel)
    """

    def __init__(self, cnn14: nn.Module) -> None:
        super().__init__()
        self.m = cnn14

    @torch.no_grad()
    def waveform_to_logmel(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Convert waveform tensor to log-mel spectrogram.

        Args:
            wav: Shape (B, T), (B, 1, T), or (B, 1, 1, T).

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
# PGD adversarial attack
# ============================================================

def pgd_attack_waveform(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    model_name: str,
    eps: float = 0.03,
    steps: int = 20,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Run PGD adversarial attack on 1-D waveform inputs.

    Args:
        model:      Trained classifier.
        X:          Waveforms of shape (N, T).
        y:          Ground-truth labels of shape (N,).
        model_name: One of "res1dnet31", "audionet" (used for input reshape).
        eps:        L-inf perturbation budget.
        steps:      Number of PGD iterations.
        batch_size: Batch size for the attack loop.

    Returns:
        Adversarial waveforms of shape (N, T), float32.
    """
    model.eval()
    X_adv     = X.copy()
    step_size = eps / max(1, steps)
    loss_fn   = nn.CrossEntropyLoss()

    for i in range(0, len(X), batch_size):
        xb = torch.tensor(X_adv[i:i + batch_size], device=device,
                          requires_grad=True)
        yb = torch.tensor(y[i:i + batch_size], device=device)
        x0 = torch.tensor(X[i:i + batch_size], device=device)

        for _ in range(steps):
            out  = model(adapt_wave_input(xb, model_name))
            loss = loss_fn(out, yb)
            loss.backward()
            xb = xb + step_size * xb.grad.sign()
            xb = torch.clamp(xb, x0 - eps, x0 + eps).detach().requires_grad_(True)

        X_adv[i:i + batch_size] = xb.detach().cpu().numpy()

    return X_adv.astype(np.float32)


def pgd_attack_2d(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    eps: float,
    steps: int = 20,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Run PGD adversarial attack in a 2-D feature space (spectrogram or log-mel).

    Args:
        model:      Classifier accepting (B, 1, H, W) inputs.
        X:          Input array of shape (N, 1, H, W), float32.
        y:          Ground-truth labels of shape (N,).
        eps:        L-inf perturbation budget.
        steps:      Number of PGD iterations.
        batch_size: Batch size for the attack loop.

    Returns:
        Adversarial array of shape (N, 1, H, W), float32.
    """
    model.eval()
    X_adv     = X.copy().astype(np.float32)
    step_size = eps / max(1, steps)
    N         = X.shape[0]

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


def get_default_eps(
    model_name: str,
    dataset: str,
    default: Optional[float] = None,
) -> float:
    """
    Return the default PGD epsilon for a (model, dataset) pair.

    Args:
        model_name: Model identifier string (e.g. "audionet", "cnn14").
        dataset:    Dataset identifier string (e.g. "audiomnist", "esc50").
        default:    Fallback value if the pair is not in PGD_EPS_DEFAULT.
                    If None (the default) a KeyError is raised when missing.

    Returns:
        PGD epsilon float.

    Raises:
        KeyError: When the pair is not in PGD_EPS_DEFAULT and default is None.
    """
    key = (model_name, dataset)
    if key in PGD_EPS_DEFAULT:
        return PGD_EPS_DEFAULT[key]
    if default is not None:
        return default
    raise KeyError(
        f"No default PGD eps for model={model_name!r}, dataset={dataset!r}. "
        f"Known keys: {list(PGD_EPS_DEFAULT)}"
    )


# ============================================================
# ROAD: in-distribution masking helpers – 1-D waveforms
# ============================================================

def _linear_fill_three_points(
    x0: float,
    xmid: float,
    x1: float,
    n: int,
) -> np.ndarray:
    """
    Piecewise-linear interpolation through three anchor points over n samples.

    The first anchor is at index 0, the middle anchor at index (n-1)//2,
    and the last anchor at index n-1.
    """
    if n <= 1:
        return np.array([x0], dtype=np.float32)
    mid = (n - 1) // 2
    out = np.empty(n, dtype=np.float32)
    t_left  = np.linspace(0.0, 1.0, mid + 1, dtype=np.float32)
    out[: mid + 1] = (1 - t_left) * x0 + t_left * xmid
    t_right = np.linspace(0.0, 1.0, n - mid, dtype=np.float32)
    out[mid:] = (1 - t_right) * xmid + t_right * x1
    return out


def _mfbb_noise_fast(
    n: int,
    original_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate smooth, low-amplitude noise via double cumsum (fast MFBB approximation).

    The noise has the same order of smoothness as fractional Brownian motion
    (Hurst exponent ≈ 1.5) but is generated in O(n) time.
    """
    if n <= 1:
        return np.zeros(n, dtype=np.float32)
    w = rng.standard_normal(n).astype(np.float32)
    s = np.cumsum(np.cumsum(w))
    s -= s.mean()
    sd = float(s.std() + 1e-8)
    return (s / sd * original_std * 0.05).astype(np.float32)


def road_generate_segment(
    x_full: np.ndarray,
    t0: int,
    t1: int,
    rng: np.random.Generator,
) -> Optional[np.ndarray]:
    """
    Generate an in-distribution replacement segment for x_full[t0:t1].

    The replacement is a piecewise-linear interpolation through three anchor
    points (start, mid, end of the segment), perturbed by smooth MFBB-style
    noise, then rescaled to match the original segment mean and std.

    Args:
        x_full: Full waveform, shape (T,).
        t0:     Start index of the segment (inclusive).
        t1:     End index of the segment (exclusive).
        rng:    NumPy random generator for reproducibility.

    Returns:
        Replacement segment of length (t1 - t0), or None if too short (< 4).
    """
    seg_len = t1 - t0
    if seg_len < 4:
        return None

    orig = x_full[t0:t1].astype(np.float32, copy=False)
    mu_o = float(orig.mean())
    sd_o = float(orig.std() + 1e-8)
    mid  = (t0 + t1) // 2

    base = _linear_fill_three_points(
        float(x_full[t0]), float(x_full[mid]), float(x_full[t1 - 1]), seg_len
    )
    rep = base + _mfbb_noise_fast(seg_len, sd_o, rng)

    if not np.all(np.isfinite(rep)):
        return None

    mu_r, sd_r = float(rep.mean()), float(rep.std() + 1e-8)
    rep = (rep - mu_r) / sd_r * sd_o + mu_o
    rep = np.clip(rep, mu_o - 8.0 * sd_o, mu_o + 8.0 * sd_o).astype(np.float32)
    return rep


def road_apply_points_only(
    x_work: np.ndarray,
    x_orig: np.ndarray,
    idx_mask: np.ndarray,
    seg_cache: dict,
    rng: np.random.Generator,
) -> None:
    """
    Replace selected waveform points with values from an in-distribution segment
    (rank-based masking protocol).

    The replacement segment spans [min(idx_mask), max(idx_mask)+1]; only the
    selected indices are written. Results are cached by (t0, t1) to avoid
    redundant computation across successive masking steps.

    Args:
        x_work:    Waveform to modify in-place, shape (T,).
        x_orig:    Original reference waveform, shape (T,).
        idx_mask:  Indices of the points to replace.
        seg_cache: Persistent dict keyed by (t0, t1) → replacement segment.
        rng:       NumPy random generator.
    """
    if idx_mask.size == 0:
        return
    t0, t1 = int(idx_mask.min()), int(idx_mask.max()) + 1
    if t1 - t0 < 4:
        return

    key = (t0, t1)
    if key not in seg_cache:
        seg_cache[key] = road_generate_segment(x_orig, t0, t1, rng)

    rep = seg_cache[key]
    pos = (idx_mask - t0).astype(np.int32)
    if rep is None:
        mid  = (t0 + t1) // 2
        base = _linear_fill_three_points(
            float(x_orig[t0]), float(x_orig[mid]), float(x_orig[t1 - 1]), t1 - t0
        )
        x_work[idx_mask] = base[pos]
    else:
        x_work[idx_mask] = rep[pos]


# ============================================================
# ROAD: in-distribution masking – 2-D feature maps
# ============================================================

@torch.no_grad()
def road_2d(X: np.ndarray, noise_std: float = 0.1) -> np.ndarray:
    """
    Generate in-distribution ROAD replacements for 2-D feature inputs.

    Each pixel is replaced by a weighted average of its 8 neighbours
    (cardinal weight = 1/6, diagonal weight = 1/12), plus i.i.d. Gaussian noise.

    Args:
        X:         Input array of shape (N, 1, H, W), float32.
        noise_std: Standard deviation of additive Gaussian noise.

    Returns:
        ROAD-imputed array of shape (N, 1, H, W), float32.
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
# Default path resolvers
# ============================================================

def default_data_dir(
    dataset: str,
    project_root: Optional[str] = None,
) -> str:
    """
    Return the default preprocessed data directory for a dataset.

    Args:
        dataset:      One of "audiomnist", "esc50", "msos".
        project_root: Project root directory. Defaults to the module-level
                      PROJECT_ROOT (parent of this package).

    Returns:
        Absolute path string.
    """
    root = Path(project_root or PROJECT_ROOT)
    paths = {
        "audiomnist": root / "experiment_utils" / "data" / "audiomnist" / "preprocessed_data",
        "esc50":      root / "experiment_utils" / "data" / "ESC50"      / "preprocessed_data",
        "msos":       root / "experiment_utils" / "data" / "MSoS"       / "preprocessed_data",
    }
    if dataset not in paths:
        raise ValueError(f"Unknown dataset: {dataset!r}. Expected one of {list(paths)}")
    return str(paths[dataset])


def default_split_txt(
    dataset: str,
    data_dir: str,
    fold: int = 1,
) -> str:
    """
    Return the default test split text file for a dataset.

    Args:
        dataset:  One of "audiomnist", "esc50", "msos".
        data_dir: Preprocessed data directory (output of ``default_data_dir``).
        fold:     ESC-50 fold index (1–5). Ignored for other datasets.

    Returns:
        Absolute path string.
    """
    filenames = {
        "audiomnist": "AudioNet_digit_0_test.txt",
        "esc50":      f"ESC50_fold{fold}_test.txt",
        "msos":       "MSOS_test.txt",
    }
    if dataset not in filenames:
        raise ValueError(f"Unknown dataset: {dataset!r}")
    return str(Path(data_dir) / filenames[dataset])


def default_ckpt(
    dataset: str,
    model_name: str,
    fold: int = 1,
    project_root: Optional[str] = None,
) -> str:
    """
    Return the default model checkpoint path.

    Args:
        dataset:      One of "audiomnist", "esc50", "msos".
        model_name:   Model identifier (e.g. "audionet", "cnn14").
        fold:         ESC-50 fold index (1–5). Ignored for other datasets.
        project_root: Project root directory. Defaults to module-level PROJECT_ROOT.

    Returns:
        Absolute path string.
    """
    root     = Path(project_root or PROJECT_ROOT)
    ckpt_dir = root / "experiment_utils" / "checkpoints"
    paths = {
        "audiomnist": ckpt_dir / "audiomnist" / model_name / "best_model.pth",
        "esc50":      ckpt_dir / "esc50"      / model_name / f"fold_{fold}" / "best_model.pth",
        "msos":       ckpt_dir / "msos"       / model_name / "best_model.pth",
    }
    if dataset not in paths:
        raise ValueError(f"Unknown dataset: {dataset!r}")
    return str(paths[dataset])


def default_expl_dir(
    dataset: str,
    model_name: str,
    fold: int = 1,
    project_root: Optional[str] = None,
) -> str:
    """
    Return the default saliency/explanation output directory.

    Args:
        dataset:      One of "audiomnist", "esc50", "msos".
        model_name:   Model identifier string.
        fold:         ESC-50 fold index (1–5). Ignored for other datasets.
        project_root: Project root directory. Defaults to module-level PROJECT_ROOT.

    Returns:
        Absolute path string to the directory that contains ``<method>.npy`` files.
    """
    root = Path(project_root or PROJECT_ROOT)
    paths = {
        "audiomnist": root / "expl_audiomnist" / model_name,
        "esc50":      root / "expl_esc50"      / model_name / f"fold_{fold}",
        "msos":       root / "expl_msos"       / model_name,
    }
    if dataset not in paths:
        raise ValueError(f"Unknown dataset: {dataset!r}")
    return str(paths[dataset])


def saliency_path(
    dataset: str,
    model_name: str,
    fold: int,
    method: str,
    project_root: Optional[str] = None,
) -> Path:
    """
    Return the full path to a specific attribution .npy file.

    This is a convenience wrapper around ``default_expl_dir`` that appends
    the method filename.

    Args:
        dataset:      One of "audiomnist", "esc50", "msos".
        model_name:   Model identifier string.
        fold:         ESC-50 fold index (1–5).
        method:       Attribution method name (e.g. "gradient_abs").
        project_root: Project root directory. Defaults to module-level PROJECT_ROOT.

    Returns:
        ``Path`` object pointing to ``<expl_dir>/<method>.npy``.
    """
    expl_dir = default_expl_dir(dataset, model_name, fold, project_root)
    return Path(expl_dir) / f"{method}.npy"


def cache_dir(
    out_root: str,
    dataset: str,
    model_name: str,
    fold: int,
    mask_type: str,
) -> Path:
    """
    Return the per-experiment cache directory for PGD / ROAD pre-computation.

    Structure: ``<out_root>/_cache/<dataset>/<model_name>[/fold_<fold>]/<mask_type>``

    Args:
        out_root:   Root output directory (e.g. ``"morf_lerf_spec"``).
        dataset:    Dataset identifier string.
        model_name: Model identifier string.
        fold:       ESC-50 fold index (1–5); subdirectory only added for esc50.
        mask_type:  One of "zero", "pgd", "road".

    Returns:
        ``Path`` object for the cache directory (not yet created).
    """
    base = Path(out_root) / "_cache" / dataset / model_name
    if dataset == "esc50":
        base = base / f"fold_{fold}"
    return base / mask_type


def output_dir(
    out_root: str,
    dataset: str,
    model_name: str,
    fold: int,
    mask_type: str,
) -> Path:
    """
    Return the per-experiment result directory for MoRF/LeRF pickle files.

    Structure: ``<out_root>/<dataset>/<model_name>[/fold_<fold>]/<mask_type>``

    Args:
        out_root:   Root output directory (e.g. ``"morf_lerf_spec"``).
        dataset:    Dataset identifier string.
        model_name: Model identifier string.
        fold:       ESC-50 fold index (1–5); subdirectory only added for esc50.
        mask_type:  One of "zero", "pgd", "road".

    Returns:
        ``Path`` object for the result directory (not yet created).
    """
    base = Path(out_root) / dataset / model_name
    if dataset == "esc50":
        base = base / f"fold_{fold}"
    return base / mask_type