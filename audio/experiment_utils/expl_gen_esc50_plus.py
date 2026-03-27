# expl_gen_esc50.py
import os
import argparse
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from captum.attr import Saliency, NoiseTunnel, IntegratedGradients

from experiment_utils.model.cnn14 import Cnn14
from experiment_utils.model.res1dnet31 import Res1dNet31
from experiment_utils.model.alexnet import AlexNet_Audio
from experiment_utils.model.audionet import AudioNet



# ============================================================
# Reproducibility / device
# ============================================================
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Utils
# ============================================================
def debug_print_shape(name, arr):
    print(f"[DEBUG] {name} saliency shape = {arr.shape}")


def is_waveform(path):
    return "waveform" in path.lower()


def is_spectrogram(path):
    return "spectrogram" in path.lower()


# ============================================================
# ESC-50 Dataset 
# ============================================================
class ESC50WaveDataset(Dataset):
    """audionet / res1dnet31 / cnn14"""

    def __init__(self, split_txt, data_dir):
        self.paths = []
        with open(split_txt) as f:
            for line in f:
                p = line.strip()
                if p and is_waveform(p):
                    self.paths.append(os.path.join(data_dir, p))
        assert len(self.paths) > 0

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        with h5py.File(self.paths[idx], "r") as f:
            x = f["data"][:]          # (1,1,1,T)
            y = int(f["label"][0][0])

        x = np.asarray(x, np.float32).reshape(-1)
        x = torch.from_numpy(x).unsqueeze(0)  # (1,T)
        return x, torch.tensor(y, dtype=torch.long)


class ESC50SpecDataset(Dataset):
    """alexnet"""

    def __init__(self, split_txt, data_dir):
        self.paths = []
        with open(split_txt) as f:
            for line in f:
                p = line.strip()
                if p and is_spectrogram(p):
                    self.paths.append(os.path.join(data_dir, p))
        assert len(self.paths) > 0

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        with h5py.File(self.paths[idx], "r") as f:
            x = f["data"][0, 0]      # (F, T)  <-- 跟 train 一樣
            y = int(f["label"][0][0])

        x = torch.from_numpy(x).float()  # (F,T)

        # normalize（跟 train 一樣）
        mean = x.mean()
        std  = x.std()
        x = (x - mean) / (std + 1e-6)

        x = x.unsqueeze(0)  # (1, F, T)
        return x, torch.tensor(y, dtype=torch.long)


def get_loader(split_txt, data_dir, model, batch_size, shuffle, num_workers):
    if model == "alexnet":
        ds = ESC50SpecDataset(split_txt, data_dir)
    else:
        ds = ESC50WaveDataset(split_txt, data_dir)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


# ============================================================
# CNN14 log-mel wrapper
# ============================================================
def _call_conv_block(conv_block, x, pool_size):
    try:
        return conv_block(x, pool_size=pool_size, pool_type="avg")
    except TypeError:
        return conv_block(x, pool_size=pool_size)


class CNN14LogmelHelper:
    def __init__(self, m):
        self.m = m

    @torch.no_grad()
    def waveform_to_logmel(self, wav):
        if wav.ndim == 3:
            wav = wav.squeeze(1)  # (B,T)
        spec = self.m.spectrogram_extractor(wav)
        return self.m.logmel_extractor(spec)

    def forward_from_logmel(self, logmel):
        x = logmel
        x = x.transpose(1, 3)
        x = self.m.bn0(x)
        x = x.transpose(1, 3)

        x = _call_conv_block(self.m.conv_block1, x, (2, 2))
        x = _call_conv_block(self.m.conv_block2, x, (2, 2))
        x = _call_conv_block(self.m.conv_block3, x, (2, 2))
        x = _call_conv_block(self.m.conv_block4, x, (2, 2))
        x = _call_conv_block(self.m.conv_block5, x, (2, 2))
        x = _call_conv_block(self.m.conv_block6, x, (1, 1))

        x = torch.mean(x, dim=3)
        x = torch.max(x, dim=2)[0] + torch.mean(x, dim=2)
        x = torch.relu(self.m.fc1(x))
        return self.m.fc_out(x)


class CNN14LogmelWrapper(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.helper = CNN14LogmelHelper(m)

    def forward(self, logmel):
        return self.helper.forward_from_logmel(logmel)


class CNN14LogmelBatchLoader:
    def __init__(self, helper, wave_loader):
        self.helper = helper
        self.wave_loader = wave_loader

    def __iter__(self):
        for wav, y in self.wave_loader:
            wav = wav.to(device)
            with torch.no_grad():
                logmel = self.helper.waveform_to_logmel(wav)
            yield logmel.cpu(), y

    def __len__(self):
        return len(self.wave_loader)


# ============================================================
# Captum helpers
# ============================================================
def get_saliency(model, loader, abs_val=False):
    sal = Saliency(model)
    outs = []

    for x, y in tqdm(loader, desc="Saliency"):
        x = x.to(device)
        y = y.to(device)
        x.requires_grad_(True)
        model.zero_grad(set_to_none=True)

        target = y.tolist() if y.ndim > 0 else int(y.item())
        e = sal.attribute(x, target=target, abs=abs_val)

        outs.append(e.detach().cpu().numpy())

    return np.concatenate(outs, axis=0)


def get_saliency_var(model, loader, nt_type, nt_samples, nt_batch, stdevs, abs_val=False):
    nt = NoiseTunnel(Saliency(model))
    outs = []

    for x, y in tqdm(loader, desc=f"NoiseTunnel ({nt_type})"):
        x = x.to(device)
        y = y.to(device)
        x.requires_grad_(True)
        model.zero_grad(set_to_none=True)

        e = nt.attribute(
            x,
            target=y,
            nt_type=nt_type,
            nt_samples=nt_samples,
            nt_samples_batch_size=nt_batch,
            stdevs=stdevs,
            abs=abs_val,
        )
        outs.append(e.detach().cpu().numpy())

    return np.concatenate(outs, axis=0)


def get_integrad(model, loader, abs_val=False, steps=50):
    ig = IntegratedGradients(model)
    outs = []

    for x, y in tqdm(loader, desc="IntegratedGradients"):
        x = x.to(device)
        y = y.to(device)
        x.requires_grad_(True)
        model.zero_grad(set_to_none=True)

        e = ig.attribute(x, target=y, n_steps=steps)
        if abs_val:
            e = e.abs()

        outs.append(e.detach().cpu().numpy())

    return np.concatenate(outs, axis=0)


def make_random_expl_like(expl):
    rand = np.zeros_like(expl)
    for i in range(expl.shape[0]):
        mu = expl[i].mean()
        sigma = expl[i].std() + 1e-8
        rand[i] = np.random.normal(mu, sigma, size=expl[i].shape)
    return rand


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["cnn14", "res1dnet31", "alexnet", "audionet"], required=True)
    parser.add_argument("--data_dir", default="data/ESC50/preprocessed_data")
    parser.add_argument("--fold", type=int, default="1")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--only_methods", default="")
    args = parser.parse_args()

    base = Path(args.data_dir)
    test_txt = base / f"ESC50_fold{args.fold}_test.txt"

    test_loader = get_loader(
        test_txt,
        args.data_dir,
        args.model,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------
    if args.model == "alexnet":
        model = AlexNet_Audio(num_classes=50).to(device)
    elif args.model == "audionet":
        model = AudioNet(num_classes=50).to(device)
    elif args.model == "res1dnet31":
        model = Res1dNet31(num_classes=50).to(device)
    else:
        model = Cnn14(num_classes=50).to(device)

    ckpt_path = (
        Path("checkpoints/esc50")
        / args.model
        / f"fold_{args.fold}"
        / "best_model.pth"
    )

    model.load_state_dict(torch.load(ckpt_path, map_location="cpu")["state_dict"])
    model.eval()

    # --------------------------------------------------------
    # Expl setup
    # --------------------------------------------------------
    is_cnn14 = args.model == "cnn14"

    if is_cnn14:
        wrapper = CNN14LogmelWrapper(model).to(device)
        expl_model = wrapper
        expl_loader = CNN14LogmelBatchLoader(wrapper.helper, test_loader)
    else:
        expl_model = model
        expl_loader = test_loader

    # --------------------------------------------------------
    # Output dir (YOUR REQUIRED PATH)
    # --------------------------------------------------------
    out_dir = Path(
        "/mnt/left/home/2025/tony/hsinyuan/AIM_audio/expl_esc50"
    ) / args.model / f"fold_{args.fold}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Methods
    # --------------------------------------------------------
    all_methods = [
        "gradient",
        "gradinput",
        "smoothgrad", "smoothgrad_sq", "vargrad",
        "integrad",
        "gradient_abs", "gradinput_abs",
        "smoothgrad_abs", "integrad_abs",
        "random",
    ]

    methods = all_methods if not args.only_methods else [
        m.strip() for m in args.only_methods.split(",") if m.strip()
    ]

    nts, bn_nt, stdevs = 16, 16, 1e-2
    grad_for_random = None

    # --------------------------------------------------------
    # Run explanations
    # --------------------------------------------------------
    for m in methods:
        print(f"\n===== {m} =====")
        save_path = out_dir / f"{m}.npy"

        if m == "gradient":
            g = get_saliency(expl_model, expl_loader)
            if is_cnn14:
                g = g.transpose(0, 1, 3, 2)
            debug_print_shape("gradient", g)
            np.save(save_path, g)
            grad_for_random = g
            continue

        if m == "gradient_abs":
            g = get_saliency(expl_model, expl_loader, True)
            if is_cnn14:
                g = g.transpose(0, 1, 3, 2)
            debug_print_shape("gradient_abs", g)
            np.save(save_path, g)
            continue

        if m in ["gradinput", "gradinput_abs"]:
            g = get_saliency(expl_model, expl_loader)
            GI, idx = [], 0

            for x, _ in expl_loader:
                bsz = x.shape[0]
                GI.append(g[idx:idx+bsz] * x.numpy())
                idx += bsz

            GI = np.concatenate(GI, axis=0)
            if is_cnn14:
                GI = GI.transpose(0, 1, 3, 2)
            if m.endswith("_abs"):
                GI = np.abs(GI)

            debug_print_shape(m, GI)
            np.save(save_path, GI)
            continue

        if m in ["smoothgrad", "smoothgrad_sq", "vargrad"]:
            e = get_saliency_var(
                expl_model, expl_loader,
                m, nts, bn_nt, stdevs
            )
            if is_cnn14:
                e = e.transpose(0, 1, 3, 2)
            debug_print_shape(m, e)
            np.save(save_path, e)
            continue

        if m == "smoothgrad_abs":
            e = get_saliency_var(
                expl_model, expl_loader,
                "smoothgrad", nts, bn_nt, stdevs, True
            )
            if is_cnn14:
                e = e.transpose(0, 1, 3, 2)
            debug_print_shape("smoothgrad_abs", e)
            np.save(save_path, e)
            continue

        if m == "integrad":
            e = get_integrad(expl_model, expl_loader)
            if is_cnn14:
                e = e.transpose(0, 1, 3, 2)
            debug_print_shape("integrad", e)
            np.save(save_path, e)
            continue

        if m == "integrad_abs":
            e = get_integrad(expl_model, expl_loader, True)
            if is_cnn14:
                e = e.transpose(0, 1, 3, 2)
            debug_print_shape("integrad_abs", e)
            np.save(save_path, e)
            continue

        if m == "random":
            if grad_for_random is None:
                grad_for_random = get_saliency(expl_model, expl_loader)
                if is_cnn14:
                    grad_for_random = grad_for_random.transpose(0, 1, 3, 2)

            r = make_random_expl_like(grad_for_random)
            debug_print_shape("random", r)
            np.save(save_path, r)
            continue

    print("\nAll ESC-50 saliency maps saved.")
    print("Saved to:", out_dir)


if __name__ == "__main__":
    main()
