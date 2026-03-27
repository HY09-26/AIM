# expl_gen_msos.py
import os
import argparse
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchaudio
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
# Dataset (MATCH train_msos.py)
class MSOSDatasetForExpl(Dataset):
    def __init__(self, split_txt, data_dir, model_type):
        self.paths = []
        with open(split_txt) as f:
            for line in f:
                p = line.strip()
                if p:
                    self.paths.append(os.path.join(data_dir, p))
        assert len(self.paths) > 0, f"Empty split file: {split_txt}"

        self.model_type = model_type

        if self.model_type == "alexnet":
            self.spec = torchaudio.transforms.Spectrogram(
                n_fft=1024,
                hop_length=512,
                power=1,
            )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        with h5py.File(self.paths[idx], "r") as f:
            x = f["data"][0, 0, 0]   # (T,)
            y = int(f["label"][0][0])

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)

        if self.model_type == "alexnet":
            x = self.spec(x)  # (F, T)
            x = (x - x.mean()) / (x.std() + 1e-8)
            x = x.unsqueeze(0)  # (1, F, T)
        else:
            x = x.unsqueeze(0)  # (1, T)

        return x, y


def get_loader(split_txt, data_dir, batch_size, shuffle, num_workers, model_type):
    ds = MSOSDatasetForExpl(split_txt, data_dir, model_type)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


# ============================================================
# CNN14 log-mel wrapper (same idea as your original)
# ============================================================
def _call_conv_block(conv_block, x, pool_size):
    try:
        return conv_block(x, pool_size=pool_size, pool_type="avg")
    except TypeError:
        return conv_block(x, pool_size=pool_size)


class CNN14LogmelHelper:
    def __init__(self, cnn14_model: nn.Module):
        self.m = cnn14_model

    @torch.no_grad()
    def waveform_to_logmel(self, wav: torch.Tensor) -> torch.Tensor:
        # wav: (B, 1, T) OR (B, T) depending on model implementation
        # Your Cnn14 usually expects (B, T). We'll flatten if needed.
        if wav.ndim == 3:
            wav = wav.squeeze(1)  # (B, T)
        spec = self.m.spectrogram_extractor(wav)
        logmel = self.m.logmel_extractor(spec)
        return logmel

    def forward_from_logmel(self, logmel: torch.Tensor) -> torch.Tensor:
        x = logmel

        x = x.transpose(1, 3)
        x = self.m.bn0(x)
        x = x.transpose(1, 3)

        if self.m.training:
            x = self.m.spec_augmenter(x)

        x = _call_conv_block(self.m.conv_block1, x, pool_size=(2, 2))
        x = nn.functional.dropout(x, p=0.2, training=self.m.training)
        x = _call_conv_block(self.m.conv_block2, x, pool_size=(2, 2))
        x = nn.functional.dropout(x, p=0.2, training=self.m.training)
        x = _call_conv_block(self.m.conv_block3, x, pool_size=(2, 2))
        x = nn.functional.dropout(x, p=0.2, training=self.m.training)
        x = _call_conv_block(self.m.conv_block4, x, pool_size=(2, 2))
        x = nn.functional.dropout(x, p=0.2, training=self.m.training)
        x = _call_conv_block(self.m.conv_block5, x, pool_size=(2, 2))
        x = nn.functional.dropout(x, p=0.2, training=self.m.training)
        x = _call_conv_block(self.m.conv_block6, x, pool_size=(1, 1))
        x = nn.functional.dropout(x, p=0.2, training=self.m.training)

        x = torch.mean(x, dim=3)
        x1, _ = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2

        x = nn.functional.dropout(x, p=0.5, training=self.m.training)
        x = nn.functional.relu_(self.m.fc1(x))
        logits = self.m.fc_out(x)
        return logits


class CNN14LogmelWrapper(nn.Module):
    def __init__(self, cnn14_model: nn.Module):
        super().__init__()
        self.base = cnn14_model
        self.helper = CNN14LogmelHelper(cnn14_model)

    def forward(self, logmel):
        return self.helper.forward_from_logmel(logmel)


class CNN14LogmelBatchLoader:
    """
    Input: wave_loader yields (wav, y)
      wav is (B,1,T) (from MSOSDatasetForExpl)
    Output: yields (logmel_cpu, y) where logmel_cpu matches wrapper input.
    """
    def __init__(self, helper, wave_loader):
        self.helper = helper
        self.wave_loader = wave_loader

    def __iter__(self):
        for wav, y in self.wave_loader:
            wav = wav.to(device)
            with torch.no_grad():
                logmel = self.helper.waveform_to_logmel(wav)
            yield logmel.detach().cpu(), y

    def __len__(self):
        return len(self.wave_loader)


# ============================================================
# Captum explanation functions
# ============================================================
def get_saliency(model, loader, abs_val=False):
    model.eval()
    expls = []
    sal = Saliency(model)

    for x, y in tqdm(loader, desc="Saliency"):
        x = x.to(device)
        y = y.to(device)
        x.requires_grad_(True)
        model.zero_grad(set_to_none=True)

        e = sal.attribute(x, target=y, abs=abs_val)
        expls.append(e.detach().cpu().numpy())

    return np.concatenate(expls, axis=0)


def get_saliency_var(model, loader, nt_type, nt_samples, nt_batch_size, stdevs, abs_val=False):
    model.eval()
    expls = []
    base = Saliency(model)
    nt = NoiseTunnel(base)

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
            nt_samples_batch_size=nt_batch_size,
            stdevs=stdevs,
            abs=abs_val,
        )
        expls.append(e.detach().cpu().numpy())

    return np.concatenate(expls, axis=0)


def get_integrad(model, loader, abs_val=False, steps=50):
    model.eval()
    expls = []
    ig = IntegratedGradients(model)

    for x, y in tqdm(loader, desc="IntegratedGradients"):
        x = x.to(device)
        y = y.to(device)
        x.requires_grad_(True)
        model.zero_grad(set_to_none=True)

        e = ig.attribute(
            x,
            target=y,
            n_steps=steps,
            method="gausslegendre",
            return_convergence_delta=False,
        )
        if abs_val:
            e = e.abs()

        expls.append(e.detach().cpu().numpy())

    return np.concatenate(expls, axis=0)


def make_random_expl_like(expl):
    rand = np.zeros_like(expl)
    for i in range(expl.shape[0]):
        mu = expl[i].mean()
        sigma = expl[i].std() if expl[i].std() > 1e-8 else 1e-8
        rand[i] = np.random.normal(mu, sigma, size=expl[i].shape)
    return rand


# ============================================================
# Checkpoint loader
# ============================================================
def load_ckpt_state_dict(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt)
    return ckpt


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["cnn14", "res1dnet31", "audionet", "alexnet"])
    parser.add_argument("--data_dir", default="data/MSoS/preprocessed_data")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--ckpt_path", default="")
    parser.add_argument("--expl_dir", default="")
    parser.add_argument("--only_methods", default="")
    args = parser.parse_args()

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

    EXPL_DIR = args.expl_dir or os.path.join(PROJECT_ROOT, "expl_msos")

    base = Path(args.data_dir)
    test_txt = base / "MSOS_test.txt"

    # IMPORTANT: loader preprocessing matches train_msos.py
    test_loader = get_loader(
        test_txt,
        args.data_dir,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        model_type=args.model,
    )

    # build model
    NUM_CLASSES = 5
    if args.model == "cnn14":
        model = Cnn14(num_classes=NUM_CLASSES).to(device)
    elif args.model == "res1dnet31":
        model = Res1dNet31(num_classes=NUM_CLASSES).to(device)
    elif args.model == "audionet":
        model = AudioNet(num_classes=NUM_CLASSES).to(device)
    else:
        model = AlexNet_Audio(num_classes=NUM_CLASSES).to(device)

    ckpt_path = (
        args.ckpt_path
        or os.path.join(
            SCRIPT_DIR,
            "checkpoints", "msos", args.model, "best_model.pth"
        )
    )
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print("[INFO] Loading ckpt:", ckpt_path)
    load_ckpt_state_dict(model, ckpt_path)
    model.eval()

    # CNN14 special handling: explain in log-mel domain (but loader matches training = waveform)
    is_cnn14 = (args.model == "cnn14")
    cnn14_wrapper = None
    cnn14_helper = None
    cnn14_logmel_loader = None

    if is_cnn14:
        cnn14_wrapper = CNN14LogmelWrapper(model).to(device)
        cnn14_wrapper.eval()
        cnn14_helper = cnn14_wrapper.helper
        cnn14_logmel_loader = CNN14LogmelBatchLoader(cnn14_helper, test_loader)

    expl_model = cnn14_wrapper if is_cnn14 else model
    expl_loader = cnn14_logmel_loader if is_cnn14 else test_loader

    model_expl_dir = os.path.join(EXPL_DIR, args.model)
    os.makedirs(model_expl_dir, exist_ok=True)

    all_methods = [
        "gradient", "gradinput",
        "smoothgrad", "smoothgrad_sq", "vargrad",
        "integrad",
        "gradient_abs", "gradinput_abs",
        "smoothgrad_abs", "integrad_abs",
        "random",
    ]

    methods = (
        [m.strip() for m in args.only_methods.split(",") if m.strip()]
        if args.only_methods else all_methods
    )

    nts = 16
    bn_nt = 16
    stdevs = 1e-2
    grad_for_random = None

    for m in methods:
        print(f"\n===== Method: {m} =====")
        save_path = os.path.join(model_expl_dir, f"{m}.npy")

        if m == "gradient":
            grad = get_saliency(expl_model, expl_loader, abs_val=False)

            # keep your old cnn14 transpose convention (so downstream code stays same)
            if is_cnn14:
                grad = np.transpose(grad, (0, 1, 3, 2))

            print(f"[DEBUG] gradient saliency shape = {grad.shape}")
            np.save(save_path, grad)
            grad_for_random = grad
            continue

        if m == "gradient_abs":
            grad = get_saliency(expl_model, expl_loader, abs_val=True)
            if is_cnn14:
                grad = np.transpose(grad, (0, 1, 3, 2))
            np.save(save_path, grad)
            continue

        if m in ["gradinput", "gradinput_abs"]:
            grad = get_saliency(expl_model, expl_loader, abs_val=False)
            GI_list, idx = [], 0

            if is_cnn14:
                # expl_loader yields (logmel_cpu, y)
                for logmel_cpu, _ in expl_loader:
                    x_np = logmel_cpu.numpy()
                    bsz = x_np.shape[0]
                    GI_list.append(grad[idx:idx+bsz] * x_np)
                    idx += bsz
                GI = np.concatenate(GI_list, axis=0)
                GI = np.transpose(GI, (0, 1, 3, 2))  # match your saved layout

            else:
                # expl_loader yields model input directly:
                #  - alexnet: (B,1,F,T)
                #  - others:  (B,1,T)
                for x_batch, _ in test_loader:
                    x_np = x_batch.numpy()
                    bsz = x_np.shape[0]
                    GI_list.append(grad[idx:idx+bsz] * x_np)
                    idx += bsz
                GI = np.concatenate(GI_list, axis=0)

            if m.endswith("_abs"):
                GI = np.abs(GI)
            np.save(save_path, GI)
            continue

        if m in ["smoothgrad", "smoothgrad_sq", "vargrad"]:
            expl = get_saliency_var(
                expl_model, expl_loader,
                m, nts, bn_nt, stdevs, abs_val=False
            )
            if is_cnn14:
                expl = np.transpose(expl, (0, 1, 3, 2))
            np.save(save_path, expl)
            continue

        if m == "smoothgrad_abs":
            expl = get_saliency_var(
                expl_model, expl_loader,
                "smoothgrad", nts, bn_nt, stdevs, abs_val=True
            )
            if is_cnn14:
                expl = np.transpose(expl, (0, 1, 3, 2))
            np.save(save_path, expl)
            continue

        if m == "integrad":
            ig = get_integrad(expl_model, expl_loader, abs_val=False)
            if is_cnn14:
                ig = np.transpose(ig, (0, 1, 3, 2))
            np.save(save_path, ig)
            continue

        if m == "integrad_abs":
            ig = get_integrad(expl_model, expl_loader, abs_val=True)
            if is_cnn14:
                ig = np.transpose(ig, (0, 1, 3, 2))
            np.save(save_path, ig)
            continue

        if m == "random":
            if grad_for_random is None:
                grad_for_random = get_saliency(expl_model, expl_loader, abs_val=False)
                if is_cnn14:
                    grad_for_random = np.transpose(grad_for_random, (0, 1, 3, 2))
            rand_expl = make_random_expl_like(grad_for_random)
            np.save(save_path, rand_expl)
            continue

        raise ValueError(f"Unknown method: {m}")

    print("\nAll explanations saved successfully!")
    print("Saved to:", model_expl_dir)


if __name__ == "__main__":
    main()
