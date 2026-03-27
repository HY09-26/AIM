# expl_gen_audiomnist.py
import os
import h5py
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from captum.attr import Saliency, NoiseTunnel, IntegratedGradients

from experiment_utils.model.cnn14 import Cnn14
from experiment_utils.model.res1dnet31 import Res1dNet31Lite
from experiment_utils.model.alexnet import AlexNet_Audio
from experiment_utils.model.audionet import AudioNet



# ============================================================
# Reproducibility / device
# ============================================================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_DIR = os.path.join(
    PROJECT_ROOT, "experiment_utils", "data", "audiomnist", "preprocessed_data"
)
SAVE_DIR = os.path.join(
    PROJECT_ROOT, "experiment_utils", "checkpoints", "audiomnist"
)
EXPL_DIR = os.path.join(
    PROJECT_ROOT, "expl_audiomnist"
)


# ============================================================
# Utils
# ============================================================
def debug_print_shape(name, arr):
    print(f"[DEBUG] {name} saliency shape = {arr.shape}")


# ============================================================
# Dataset / Loader (SELF-CONTAINED)
# ============================================================

class AudioMNISTDataset(Dataset):
    """
    alexnet      -> spectrogram (1,F,T)
    audionet     -> waveform (1,T)
    res1dnet31   -> waveform (1,T)
    cnn14        -> waveform (1,T) -> logmel internally
    """

    def __init__(self, split_txt, data_dir, model_name):
        self.paths = []
        with open(split_txt) as f:
            for line in f:
                p = line.strip()
                if p:
                    self.paths.append(os.path.join(data_dir, p))

        assert len(self.paths) > 0, f"Empty split file: {split_txt}"
        self.model_name = model_name

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        with h5py.File(self.paths[idx], "r") as f:
            x = f["data"][:]           # could be (1,1,T) or (1,1,1,T)
            y = int(f["label"][0][0])

        if self.model_name == "alexnet":
            # spectrogram: expect (1,F,T)
            x = torch.tensor(x, dtype=torch.float32).squeeze(0)
        else:
            # waveform: MUST be (1,T) for Conv1d models
            x = np.asarray(x, np.float32).reshape(-1)  # <<< FIX
            x = torch.from_numpy(x).unsqueeze(0)      # (1,T)

        return x, torch.tensor(y, dtype=torch.long)



def get_loader(split_txt, data_dir, model_name, batch_size, shuffle, num_workers):
    ds = AudioMNISTDataset(split_txt, data_dir, model_name)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


# ============================================================
# CNN14 log-mel wrapper (same as ESC50 / MSOS)
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

        e = sal.attribute(x, target=y, abs=abs_val)
        outs.append(e.detach().cpu().numpy())

    return np.concatenate(outs, axis=0)


def get_saliency_var(
    model, loader, nt_type,
    nt_samples, nt_batch_size, stdevs,
    abs_val=False
):
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
            nt_samples_batch_size=nt_batch_size,
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

        e = ig.attribute(
            x,
            target=y,
            n_steps=steps,
            method="gausslegendre",
            return_convergence_delta=False,
        )
        if abs_val:
            e = e.abs()

        outs.append(e.detach().cpu().numpy())

    return np.concatenate(outs, axis=0)


def make_random_expl_like(grad_expl):
    rand = np.zeros_like(grad_expl)
    for i in range(grad_expl.shape[0]):
        mu = grad_expl[i].mean()
        sigma = grad_expl[i].std()
        if sigma < 1e-8:
            sigma = 1e-8
        rand[i] = np.random.normal(mu, sigma, size=grad_expl[i].shape)
    return rand


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    model_name = "audionet"   # alexnet / audionet / res1dnet31 / cnn14
    repeat_idx = 0

    TEST_SPLIT_MAP = {
        "alexnet": "AlexNet_digit_0_test.txt",
        "audionet": "AudioNet_digit_0_test.txt",
        "res1dnet31": "AudioNet_digit_0_test.txt",
        "cnn14": "AudioNet_digit_0_test.txt",
    }

    # --------------------------------------------------------
    # Loader
    # --------------------------------------------------------
    test_loader = get_loader(
        os.path.join(DATASET_DIR, TEST_SPLIT_MAP[model_name]),
        DATASET_DIR,
        model_name,
        batch_size=4,
        shuffle=False,
        num_workers=4,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    model_dict = {
        "alexnet": AlexNet_Audio,
        "audionet": AudioNet,
        "res1dnet31": Res1dNet31Lite,
        "cnn14": Cnn14,
    }

    ckpt_path = os.path.join(
        SAVE_DIR,
        model_name,
        "best_model.pth",
    )

    print("[INFO] Loading model:", ckpt_path)
    model = model_dict[model_name](num_classes=10).to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # --------------------------------------------------------
    # Expl setup
    # --------------------------------------------------------
    is_cnn14 = model_name == "cnn14"

    if is_cnn14:
        wrapper = CNN14LogmelWrapper(model).to(device)
        expl_model = wrapper
        expl_loader = CNN14LogmelBatchLoader(wrapper.helper, test_loader)
    else:
        expl_model = model
        expl_loader = test_loader

    # --------------------------------------------------------
    # Output dir
    # --------------------------------------------------------
    model_expl_dir = os.path.join(
        EXPL_DIR,
        model_name
    )
    os.makedirs(model_expl_dir, exist_ok=True)

    # --------------------------------------------------------
    # Methods
    # --------------------------------------------------------
    methods = [
        "gradient",
        "gradinput",
        "smoothgrad", "smoothgrad_sq", "vargrad",
        "integrad",
        "gradient_abs", "gradinput_abs",
        "smoothgrad_abs",
        "integrad_abs",
        "random",
    ]

    nts, bn_nt, stdevs = 16, 16, 1e-2
    grad_for_random = None

    # --------------------------------------------------------
    # Run explanations
    # --------------------------------------------------------
    for m in methods:
        print(f"\n===== Method: {m} =====")
        save_path = os.path.join(model_expl_dir, f"{m}.npy")

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

    print("\n All AudioMNIST explanations saved.")
    print(" Saved to:", model_expl_dir)
