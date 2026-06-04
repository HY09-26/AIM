"""Frequency masking — adversarial example (AE) replacement (MoRF / LeRF)."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pickle
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm
import torch
import torch.nn as nn

from experiment_utils.model import EEGNet, InterpretableCNN, SCCNet, EEGNet_SSVEP, InterpretableCNN_SSVEP
from experiment_utils.utils import evaluate_an_epoch, evaluate_an_epoch_auc, get_loader
from experiment_utils.eeg_config import get_config, IRISHSIEH_DIR, SAVE_DIR, mat_file
from experiment_utils.masking_utils import find_neighbors

parser = argparse.ArgumentParser()
parser.add_argument("--dataname",     default="ERN", choices=["MI", "ERN", "SSVEP"])
parser.add_argument("--abs_saliency", type=int, choices=[0, 1], required=True,
                    help="1 = absolute saliency values")
parser.add_argument("--rep",          type=int, required=True,
                    help="Repeat index (0-based, 0–4)")
args = parser.parse_args()

cfg      = get_config(args.dataname)
dataname = args.dataname
rep      = args.rep
eval_fn  = evaluate_an_epoch_auc if cfg.use_auc else evaluate_an_epoch

os.environ["CUDA_VISIBLE_DEVICES"]    = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.set_default_dtype(torch.float64)
torch.backends.cudnn.benchmark    = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

AE_DIR   = f"{IRISHSIEH_DIR}/repeat{rep}/ae/{dataname}"
EXPL_DIR = f"{IRISHSIEH_DIR}/repeat{rep}/expl/{dataname}"
OUT_DIR  = os.path.join(SAVE_DIR, f"repeat{rep}/fq_test_ae/{dataname}/fft_full_single")
os.makedirs(OUT_DIR, exist_ok=True)

modes  = [('mo', -1), ('le', 1)]
leg    = ['gradient', 'gradientxinput', 'smoothgrad', 'smoothgrad_sq',
          'vargrad', 'inte_grad', 'random']
fq_dim = 10
n_subs = len(cfg.sub_list)
filler = '' if args.abs_saliency else 'n'
sfreq  = cfg.sfreq


def mask_freq_ae(model, xtest, aes, benign_fq, ae_fq, grad_fq, k_pct, labels, mode):
    """Replace a frequency band with adversarial spectrum, evaluate model."""
    ts           = xtest.shape[-1]
    f_local      = np.fft.fftfreq(ts, d=1 / sfreq)
    uq_idx       = np.where(np.isclose(f_local, -(sfreq / 2), 0.5))[0][0]

    # Remove DC offset before operating in frequency domain
    dc_b = xtest.mean(axis=-1)[:, :, None]
    dc_a = aes.mean(axis=-1)[:, :, None]
    xtest_ac = xtest - np.repeat(dc_b, ts, axis=-1)
    aes_ac   = aes   - np.repeat(dc_a, ts, axis=-1)

    fseries_data    = np.fft.fft(xtest_ac)
    fseries_ae      = np.fft.fft(aes_ac)
    fseries_mixture = fseries_data.copy()

    for t in range(xtest.shape[0]):
        _, nb, _ = find_neighbors(
            fseries_data[t, :, 1:uq_idx],
            np.abs(grad_fq[t, :, 1:uq_idx]),
            np.round(k_pct * 0.01, 2),
            mode,
        )
        fseries_mixture[t, :, nb[0]:nb[1]+1]            = fseries_ae[t, :, nb[0]:nb[1]+1]
        fseries_mixture[t, :, ts - nb[1]:ts - nb[0]+1]  = fseries_ae[t, :, ts - nb[1]:ts - nb[0]+1]

    mixture = np.fft.ifft(fseries_mixture).real
    _, loader = get_loader(cfg.batch_size, mixture, labels)
    loss, acc, out = eval_fn(model, device, loader, nn.CrossEntropyLoss())
    return loss, acc, out


if __name__ == '__main__':
    model_type = (
        {'eegnet': EEGNet_SSVEP, 'icnn': InterpretableCNN_SSVEP, 'sccnet': SCCNet}
        if cfg.use_ssvep_model else
        {'eegnet': EEGNet, 'icnn': InterpretableCNN, 'sccnet': SCCNet}
    )

    for m, ModelClass in model_type.items():
        hists, hists1 = [], []

        for l in leg:
            if args.abs_saliency and l in ('smoothgrad_sq', 'vargrad'):
                continue

            testmodel = ModelClass(**cfg.kwerg).to(device)
            print(f"rep{rep} {m} {l}")

            hist  = dict(acc =np.zeros((n_subs, fq_dim)),
                         loss=np.zeros((n_subs, fq_dim)),
                         out =np.zeros((n_subs, fq_dim, cfg.n_trials, cfg.n_classes_out)))
            hist1 = dict(acc =np.zeros((n_subs, fq_dim)),
                         loss=np.zeros((n_subs, fq_dim)),
                         out =np.zeros((n_subs, fq_dim, cfg.n_trials, cfg.n_classes_out)))

            for s in tqdm(range(n_subs)):
                sub = cfg.sub_list[s]

                ckpt = torch.load(
                    os.path.join(IRISHSIEH_DIR, f"models/{dataname}/bests_repeat{rep}/sub{sub}-{m}.pth"),
                    map_location="cpu", weights_only=False,
                )
                testmodel.load_state_dict(ckpt["state_dict"])
                testmodel.eval()

                mat   = loadmat(os.path.join(cfg.dataset_dir, mat_file(dataname, sub)))
                xtest = mat["x_test"][cfg.xtest_sl]
                ytest = mat["y_test"].squeeze()[cfg.xtest_sl]
                aes   = np.load(os.path.join(AE_DIR, f'{m}/sub{sub}.npy')).squeeze()

                if l != 'gradientxinput':
                    grads = np.load(os.path.join(EXPL_DIR, f'{m}/sub{sub}_{l}.npy'),
                                    allow_pickle=True)
                else:
                    grads = np.load(os.path.join(EXPL_DIR, f'{m}/sub{sub}_gradient.npy'),
                                    allow_pickle=True)
                    grads = np.multiply(grads, mat["x_test"][cfg.gradxi_sl])

                if args.abs_saliency:
                    grads = np.abs(grads)

                benign_fq = np.fft.fft(xtest)
                ae_fq     = np.fft.fft(aes)
                grad_fq   = np.fft.fft(grads)

                for k in range(1, fq_dim + 1):
                    loss, acc, out = mask_freq_ae(
                        testmodel, xtest, aes, benign_fq, ae_fq, grad_fq,
                        k * 5, ytest, modes[0][0])
                    hist['acc'][s][k-1]  = acc
                    hist['loss'][s][k-1] = loss
                    hist['out'][s][k-1]  = out

                    loss, acc, out = mask_freq_ae(
                        testmodel, xtest, aes, benign_fq, ae_fq, grad_fq,
                        k * 5, ytest, modes[1][0])
                    hist1['acc'][s][k-1]  = acc
                    hist1['loss'][s][k-1] = loss
                    hist1['out'][s][k-1]  = out

            hists.append(hist)
            hists1.append(hist1)

        with open(os.path.join(OUT_DIR, f'{m}_fq_{filler}abs_{modes[0][0]}rf.pickle'), 'wb') as f:
            pickle.dump(hists, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(OUT_DIR, f'{m}_fq_{filler}abs_{modes[1][0]}rf.pickle'), 'wb') as f:
            pickle.dump(hists1, f, protocol=pickle.HIGHEST_PROTOCOL)
