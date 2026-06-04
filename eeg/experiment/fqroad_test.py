"""Frequency masking — 1/f polynomial-fitting (ROAD) replacement (MoRF / LeRF)."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pickle
import shutil
import numpy as np
from scipy.io import loadmat, savemat
from scipy.optimize import curve_fit
from tqdm import tqdm
import torch
import torch.nn as nn

from experiment_utils.model import EEGNet, InterpretableCNN, SCCNet, EEGNet_SSVEP, InterpretableCNN_SSVEP
from experiment_utils.utils import evaluate_an_epoch, evaluate_an_epoch_auc, get_loader
from experiment_utils.eeg_config import get_config, IRISHSIEH_DIR, SAVE_DIR, mat_file
from experiment_utils.masking_utils import find_neighbors

parser = argparse.ArgumentParser()
parser.add_argument("--dataname",     default="SSVEP", choices=["MI", "ERN", "SSVEP"])
parser.add_argument("--abs_saliency", type=int, choices=[0, 1], required=True,
                    help="1 = absolute saliency values")
parser.add_argument("--rep",          type=int, required=True,
                    help="Repeat index; SSVEP only has reps 1 and 2")
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

EXPL_DIR  = f"{IRISHSIEH_DIR}/repeat{rep}/expl/{dataname}"
OUT_DIR   = os.path.join(SAVE_DIR, f"repeat{rep}/fq_test_road/{dataname}_si")
SPINT_DIR = os.path.join(SAVE_DIR, f"repeat{rep}/fq_test_road/res_data/{dataname}")
os.makedirs(OUT_DIR,   exist_ok=True)
os.makedirs(SPINT_DIR, exist_ok=True)

modes  = [('mo', -1), ('le', 1)]
leg    = ['gradient', 'gradientxinput', 'smoothgrad', 'smoothgrad_sq',
          'vargrad', 'inte_grad', 'random']
fq_dim = 10
n_subs = len(cfg.sub_list)
filler = '' if args.abs_saliency else 'n'
sfreq  = cfg.sfreq
lowfq  = cfg.lowfq
highfq = cfg.highfq
sqrt2  = np.sqrt(2)


def _powlaw_imputation(benign_fq, grad_fq, nb, ts, freqs):
    """Fit 1/f power law and synthesise replacement spectrum for one trial."""
    uq_idx = np.where(np.isclose(freqs, -(sfreq / 2), 0.5))[0][0]
    ch     = benign_fq.shape[0]

    def overf(x, aa, bb, cc, dd):
        yfit = aa * x**3 + bb * x**2 + cc * x + dd
        if freqs[nb[0]] > 0:
            yfit[x == 1 / freqs[nb[0]]] = np.abs(grad_fq[:, nb[0]]).mean() / sqrt2
        else:
            yfit[x > 1 / freqs[lowfq]] = np.abs(grad_fq[:, lowfq]).mean() / sqrt2
        yfit[x == 1 / freqs[nb[1]]] = np.abs(grad_fq[:, nb[1]]).mean() / sqrt2
        return yfit

    popt, _ = curve_fit(overf, 1 / freqs[lowfq:uq_idx],
                        np.abs(benign_fq[:, lowfq:uq_idx]).mean(axis=0) / sqrt2)
    powlaw = np.poly1d(popt)

    n_band = nb[1] + 1 - nb[0]
    impt   = np.zeros(n_band)
    for i in range(n_band):
        fi = nb[0] + i
        if fi < lowfq:
            impt[i:] = powlaw(1 / freqs[lowfq])
        else:
            impt[i:] = powlaw(1 / freqs[fi:nb[1]+1])
            break

    impt_r = np.repeat(impt[None, :], ch, axis=0)
    impt_i = np.repeat(impt[None, :], ch, axis=0)
    for c in range(ch):
        impt_r[c] *= np.sign(benign_fq[c, nb[0]:nb[1]+1].real)
        impt_i[c] *= np.sign(benign_fq[c, nb[0]:nb[1]+1].imag)

    return impt_r + 1j * impt_i, popt


def mask_freq_road(model, modelname, xtest, benign_fq, grad_fq, k_pct,
                   labels, sub, method, mode):
    """Replace a frequency band with 1/f-fitted spectrum, cache to disk, evaluate."""
    dir_str  = 'abs' if args.abs_saliency else 'nabs'
    cache_dir = os.path.join(SPINT_DIR, modelname, dir_str)
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f'sub{sub}_top{k_pct:02d}_{method}_{mode[0]}.mat')

    if os.path.isfile(cache_path):
        mixture = loadmat(cache_path)['road']
    else:
        ts     = xtest.shape[-1]
        freqs  = np.fft.fftfreq(ts, d=1 / sfreq)
        tr     = xtest.shape[0]
        mixture_fq = benign_fq.copy()
        neighbors  = np.zeros((tr, 3))
        fits       = np.zeros((tr, 4))

        for t in range(tr):
            _, nb, _ = find_neighbors(
                benign_fq[t, :, 1:np.where(np.isclose(freqs, -(sfreq / 2), 0.5))[0][0]],
                np.abs(grad_fq[t, :, 1:np.where(np.isclose(freqs, -(sfreq / 2), 0.5))[0][0]]),
                k_pct * 0.01,
                mode[0],
            )
            neighbors[t] = [0, nb[0], nb[1]]
            synth, popt   = _powlaw_imputation(benign_fq[t], grad_fq[t], nb, ts, freqs)
            fits[t]       = popt

            mixture_fq[t, :, nb[0]:nb[1]+1] = synth
            if nb[0] > 0:
                mixture_fq[t, :, ts - nb[1]:ts - nb[0]+1] = np.flip(synth.real, axis=-1) - 1j * synth.imag
            else:
                n_band = nb[1] - nb[0]
                mixture_fq[t, :, ts - nb[1]:ts - nb[0]+1] = (
                    np.flip(synth.real, axis=-1)[:, :n_band] - 1j * synth.imag[:, :n_band]
                )

        mixture = np.fft.ifft(mixture_fq).real
        savemat(cache_path, dict(road=mixture, neighbor=neighbors, fit=fits))

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
                grad_fq   = np.fft.fft(grads)

                for k in range(1, fq_dim + 1):
                    loss, acc, out = mask_freq_road(
                        testmodel, m, xtest, benign_fq, grad_fq,
                        k * 5, ytest, sub, l, modes[0])
                    hist['acc'][s][k-1]  = acc
                    hist['loss'][s][k-1] = loss
                    hist['out'][s][k-1]  = out

                    loss, acc, out = mask_freq_road(
                        testmodel, m, xtest, benign_fq, grad_fq,
                        k * 5, ytest, sub, l, modes[1])
                    hist1['acc'][s][k-1]  = acc
                    hist1['loss'][s][k-1] = loss
                    hist1['out'][s][k-1]  = out

            hists.append(hist)
            hists1.append(hist1)

        with open(os.path.join(OUT_DIR, f'{m}_fq_{filler}abs_{modes[0][0]}rf.pickle'), 'wb') as f:
            pickle.dump(hists, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(OUT_DIR, f'{m}_fq_{filler}abs_{modes[1][0]}rf.pickle'), 'wb') as f:
            pickle.dump(hists1, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Clean up per-model cache now that results are saved
        cache_model_dir = os.path.join(SPINT_DIR, m)
        if os.path.isdir(cache_model_dir):
            shutil.rmtree(cache_model_dir)
