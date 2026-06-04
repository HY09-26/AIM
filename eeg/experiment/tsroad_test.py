"""Time-segment masking — MFBB (fractional Brownian bridge) replacement (MoRF / LeRF)."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pickle
import shutil
import numpy as np
import mne
from scipy.io import loadmat, savemat
from tqdm import tqdm
import torch
import torch.nn as nn

from experiment_utils.model import EEGNet, InterpretableCNN, SCCNet, EEGNet_SSVEP, InterpretableCNN_SSVEP
from experiment_utils.utils import evaluate_an_epoch, evaluate_an_epoch_auc, get_loader
from experiment_utils.eeg_config import get_config, IRISHSIEH_DIR, SAVE_DIR, mat_file
from experiment_utils.masking_utils import find_crop
from experiment_utils.mfbb import MFBB

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

EXPL_DIR = f"{IRISHSIEH_DIR}/repeat{rep}/expl/{dataname}"
MFBB_DIR = f"{IRISHSIEH_DIR}/repeat{rep}/mfbbs/{dataname}"
OUT_DIR  = os.path.join(SAVE_DIR, f"repeat{rep}/ts_test_road/{dataname}_sto")
os.makedirs(OUT_DIR,  exist_ok=True)
os.makedirs(MFBB_DIR, exist_ok=True)

modes  = [('mo', -1), ('le', 1)]
leg    = ['gradient', 'gradientxinput', 'smoothgrad', 'smoothgrad_sq',
          'vargrad', 'inte_grad', 'random']
ts_dim = 11
n_subs = len(cfg.sub_list)
filler = '' if args.abs_saliency else 'n'

# Per-dataset bandpass cutoffs for post-MFBB filtering
_BANDPASS = {'MI': (4, 38), 'ERN': (1, 40), 'SSVEP': (1, 50)}
l_freq, h_freq = _BANDPASS[dataname]

# MNE info object for bandpass filtering
_CH_NAMES = {
    'MI': ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
           'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
           'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
           'P1', 'Pz', 'P2', 'POz'],
    'ERN': ['Fp1', 'Fp2', 'AF7', 'AF3', 'AF4', 'AF8',
            'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
            'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8',
            'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8',
            'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8',
            'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
            'PO7', 'POz', 'PO8', 'O1', 'O2'],
    'SSVEP': ['PO7', 'PO3', 'O1', 'POz', 'Oz', 'PO4', 'O2', 'PO8'],
}
_mne_info = mne.create_info(
    ch_names=_CH_NAMES[dataname],
    sfreq=cfg.kwerg['sfreq'],
    ch_types="eeg",
)


def mask_timeseg_road(model, modelname, benigns, labels, sub, method,
                      start_idx, wsize, topk, mode):
    """Replace a time window with MFBB-generated noise, filter, cache, evaluate."""
    dir_end    = filler + 'abs'
    cache_dir  = os.path.join(MFBB_DIR, modelname, dir_end, f'sub{sub}')
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f'sub{sub}_top{topk}_{method}_{mode[0]}_norm.mat')

    if os.path.isfile(cache_path):
        filt = loadmat(cache_path)['road']
    else:
        benigns = benigns.squeeze()
        tr, ch, ts = benigns.shape
        mixtures = benigns.copy()

        Ti = np.array([0, wsize // 2, wsize - 1])
        Xi = benigns[:, :, start_idx + Ti]

        for t in range(tr):
            for c in range(ch):
                mfbb, _ = MFBB(Xi[t, c], Xi[t, c, 2], 1, 1e-5, wsize // 2,
                                benigns[t, c].std())
                mfbb -= mfbb.mean()
                mixtures[t, c, start_idx:start_idx + wsize] = mfbb

        # Bandpass filter the MFBB-imputed signal
        raw_data = np.zeros((ch, tr * ts))
        for t in range(tr):
            raw_data[:, t*ts:(t+1)*ts] = mixtures[t]
        raw = mne.io.RawArray(raw_data, _mne_info, verbose='WARNING')
        raw.filter(l_freq=l_freq, h_freq=h_freq, verbose='WARNING')
        data, _ = raw[:]

        filt = benigns.copy()
        for t in range(tr):
            filt[t, :, start_idx:start_idx + wsize] = \
                data[:, t*ts:(t+1)*ts][:, start_idx:start_idx + wsize]

        savemat(cache_path, dict(road=filt, idx=start_idx, win=wsize))

    _, loader = get_loader(cfg.batch_size, filt, labels)
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

            hist  = dict(acc =np.zeros((n_subs, ts_dim)),
                         loss=np.zeros((n_subs, ts_dim)),
                         out =np.zeros((n_subs, ts_dim, cfg.n_trials, cfg.n_classes_out)))
            hist1 = dict(acc =np.zeros((n_subs, ts_dim)),
                         loss=np.zeros((n_subs, ts_dim)),
                         out =np.zeros((n_subs, ts_dim, cfg.n_trials, cfg.n_classes_out)))

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

                for k in range(1, ts_dim + 1):
                    max_idx, min_idx, wsize = find_crop(grads.mean(axis=0),
                                                        np.round(k * 0.05, 2))
                    if wsize % 2 != 0:
                        wsize -= 1

                    loss, acc, out = mask_timeseg_road(
                        testmodel, m, xtest, ytest, sub, l,
                        max_idx, wsize, k, modes[0])
                    hist['acc'][s][k-1]  = acc
                    hist['loss'][s][k-1] = loss
                    hist['out'][s][k-1]  = out

                    loss, acc, out = mask_timeseg_road(
                        testmodel, m, xtest, ytest, sub, l,
                        min_idx, wsize, k, modes[1])
                    hist1['acc'][s][k-1]  = acc
                    hist1['loss'][s][k-1] = loss
                    hist1['out'][s][k-1]  = out

            hists.append(hist)
            hists1.append(hist1)

        with open(os.path.join(OUT_DIR, f'{m}_ts_{filler}abs_{modes[0][0]}rf.pickle'), 'wb') as f:
            pickle.dump(hists, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(OUT_DIR, f'{m}_ts_{filler}abs_{modes[1][0]}rf.pickle'), 'wb') as f:
            pickle.dump(hists1, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Clean up per-model MFBB cache now that results are saved
        cache_model_dir = os.path.join(MFBB_DIR, m, filler + 'abs')
        if os.path.isdir(cache_model_dir):
            shutil.rmtree(cache_model_dir)
