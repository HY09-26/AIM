"""Channel masking — zero replacement (MoRF / LeRF)."""
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
OUT_DIR  = os.path.join(SAVE_DIR, f"repeat{rep}/ch_test_zero/{dataname}")
os.makedirs(OUT_DIR, exist_ok=True)

modes    = [('mo', -1), ('le', 1)]
leg      = ['gradient', 'gradientxinput', 'smoothgrad', 'smoothgrad_sq',
            'vargrad', 'inte_grad', 'random']
n_ch     = cfg.kwerg['channels']
n_subs   = len(cfg.sub_list)
filler   = '' if args.abs_saliency else 'n'


def mask_channels_zero(model, xtest, chrank, labels):
    """Zero out ranked channels, evaluate model."""
    mixtures = xtest.squeeze().copy()
    for t in range(mixtures.shape[0]):
        mixtures[t, chrank[t]] = 0.0
    _, loader = get_loader(cfg.batch_size, mixtures, labels)
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

            hist  = dict(acc =np.zeros((n_subs, n_ch)),
                         loss=np.zeros((n_subs, n_ch)),
                         out =np.zeros((n_subs, n_ch, cfg.n_trials, cfg.n_classes_out)))
            hist1 = dict(acc =np.zeros((n_subs, n_ch)),
                         loss=np.zeros((n_subs, n_ch)),
                         out =np.zeros((n_subs, n_ch, cfg.n_trials, cfg.n_classes_out)))

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

                # sum saliency over time → per-channel score, shape (n_trials, n_ch)
                ch_sal = grads.sum(axis=-1)
                for k in range(1, n_ch + 1):
                    chrankM = np.argsort(ch_sal * modes[0][1], axis=-1)[:, :k]
                    chrankL = np.argsort(ch_sal * modes[1][1], axis=-1)[:, :k]

                    loss, acc, out = mask_channels_zero(testmodel, xtest, chrankM, ytest)
                    hist['acc'][s][k-1]  = acc
                    hist['loss'][s][k-1] = loss
                    hist['out'][s][k-1]  = out

                    loss, acc, out = mask_channels_zero(testmodel, xtest, chrankL, ytest)
                    hist1['acc'][s][k-1]  = acc
                    hist1['loss'][s][k-1] = loss
                    hist1['out'][s][k-1]  = out

            hists.append(hist)
            hists1.append(hist1)

        with open(os.path.join(OUT_DIR, f'{m}_ch_{filler}abs_{modes[0][0]}rf.pickle'), 'wb') as f:
            pickle.dump(hists, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(OUT_DIR, f'{m}_ch_{filler}abs_{modes[1][0]}rf.pickle'), 'wb') as f:
            pickle.dump(hists1, f, protocol=pickle.HIGHEST_PROTOCOL)
